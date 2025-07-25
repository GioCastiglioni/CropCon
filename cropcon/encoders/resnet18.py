from logging import Logger

import torch
import torch.nn as nn
from torchvision import models

from cropcon.encoders.base import Encoder


class ResNet18(Encoder):

    def __init__(
        self,
        input_bands: dict[str, list[str]],
        input_size: int,
        download_url: str,
        multi_temporal: int,
        encoder_weights: str | None = None,
    ):
        super().__init__(
            model_name="ResNet18",
            encoder_weights=encoder_weights,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=0,
            output_dim=[64, 128, 256, 512],
            output_layers=[1,2,3,4],
            multi_temporal=False,
            multi_temporal_output=False,
            pyramid_output=False,
            download_url=download_url,
        )

        self.in_channels = len(input_bands["optical"])  # number of optical bands

        net = models.resnet18(weights=None)
        # Modify input conv to support custom input channels
        net.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.initial = nn.Sequential(
            net.conv1,   # output: 64 x H/2 x W/2
            net.bn1,
            net.relu,
        )
        self.maxpool = net.maxpool  # output: 64 x H/4 x W/4
        self.layer1 = net.layer1    # output: 64 x H/4 x W/4
        self.layer2 = net.layer2    # output: 128 x H/8 x W/8
        self.layer3 = net.layer3    # output: 256 x H/16 x W/16
        self.layer4 = net.layer4    # output: 512 x H/32 x W/32

    def forward(self, img):
        x = img["optical"]
        x0 = self.initial(x)     # 64, H/2
        x1 = self.maxpool(x0)    # 64, H/4
        x2 = self.layer1(x1)     # 64, H/4
        x3 = self.layer2(x2)     # 128, H/8
        x4 = self.layer3(x3)     # 256, H/16
        x5 = self.layer4(x4)     # 512, H/32
        return [x2, x3, x4, x5]

    def load_encoder_weights(self, logger: Logger, from_scratch: bool = False) -> None:
        if from_scratch or self.encoder_weights is None:
            logger.info("Training encoder from scratch.")
            return

        logger.info(f"Loading encoder weights from: {self.encoder_weights}")
        
        state_dict = torch.load(self.encoder_weights, map_location="cpu")["state_dict"]

        # Remove 'module.encoder_q.' prefix from keys
        new_state_dict = {
            k.replace("module.encoder_q.", ""): v
            for k, v in state_dict.items()
            if k.startswith("module.encoder_q.") and not k.startswith("module.encoder_q.fc")
        }

        # Construct a temporary torch model to use `load_state_dict`
        dummy_model = models.resnet18(weights=None)
        dummy_model.conv1 = nn.Conv2d(13, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Load weights
        missing_keys, unexpected_keys = dummy_model.load_state_dict(new_state_dict, strict=False)

        # Now copy weights layer by layer to your custom class
        self.initial[0].load_state_dict(dummy_model.conv1.state_dict())  # conv1
        self.initial[1].load_state_dict(dummy_model.bn1.state_dict())    # bn1
        # ReLU has no weights
        self.maxpool = dummy_model.maxpool
        self.layer1.load_state_dict(dummy_model.layer1.state_dict())
        self.layer2.load_state_dict(dummy_model.layer2.state_dict())
        self.layer3.load_state_dict(dummy_model.layer3.state_dict())
        self.layer4.load_state_dict(dummy_model.layer4.state_dict())

        logger.info("Encoder weights successfully loaded into ResNet18.")
