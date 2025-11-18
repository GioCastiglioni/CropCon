import json
import os
from datetime import datetime

import numpy as np
import torch

from cropcon.datasets.base import RawGeoFMDataset

class SSL4EO(RawGeoFMDataset):
    def __init__(
        self,
        split: str,
        dataset_name: str,
        multi_modal: bool,
        multi_temporal: int,
        support_test: bool,
        root_path: str,
        classes: list,
        num_classes: int,
        ignore_index: int,
        img_size: int,
        bands: dict[str, list[str]],
        distribution: list[int],
        data_mean: dict[str, list[str]],
        data_std: dict[str, list[str]],
        data_min: dict[str, list[str]],
        data_max: dict[str, list[str]],
        download_url: str,
        auto_download: bool,
        fold_config: int
    ):
        """Initialize the PASTIS dataset.

        Args:
            split (str): split of the dataset (train, val).
            dataset_name (str): dataset name.
            multi_modal (bool): if the dataset is multi-modal.
            multi_temporal (int): number of temporal frames.
            root_path (str): root path of the dataset.
            classes (list): classes of the dataset.
            num_classes (int): number of classes.
            ignore_index (int): index to ignore for metrics and loss.
            img_size (int): size of the image.
            bands (dict[str, list[str]]): bands of the dataset.
            distribution (list[int]): class distribution.
            data_mean (dict[str, list[str]]): mean for each band for each modality.
            Dictionary with keys as the modality and values as the list of means.
            e.g. {"s2": [b1_mean, ..., bn_mean], "s1": [b1_mean, ..., bn_mean]}
            data_std (dict[str, list[str]]): str for each band for each modality.
            Dictionary with keys as the modality and values as the list of stds.
            e.g. {"s2": [b1_std, ..., bn_std], "s1": [b1_std, ..., bn_std]}
            data_min (dict[str, list[str]]): min for each band for each modality.
            Dictionary with keys as the modality and values as the list of mins.
            e.g. {"s2": [b1_min, ..., bn_min], "s1": [b1_min, ..., bn_min]}
            data_max (dict[str, list[str]]): max for each band for each modality.
            Dictionary with keys as the modality and values as the list of maxs.
            e.g. {"s2": [b1_max, ..., bn_max], "s1": [b1_max, ..., bn_max]}
            download_url (str): url to download the dataset.
            auto_download (bool): whether to download the dataset automatically.
            fold_config (int): configuration of folds to split the data
        """
        super(SSL4EO, self).__init__(
            split=split,
            dataset_name=dataset_name,
            multi_modal=multi_modal,
            multi_temporal=multi_temporal,
            support_test=support_test,
            root_path=root_path,
            classes=classes,
            num_classes=num_classes,
            ignore_index=ignore_index,
            img_size=img_size,
            bands=bands,
            distribution=distribution,
            data_mean=data_mean,
            data_std=data_std,
            data_min=data_min,
            data_max=data_max,
            download_url=download_url,
            auto_download=auto_download,
            fold_config=fold_config
        )
            
        self.modalities = ["s2"]
        self.nb_split = 1

        reference_date = "2019-12-08"
        self.reference_date = datetime(*map(int, reference_date.split("-")))

        with open(os.path.join(self.root_path, "s2a_manifest_geospatial_weighted.json"), 'r') as f:
            manifest_data = json.load(f)
        
        self.meta_samples = [
            sample for sample in manifest_data["samples"] 
            if sample["partition"] == split
        ]

        self.num_classes = 1 # NO LABEL

    def __getitem__(self, i: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Get the item at index i.

        Args:
            i (int): index of the item.

        Returns:
            dict[str, torch.Tensor | dict[str, torch.Tensor]]: output dictionary following the format
            {"image":
                {"optical": torch.Tensor},
            "target": torch.Tensor,
            "metadata": torch.Tensor}.
        """
        item_metadata = self.meta_samples[i]

        optical_ts = torch.from_numpy(np.load(item_metadata["npy_path"]))
        time_positions = torch.from_numpy(np.array(item_metadata["days_from_min"]))

        if self.multi_temporal == 1:
            # we only take the last frame
            optical_indexes = torch.Tensor([-1]).long()
            optical_ts = optical_ts[:, optical_indexes]

            metadata = torch.Tensor([time_positions[optical_indexes].float()])
        else:
            # select evenly spaced samples
            optical_indexes = torch.linspace(
                0, optical_ts.shape[1] - 1, self.multi_temporal, dtype=torch.long
            )
            optical_ts = optical_ts[:, optical_indexes]

            metadata = time_positions[optical_indexes].float()

        return {
            "image": {
                "optical": optical_ts.to(torch.float32),
            },
            "target": torch.empty(1,1),
            "metadata": metadata,
        }

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            int: length of the dataset.
        """
        return len(self.meta_samples)

    @staticmethod
    def download():
        pass
