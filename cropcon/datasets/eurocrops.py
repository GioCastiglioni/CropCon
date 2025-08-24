import json
import os
from pathlib import Path
from datetime import datetime
from typing import Union, Dict
import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from einops import rearrange

from cropcon.datasets.base import RawGeoFMDataset, temporal_subsampling


class EuroCrops(RawGeoFMDataset):
    def __init__(
        self,
        split: str,
        dataset_name: str,
        multi_modal: bool,
        multi_temporal: int,
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
        support_test: bool,
        fold_config: int,
        reference_date: str = "2021-10-05",
        cover=0,
    ):
        """Initializes the dataset.

        Args:
            split (str): split of the dataset (train, val, test)
            dataset_name (str): dataset name
            multi_modal (bool): whether the dataset is multi_modal
            multi_temporal (int): number of temporal frames
            root_path (str): root path of the dataset
            classes (list): dataset classes names
            num_classes (int): number of classes
            ignore_index (int): index to ignore
            img_size (int): dataset's image size
            bands (dict[str, list[str]]): bands of the dataset
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
        """
        super(EuroCrops, self).__init__(
            split=split,
            dataset_name=dataset_name,
            multi_modal=multi_modal,
            multi_temporal=multi_temporal,
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
            support_test=support_test,
            fold_config=fold_config,
            auto_download=auto_download
        )

        if multi_modal:
            raise Exception("Multimodal not implemented")
        if split not in ["train", "val", "test"]:
            raise Exception("Not recognized split argument")

        self.root_path = Path(root_path)
        metadata_path = self.root_path / "metadata.geojson"
        annotations_path = self.root_path / "ANNOTATIONS"
        s2_path = self.root_path / "DATA_S2"
        assert metadata_path.exists(), f"metadata.geojson not found {metadata_path}"
        assert annotations_path.exists(), f"annotations directory not found {annotations_path}"
        assert s2_path.exists(), f"S2 directory not found{s2_path}"

        self.modalities = ["S2",]
        self.reference_date = pd.to_datetime(reference_date)
        # self.num_classes = 283

        self.meta_patch = (
                gpd.read_file(metadata_path)
                .astype({
                    "id": int,
                    "patch_n": int,
                })
                .set_index("id").sort_index()
        )

        with open(self.root_path / "splits.json") as f:
            splits = json.load(f)
        split_patches = np.concat([
            split_by_tiles[self.split]
            for split_by_tiles in splits.values()
            ]).astype(int)
        self.meta_patch = self.meta_patch[
            self.meta_patch.index.isin(split_patches)
        ]
        if cover > 0:
            self.meta_patch = self.meta_patch[
                self.meta_patch["parcel_cover"] > cover
                ]
        self.len = self.meta_patch.shape[0]
        self.id_patches = self.meta_patch.index

    def __len__(self):
        return self.len

    def prepare_dates(self, date_dict):
        """Date formating."""
        if type(date_dict) is str:
            date_dict = json.loads(date_dict)
        d = pd.DataFrame().from_dict(date_dict, orient="index")
        d = d[0].apply(lambda x: (pd.to_datetime(x) - self.reference_date).days)
        return torch.tensor(d.values)

    def __getitem__(self, i: int) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]]:
        line = self.meta_patch.iloc[i]
        id_patch = self.id_patches[i]
        name = "{:05}".format(id_patch)

        target = torch.from_numpy(
            np.load(
                self.root_path / f"ANNOTATIONS/ParcelIDs_{name}.npy"
                )
            )
        output = {"target": target, "name": name}
        for modality in self.modalities:
            path = self.root_path / f"DATA_{modality}" / f"{modality}_{name}.npy"
            array = np.load(path)
            assert array.ndim == 4, f"Unsupported array shape {array.shape}"
            output[modality] = torch.from_numpy(array).to(torch.float32)
            output[f"{modality}_dates"] = self.prepare_dates(line[f"dates_{modality}"])

        optical_ts = rearrange(output["S2"], "t c h w -> c t h w")
        if self.multi_temporal == 1:
            # we only take the last frame
            optical_indexes = torch.Tensor([-1]).long()
            optical_ts = optical_ts[:, optical_indexes]

            metadata = torch.Tensor([output["S2_dates"][optical_indexes].float()])
        else:
            # select evenly spaced samples
            optical_whole_range_indexes = torch.linspace(
                0, optical_ts.shape[1] - 1, min(35, optical_ts.shape[1]), dtype=torch.long
            )
            optical_indexes = temporal_subsampling(
                self.multi_temporal, optical_whole_range_indexes
                )

            optical_ts = optical_ts[:, optical_indexes]

            metadata = output["S2_dates"][optical_indexes].float()



        return {
            "image": {
                "optical": optical_ts,
            },
            "target": output["target"].to(torch.int64),
            "metadata": metadata  # solo fechas de S2
        }

    @staticmethod
    def download():
        pass
