import json
import os
from datetime import datetime
from typing import Union, Dict
import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from einops import rearrange

from cropcon.datasets.base import RawGeoFMDataset, temporal_subsampling


def prepare_dates(date_dict, reference_date):
    if isinstance(date_dict, str):
        date_dict = json.loads(date_dict)
    d = pd.DataFrame().from_dict(date_dict, orient="index")
    d = d[0].apply(
        lambda x: (
            datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
            - reference_date
        ).days
    )
    return torch.tensor(d.values)


class CropsChile(RawGeoFMDataset):
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
        fold_config: int,
        support_test: bool,
        reference_date: str = "2020-01-01",   # "YYYY-MM-DD" o vacío
        cover: float = 0.0,
        obj: str = "horticola",
    ):
        super().__init__(
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
            fold_config=fold_config,
            support_test=support_test,
            download_url=download_url,
            auto_download=auto_download,
        )

        meta_path = os.path.join(root_path, "metadata_full.geojson")
        self.meta_patch: gpd.GeoDataFrame = gpd.read_file(meta_path)

        with open(meta_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        id2dates = {feat["properties"]["id"]: feat["properties"].get("dates", [])
                    for feat in raw["features"]}
        self.meta_patch["dates"] = self.meta_patch["id"].map(id2dates)

                # Filtro por cobertura (en metadata es 'coverage')
        if cover and cover > 0:
            if "coverage" not in self.meta_patch.columns:
                raise KeyError("Column 'coverage' not found in metadata.")
            self.meta_patch = self.meta_patch[self.meta_patch["coverage"] > cover].copy()

        # Asegurar split_index
        if "split_index" not in self.meta_patch.columns:
            n = len(self.meta_patch)
            folds = np.tile(np.arange(1, 11), n // 10 + 1)[:n]
            self.meta_patch["split_index"] = folds

        assert split in ["train", "val", "test"], "Invalid split"
        if split == "train":
            folds = [1, 2, 3, 4, 5, 6]
        elif split == "val":
            folds = [7, 8]
        else:
            folds = [9, 10]

        # Modalidades esperadas según estructura en disco
        self.modalities = ["S2", "elevation", "mTPI", "landforms"]

        # Resolver reference_date: si no viene, usar mínima fecha observada en 'dates'
        if reference_date:
            self.reference_date = datetime(*map(int, reference_date.split("-")))
        else:
            if "dates" not in self.meta_patch.columns:
                raise KeyError("Column 'dates' not found; provide reference_date explicitly.")
            all_dates = pd.to_datetime(
                np.concatenate(self.meta_patch["dates"].values).ravel()
            )
            self.reference_date = pd.Timestamp(all_dates.min()).to_pydatetime()

        # Filtro por tipo de objetivo (usa columna 'original_folder')
        if obj not in {"horticola", "fruticola", "mixto"}:
            raise ValueError("obj must be one of {'horticola','fruticola','mixto'}")
        self.obj = obj

        if obj == "horticola":
            self.num_classes = 31
            self.meta_patch = self.meta_patch[
                self.meta_patch["original_folder"].str.contains("horticola", case=False, na=False)
            ].copy()
        elif obj == "fruticola":
            self.num_classes = 60
            self.meta_patch = self.meta_patch[
                self.meta_patch["original_folder"].str.contains("fruticola", case=False, na=False)
            ].copy()
        elif obj == "mixto":
            self.num_classes = 90
            # sin filtro adicional

        # Aplicar folds
        self.meta_patch = self.meta_patch[self.meta_patch["split_index"].isin(folds)].copy()

        # Reindexar desde 0
        self.meta_patch.sort_index(inplace=True)
        self.meta_patch.index = range(len(self.meta_patch))

        self.memory_dates = {}

        self.len = self.meta_patch.shape[0]
        self.id_patches = self.meta_patch.index

        # Construcción de tablas de fechas (solo S2) con respecto a reference_date
        # En metadata el campo es 'dates' (lista de strings YYYY-MM-DD)
        self.date_range = np.arange(-500, 3000, dtype=int)  # [-200, 599]
        self.date_tables: Dict[str, Dict[int, np.ndarray]] = {s: None for s in self.modalities}

        # Crear una vista dates_S2 unificada para no tocar el resto del código
        if "dates" not in self.meta_patch.columns:
            raise KeyError("Column 'dates' not found in metadata.")
        self.meta_patch["dates_S2"] = self.meta_patch["dates"]

        # Tabla binaria de presencia de fechas relativas
        date_table = pd.DataFrame(
            index=self.meta_patch.index, columns=self.date_range, dtype=int
        )
        date_table[:] = 0

        for pid, date_list in self.meta_patch["dates_S2"].items():
            if isinstance(date_list, (list, tuple, np.ndarray)):
                rel = pd.Series(date_list).apply(
                    lambda x: (datetime.strptime(x, "%Y-%m-%d") - self.reference_date).days
                )
                rel = rel[(rel >= self.date_range.min()) & (rel <= self.date_range.max())]
                if len(rel) > 0:
                    date_table.loc[pid, rel.values] = 1
            elif isinstance(date_list, str):
                # Soporte alternativo: string con fechas separadas por coma
                rel = pd.Series([s.strip() for s in date_list.split(",") if s.strip()]).apply(
                    lambda x: (datetime.strptime(x, "%Y-%m-%d") - self.reference_date).days
                )
                rel = rel[(rel >= self.date_range.min()) & (rel <= self.date_range.max())]
                if len(rel) > 0:
                    date_table.loc[pid, rel.values] = 1
            else:
                # Sin fechas válidas → fila queda en cero
                pass

        self.date_tables["S2"] = {
            idx: date_table.loc[idx].to_numpy(dtype=int) for idx in date_table.index
        }

    def __len__(self):
        return self.len

    def get_dates(self, id_patch: int, sat: str) -> torch.Tensor:
        """Devuelve las fechas relativas (días desde reference_date) donde hay observación."""
        table = self.date_tables.get(sat)
        if table is None:
            return torch.empty(0, dtype=torch.int32)
        row = table.get(id_patch)
        if row is None:
            return torch.empty(0, dtype=torch.int32)
        indices = np.where(row == 1)[0]
        if indices.size == 0:
            return torch.empty(0, dtype=torch.int32)
        # Mapear índices al valor real de día relativo
        rel_days = self.date_range[indices]
        return torch.tensor(rel_days, dtype=torch.int32)

    def __getitem__(self, i: int) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        line = self.meta_patch.iloc[i]
        id_patch = self.id_patches[i]
        name = line["patch_file"]
        Idi = name.split("_")[1]  # Asumiendo que el ID está en el nombre del archivo

        # Target
        target_path = os.path.join(self.root_path, f"ANNOTATIONS", f"ParcelIDs_{Idi}")
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Missing target file: {target_path}")
        target = torch.from_numpy(np.load(target_path))

        data: Dict[str, torch.Tensor] = {}
        metadata: Dict[str, torch.Tensor] = {}

        for modality in self.modalities:
            data_dir = f"DATA_{modality.upper()}"
            if modality == "S2":

                path = os.path.join(self.root_path, data_dir, f"S2_{Idi}")
            else:
                path = os.path.join(self.root_path, data_dir, f"{Idi}")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing data file for {modality}: {path}")

            array = np.load(path)

            if array.ndim == 4:
                # (T, C, H, W) → muestrear T a multi_temporal y reordenar a (C, T, H, W)
                if array.shape[0] < 1:
                    raise ValueError(f"{modality} has zero time steps for patch {name}")
                optical_whole_range_indexes = torch.linspace(
                    0, array.shape[0] - 1, self.multi_temporal, dtype=torch.long
                )
                idx = temporal_subsampling(
                    self.multi_temporal, optical_whole_range_indexes, [self.multi_temporal]
                )
                tensor = torch.from_numpy(array).to(torch.float32)[idx]
                tensor = rearrange(tensor, "t c h w -> c t h w")
            else:
                # Estático: (C, H, W)
                tensor = torch.from_numpy(array).to(torch.float32)[None,None,:,:]

            if modality == "S2":
                data["optical"] = tensor
                # Fechas correspondientes a los frames muestreados
                dates = self.get_dates(id_patch, modality)
                if dates.numel() > 0:

                    optical_whole_range_indexes = torch.linspace(
                        0, dates.shape[0] - 1, self.multi_temporal, dtype=torch.long
                    )
                    idx = temporal_subsampling(
                        self.multi_temporal, optical_whole_range_indexes, [self.multi_temporal]
                        )
                    metadata[modality] = dates[idx]
                else:
                    # Sin fechas → vector vacío (o ceros si prefieres)
                    metadata[modality] = torch.empty(0, dtype=torch.int32)
            else:
                data[modality] = tensor

        return {
            "image": data,
            "target": target.to(torch.int64),
            "metadata": metadata,
        }

    @staticmethod
    def download():
        pass