import torch
from pathlib import Path
from abc import abstractmethod
import numpy as np
import multiprocessing
from utils import augmentations, experiment_manager, geofiles
from affine import Affine


class AbstractPopDataset(torch.utils.data.Dataset):

    def __init__(self, cfg: experiment_manager.CfgNode, unit_nr: int = None):
        super().__init__()
        self.cfg = cfg
        self.root_path = Path(cfg.PATHS.DATASET)
        self.unit_nr = unit_nr
        metadata_fname = 'metadata.json' if unit_nr is None else f'metadata_unit{unit_nr}.json'
        self.metadata = geofiles.load_json(self.root_path / metadata_fname)
        self.samples = self.metadata['samples']
        self.indices = [['B2', 'B3', 'B4', 'B8'].index(band) for band in cfg.DATALOADER.SPECTRAL_BANDS]
        self.year = cfg.DATALOADER.YEAR
        self.season = cfg.DATALOADER.SEASON

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def _get_s2_img(self, year: int, season: str) -> np.ndarray:
        file = self.root_path / 'kigali' / 's2' / f's2_{year}_{season}.tif'
        img, geo_transform, crs = geofiles.read_tif(file)
        self.geo_transform = geo_transform
        self.crs = crs
        img = img[:, :, self.indices]
        return img.astype(np.float32)

    def _get_unit_pop(self, unit_nr: int, year: int) -> int:
        return int(self.metadata['census'][str(unit_nr)][f'pop{year}'])

    def _get_unit_popgrowth(self, unit_nr: int) -> int:
        return int(self.metadata['census'][str(unit_nr)]['difference'])

    def _get_unit_split(self, unit_nr: int) -> str:
        return str(self.metadata['census'][str(unit_nr)][f'split'])

    def _get_pop_label(self, site: str, year: int, i: int, j: int) -> float:
        for s in self.metadata:
            if s['site'] == site and s['year'] == year and s['i'] == i and s['j'] == j:
                return float(s['pop'])
        raise Exception('sample not found')

    def get_pop_grid_geo(self, resolution: int = 100) -> tuple:
        _, _, x_origin, _, _, y_origin, *_ = self.geo_transform
        pop_transform = (x_origin, resolution, 0.0, y_origin, 0.0, -resolution)
        pop_transform = Affine.from_gdal(*pop_transform)
        return pop_transform, self.crs

    def get_pop_grid(self) -> np.ndarray:
        site_samples = [s for s in self.samples if s['site'] == 'kigali']
        m = max([s['i'] for s in site_samples]) + 1
        n = max([s['j'] for s in site_samples]) + 1
        arr = np.full((m, n, 2), fill_value=np.nan, dtype=np.float32)
        return arr

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'


# dataset for urban extraction with building footprints
class PopDataset(AbstractPopDataset):

    def __init__(self, cfg: experiment_manager.CfgNode, run_type: str, no_augmentations: bool = False):
        super().__init__(cfg)

        # handling transformations of data
        self.no_augmentations = no_augmentations
        self.transform = augmentations.compose_transformations(cfg.AUGMENTATION, no_augmentations)

        # subset samples
        self.run_type = run_type
        self.samples = [s for s in self.samples if not bool(s['isnan']) and s['split'] == run_type]

        manager = multiprocessing.Manager()
        self.samples = manager.list(self.samples)

        self.img = self._get_s2_img(self.year, self.season)

        self.length = len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]
        i, j, unit = s['i'], s['j'], s['unit']

        i_start, i_end = i * 10, (i + 1) * 10
        j_start, j_end = j * 10, (j + 1) * 10
        patch = self.img[i_start:i_end, j_start:j_end, ]
        x = self.transform(patch)

        y = s[f'pop{self.year}']

        item = {
            'x': x,
            'y': torch.tensor([y]),
            'year': self.year,
            'season': self.season,
            'unit': unit,
            'i': i,
            'j': j,
        }

        return item

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'


# dataset for urban extraction with building footprints
class BitemporalCensusUnitDataset(AbstractPopDataset):

    def __init__(self, cfg: experiment_manager.CfgNode, unit_nr: int, no_augmentations: bool = False):
        super().__init__(cfg, unit_nr)

        # handling transformations of data
        self.no_augmentations = no_augmentations
        self.transform = augmentations.compose_transformations(cfg.AUGMENTATION, no_augmentations)

        manager = multiprocessing.Manager()
        self.samples = manager.list(self.samples)

        self.t1, self.t2 = 2016, 2020
        self.img_t1 = self._get_s2_img(self.t1, self.season)
        self.img_t2 = self._get_s2_img(self.t2, self.season)

        self.length = len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]
        i, j, unit = s['i'], s['j'], s['unit']

        i_start, i_end = i * 10, (i + 1) * 10
        j_start, j_end = j * 10, (j + 1) * 10
        patch_t1 = self.img_t1[i_start:i_end, j_start:j_end, ]
        patch_t2 = self.img_t2[i_start:i_end, j_start:j_end, ]
        x_t1 = self.transform(patch_t1)
        x_t2 = self.transform(patch_t2)

        item = {
            'x_t1': x_t1,
            'x_t2': x_t2,
            'season': self.season,
            'unit': unit,
            'i': i,
            'j': j,
        }

        return item

    def get_label(self) -> dict:
        pop_t1 = self._get_unit_pop(self.unit_nr, self.t1)
        pop_t2 = self._get_unit_pop(self.unit_nr, self.t2)
        diff = self._get_unit_popgrowth(self.unit_nr)

        label = {
            'y_t1': torch.tensor([pop_t1]),
            'y_t2': torch.tensor([pop_t2]),
            'y_diff': torch.tensor([diff]),
        }

        return label

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'


# dataset for urban extraction with building footprints
class PopInferenceDataset(AbstractPopDataset):

    def __init__(self, cfg: experiment_manager.CfgNode, year: int, nonans: bool = False):
        super().__init__(cfg)

        # handling transformations of data
        self.no_augmentations = True
        self.transform = augmentations.compose_transformations(cfg.AUGMENTATION, self.no_augmentations)

        if nonans:
            self.samples = [s for s in self.samples if s['unit'] != 0]
        manager = multiprocessing.Manager()
        self.samples = manager.list(self.samples)

        self.year = year
        self.img = self._get_s2_img(self.year, self.season)

        self.length = len(self.samples)

    def __getitem__(self, index):

        s = self.samples[index]
        i, j, unit, isnan = s['i'], s['j'], s['unit'], bool(s['isnan'])

        i_start, i_end = i * 10, (i + 1) * 10
        j_start, j_end = j * 10, (j + 1) * 10
        patch = self.img[i_start:i_end, j_start:j_end, ]
        x = self.transform(patch)

        y = s[f'pop{self.year}'] if f'pop{self.year}' in s.keys() else np.nan

        item = {
            'x': x,
            'y': np.nan if isnan else y,
            'unit': unit,
            'i': i,
            'j': j,
        }

        return item

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'
