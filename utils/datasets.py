import torch
from pathlib import Path
from abc import abstractmethod
import numpy as np
import multiprocessing
from utils import augmentations, experiment_manager, geofiles
from affine import Affine


class AbstractPopDataset(torch.utils.data.Dataset):

    def __init__(self, cfg: experiment_manager.CfgNode, run_type: str):
        super().__init__()
        self.cfg = cfg
        self.root_path = Path(cfg.PATHS.DATASET)
        samples_file = self.root_path / 'samples.json'
        self.samples = geofiles.load_json(samples_file)

        self.sites = cfg.DATALOADER.SITES
        for site in self.sites:
            self.samples = [s for s in self.samples if s['site'] == site]

        self.run_type = run_type
        self.indices = [['B2', 'B3', 'B4', 'B8'].index(band) for band in cfg.DATALOADER.SPECTRAL_BANDS]
        self.season = cfg.DATALOADER.SEASON

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def _get_s2_patch(self, site: str, year: int, season: str, i: int, j: int) -> np.ndarray:
        file = self.root_path / site / str(year) / f's2_{year}_{season}_{i:03d}_{j:03d}.tif'
        img, _, _ = geofiles.read_tif(file)
        img = img[:, :, self.indices]
        return img.astype(np.float32)

    def _get_patch_geo(self, site: str, i: int, j: int) -> tuple:
        file = self.root_path / site / '2016' / f's2_2016_wet_{i:03d}_{j:03d}.tif'
        _, transform, crs = geofiles.read_tif(file)
        return transform, crs

    def _get_pop_label(self, site: str, year: int, i: int, j: int) -> float:
        for s in self.metadata:
            if s['site'] == site and s['year'] == year and s['i'] == i and s['j'] == j:
                return float(s['pop'])
        raise Exception('sample not found')

    def get_pop_grid_geo(self, site: str, resolution: int = 100) -> tuple:
        file = self.root_path / site / '2016' / f's2_2016_wet_{0:03d}_{0:03d}.tif'
        _, s2_transform, crs = geofiles.read_tif(file)
        _, _, x_origin, _, _, y_origin, *_ = s2_transform
        pop_transform = (x_origin, resolution, 0.0, y_origin, 0.0, -resolution)
        pop_transform = Affine.from_gdal(*pop_transform)
        return pop_transform, crs

    def get_pop_grid(self, site: str) -> np.ndarray:
        site_samples = [s for s in self.samples if s['site'] == site]
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

    def __init__(self, cfg: experiment_manager.CfgNode, run_type: str, no_augmentations: bool = False,
                 disable_unlabeled: bool = False):
        super().__init__(cfg, run_type)

        # handling transformations of data
        self.no_augmentations = no_augmentations
        self.transform = augmentations.compose_transformations(cfg.AUGMENTATION, no_augmentations)

        # subset samples
        self.samples = [s for s in self.samples if not bool(s['isnan'])]
        if run_type == 'training':
            self.samples = [s for s in self.samples if s['random'] <= self.cfg.DATALOADER.SPLIT]
        if run_type == 'validation':
            self.samples = [s for s in self.samples if s['random'] > self.cfg.DATALOADER.SPLIT]

        # unlabeled data for semi-supervised learning
        if (cfg.DATALOADER.INCLUDE_UNLABELED):
            pass

        manager = multiprocessing.Manager()
        self.samples = manager.list(self.samples)

        self.length = len(self.samples)

    def __getitem__(self, index):

        s = self.samples[index]
        site, year, i, j = s['site'], s['year'], s['i'], s['j']

        y = s['pop']

        if self.season == 'wet' or self.season == 'dry':
            season = self.season
        else:
            season = 'wet' if np.random.rand(1) > 0.5 else 'dry'
        img = self._get_s2_patch(site, year, season, i, j)

        x = self.transform(img)

        item = {
            'x': x,
            'y': torch.tensor([y]),
            'site': site,
            'year': year,
            'season': season,
            'i': i,
            'j': j,
        }

        return item

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'


# dataset for urban extraction with building footprints
class PopInferenceDataset(AbstractPopDataset):

    def __init__(self, cfg: experiment_manager.CfgNode):
        super().__init__(cfg, 'inference')

        # handling transformations of data
        self.no_augmentations = True
        self.transform = augmentations.compose_transformations(cfg.AUGMENTATION, self.no_augmentations)

        manager = multiprocessing.Manager()
        self.samples = manager.list(self.samples)

        self.length = len(self.samples)

    def __getitem__(self, index):

        s = self.samples[index]
        site, year, i, j, isnan, pop = s['site'], s['year'], s['i'], s['j'], bool(s['isnan']), float(s['pop'])

        img = self._get_s2_patch(site, year, 'wet', i, j)

        # # resampling images to desired patch size
        # if img.shape[0] != self.patch_size or img.shape[1] != self.patch_size:
        #     img = cv2.resize(img, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)

        x = self.transform(img)

        item = {
            'x': x,
            'y': np.nan if isnan else pop,
            'site': site,
            'year': year,
            'i': i,
            'j': j,
        }

        return item

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'
