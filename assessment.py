import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from utils import experiment_manager, networks, datasets, parsers, geofiles


def total_population(cfg: experiment_manager.CfgNode, run_type: str = 'all'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()
    ds = datasets.PopDataset(cfg, run_type, no_augmentations=True, disable_unlabeled=True)
    y_gt, y_pred = 0, 0

    with torch.no_grad():
        for item in ds:
            x = item['x'].to(device)
            y = item['y']
            pred = net(x.unsqueeze(0))
            y_gt += y.item()
            y_pred += pred.cpu().item()

    print(y_gt, y_pred)


def produce_population_grid(cfg: experiment_manager.CfgNode, site: str = 'kigali', year: int = 2016):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()
    ds = datasets.PopInferenceDataset(cfg)
    arr = ds.get_pop_grid(site)
    tracker = 0
    with torch.no_grad():
        for item in ds:
            x = item['x'].to(device)
            pred = net(x.unsqueeze(0)).cpu().item()
            gt = item['y']
            i, j = item['i'], item['j']
            arr[i, j, 0] = float(pred)
            arr[i, j, 1] = gt
            tracker += 1
            if tracker % 1_000 == 0:
                print(tracker)

    transform, crs = ds.get_pop_grid_geo(site)
    file = Path(cfg.PATHS.OUTPUT) / 'inference' / 'population_grids' / f'pop_{site}_{year}.tif'
    geofiles.write_tif(file, arr, transform, crs)


if __name__ == '__main__':
    args = parsers.assessment_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    # total_population(cfg, run_type=args.run_type)
    produce_population_grid(cfg)
