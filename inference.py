import torch
from pathlib import Path
import numpy as np
from utils import experiment_manager, networks, datasets, parsers, geofiles


def produce_population_grids(cfg: experiment_manager.CfgNode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()
    for year in range(2016, 2021):
        ds = datasets.PopInferenceDataset(cfg, year)
        arr = ds.get_pop_grid()
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

        transform, crs = ds.get_pop_grid_geo()
        file = Path(cfg.PATHS.OUTPUT) / 'inference' / 'population_grids' / f'pop_kigali_{year}_{cfg.NAME}.tif'
        geofiles.write_tif(file, arr, transform, crs)


def produce_unit_stats(cfg: experiment_manager.CfgNode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()

    data = {}

    for year in [2016, 2020]:
        ds = datasets.PopInferenceDataset(cfg, year, nonans=True)
        tracker = 0
        with torch.no_grad():
            for item in ds:
                x = item['x'].to(device)
                pred = net(x.unsqueeze(0)).cpu().item()
                gt = item['y']
                unit = int(item['unit'])
                if str(unit) not in data.keys():
                    data[str(unit)] = {}
                unit_data = data[str(unit)]
                if f'pred_pop{year}' in unit_data.keys():
                    unit_data[f'pred_pop{year}'] += pred
                    if not np.isnan(gt):
                        unit_data[f'gt_pop{year}'] += gt
                else:
                    unit_data[f'pred_pop{year}'] = pred
                    if not np.isnan(gt):
                        unit_data[f'gt_pop{year}'] = gt
                tracker += 1
                if tracker % 1_000 == 0:
                    print(tracker)

    out_file = Path(cfg.PATHS.OUTPUT) / 'inference' / 'unit_stats' / f'pop_kigali_{cfg.NAME}.json'
    geofiles.write_json(out_file, data)


if __name__ == '__main__':
    args = parsers.deployment_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)

    produce_unit_stats(cfg)
    # produce_population_grids(cfg, year)
