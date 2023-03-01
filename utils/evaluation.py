import torch
from torch.utils import data as torch_data
import wandb
from utils import datasets, networks, experiment_manager, dataset_helpers
import numpy as np
from scipy import stats
import sys


class RegressionEvaluation(object):
    def __init__(self):
        self.predictions = []
        self.labels = []

    def add_sample_numpy(self, pred: np.ndarray, label: np.ndarray):
        self.predictions.extend(pred.flatten())
        self.labels.extend(label.flatten())

    def add_sample_torch(self, pred: torch.tensor, label: torch.tensor):
        pred = pred.float().detach().cpu().numpy()
        label = label.float().detach().cpu().numpy()
        self.add_sample_numpy(pred, label)

    def reset(self):
        self.predictions = []
        self.labels = []

    def root_mean_square_error(self) -> float:
        return np.sqrt(np.sum(np.square(np.array(self.predictions) - np.array(self.labels))) / len(self.labels))

    def r_square(self) -> float:
        slope, intercept, r_value, p_value, std_err = stats.linregress(self.labels, self.predictions)
        return r_value


def model_evaluation(net: networks.PopulationNet, cfg: experiment_manager.CfgNode, run_type: str, epoch: float,
                          step: int, max_samples: int = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    measurer = RegressionEvaluation()
    dataset = datasets.PopDataset(cfg, run_type, no_augmentations=True)
    dataloader_kwargs = {
        'batch_size': 1,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle': True,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    max_samples = len(dataset) if max_samples is None else max_samples
    counter = 0

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            img = batch['x'].to(device)
            label = batch['y'].to(device)
            pred = net(img)
            measurer.add_sample_torch(pred, label)
            counter += 1
            if counter == max_samples or cfg.DEBUG:
                break

    # assessment
    rmse = measurer.root_mean_square_error()
    print(f'RMSE {run_type} {rmse:.3f}')
    wandb.log({
        f'{run_type} rmse': rmse,
        'step': step,
        'epoch': epoch,
    })

    return rmse


def model_evaluation_units(net: networks.PopulationChangeNet, cfg: experiment_manager.CfgNode, run_type: str,
                           epoch: float, step: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    measurer_change = RegressionEvaluation()
    measurer_t1 = RegressionEvaluation()
    measurer_t2 = RegressionEvaluation()

    units = dataset_helpers.get_units(cfg.PATHS.DATASET, run_type)

    for i_unit, unit in enumerate(units):
        dataset = datasets.BitemporalCensusUnitDataset(cfg=cfg, unit_nr=int(unit), no_augmentations=True)
        dataloader_kwargs = {
            'batch_size': cfg.TRAINER.BATCH_SIZE,
            'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
            'shuffle': False,
            'drop_last': False,
            'pin_memory': True,
        }
        dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)
        pred_change = pred_t1 = pred_t2 = 0

        for i, batch in enumerate(dataloader):
            x_t1 = batch['x_t1'].to(device)
            x_t2 = batch['x_t2'].to(device)
            with torch.no_grad():
                pred_change, pred_t1, pred_t2 = net(x_t1, x_t2)

        y = dataset.get_label()
        y_change, y_t1, y_t2 = y['y_diff'].to(device), y['y_t1'].to(device), y['y_t2'].to(device)
        pred_change = torch.sum(pred_change, dim=0).detach()
        pred_t1 = torch.sum(pred_t1, dim=0).detach()
        pred_t2 = torch.sum(pred_t2, dim=0).detach()
        measurer_change.add_sample_torch(pred_change, y_change)
        measurer_t1.add_sample_torch(pred_t1, y_t1)
        measurer_t2.add_sample_torch(pred_t2, y_t2)

        unit_str = f'{i_unit + 1:03d}/{len(units)}: Unit {unit} ({len(dataset)})'
        results_str = f'Pred: {pred_change.cpu().item():.0f}; GT: {y_change.cpu().item():.0f}'
        sys.stdout.write("\r%s" % f'Eval ({run_type})' + ' ' + unit_str + ' ' + results_str)
        sys.stdout.flush()

        # if i_unit:
        #     break

    # assessment
    for measurer, name in zip([measurer_change, measurer_t1, measurer_t2], ['diff', 'pop_t1', 'pop_t2']):
        rmse = measurer.root_mean_square_error()
        r2 = measurer.r_square()
        wandb.log({
            f'{run_type} {name} rmse': rmse,
            f'{run_type} {name} r2': r2,
            'step': step,
            'epoch': epoch,
        })

        if name == 'diff':
            eval_str = f'RMSE: {rmse:.0f}; R2: {r2:.2f}'
            sys.stdout.write("\r%s" % f'Eval ({run_type})' + ' ' + eval_str + '\n')
            sys.stdout.flush()


def model_change_evaluation_units(net: networks.PopulationChangeNet, cfg: experiment_manager.CfgNode, run_type: str,
                           epoch: float, step: int, verbose: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    measurer_change_ete = RegressionEvaluation()  # end-to-end
    measurer_change_pc = RegressionEvaluation()  # post classification
    measurer_t1 = RegressionEvaluation()
    measurer_t2 = RegressionEvaluation()

    units = dataset_helpers.get_units(cfg.PATHS.DATASET, run_type)

    for i_unit, unit in enumerate(units):
        dataset = datasets.BitemporalCensusUnitDataset(cfg=cfg, unit_nr=int(unit), no_augmentations=True)
        dataloader_kwargs = {
            'batch_size': cfg.TRAINER.BATCH_SIZE,
            'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
            'shuffle': False,
            'drop_last': False,
            'pin_memory': True,
        }
        dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)
        pred_change_ete = pred_t1 = pred_t2 = 0

        for i, batch in enumerate(dataloader):
            x_t1 = batch['x_t1'].to(device)
            x_t2 = batch['x_t2'].to(device)
            with torch.no_grad():
                pred_change_ete, pred_t1, pred_t2 = net(x_t1, x_t2)

        y = dataset.get_label()
        y_change, y_t1, y_t2 = y['y_diff'].to(device), y['y_t1'].to(device), y['y_t2'].to(device)
        pred_change_ete = torch.sum(pred_change_ete, dim=0).detach()
        pred_t1 = torch.sum(pred_t1, dim=0).detach()
        pred_t2 = torch.sum(pred_t2, dim=0).detach()
        pred_change_pc = pred_t2 - pred_t1
        measurer_change_ete.add_sample_torch(pred_change_ete, y_change)
        measurer_change_pc.add_sample_torch(pred_change_pc, y_change)
        measurer_t1.add_sample_torch(pred_t1, y_t1)
        measurer_t2.add_sample_torch(pred_t2, y_t2)

        if verbose:
            unit_str = f'{i_unit + 1:03d}/{len(units)}: Unit {unit} ({len(dataset)})'
            results_str = f'Pred ETE: {pred_change_ete.cpu().item():.0f}; Pred PC: {pred_change_pc.cpu().item():.0f}; GT: {y_change.cpu().item():.0f}'
            sys.stdout.write("\r%s" % f'Eval ({run_type})' + ' ' + unit_str + ' ' + results_str)
            sys.stdout.flush()

        # if i_unit:
        #     break

    # assessment
    return_value = None
    for measurer, name in zip([measurer_change_ete, measurer_change_pc, measurer_t1, measurer_t2],
                              ['diff', 'diff_pc', 'pop_t1', 'pop_t2']):
        rmse = measurer.root_mean_square_error()
        r2 = measurer.r_square()
        wandb.log({
            f'{run_type} {name} rmse': rmse,
            f'{run_type} {name} r2': r2,
            'step': step,
            'epoch': epoch,
        })

        if name == 'diff':
            return_value = rmse
            if verbose:
                eval_str = f'RMSE: {rmse:.0f}; R2: {r2:.2f}'
                sys.stdout.write("\r%s" % f'Eval ({run_type})' + ' ' + eval_str + '\n')
                sys.stdout.flush()

    return return_value