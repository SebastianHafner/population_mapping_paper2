import torch
from torch.utils import data as torch_data
import wandb
from utils import datasets, metrics, networks, experiment_manager
import numpy as np
from scipy import stats


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


def model_evaluation_unit(net: networks.PopulationChangeNet, cfg: experiment_manager.CfgNode, run_type: str,
                          epoch: float, step: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    measurer_change = RegressionEvaluation()
    measurer_t1 = RegressionEvaluation()
    measurer_t2 = RegressionEvaluation()

    for training_unit in training_units:
        dataset = datasets.BitemporalCensusUnitDataset(cfg=cfg, unit_nr=int(training_unit))
        dataloader_kwargs = {
            'batch_size': cfg.TRAINER.BATCH_SIZE,
            'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
            'shuffle': cfg.DATALOADER.SHUFFLE,
            'drop_last': False,
            'pin_memory': True,
        }
        dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)
        pred_change = pred_t1 = pred_t2 = 0

        for i, batch in enumerate(dataloader):
            net.train()
            optimizer.zero_grad()

            x_t1 = batch['x_t1'].to(device)
            x_t2 = batch['x_t2'].to(device)
            pred_change, pred_t1, pred_t2 = net(x_t1, x_t2)

        y = dataset.get_label()
        y_change, y_t1, y_t2 = y['y_diff'].to(device), y['y_t1'].to(device), y['y_t2'].to(devce)
        pred_change = torch.sum(pred_change, dim=0)
        pred_t1 = torch.sum(pred_t1, dim=0)
        pred_t2 = torch.sum(pred_t2, dim=0)
        measurer_change.add_sample_torch(pred_change, y_change)
        measurer_t1.add_sample_torch(pred_t1, y_t1)
        measurer_t2.add_sample_torch(pred_t2, y_t2)

    # assessment
    for measurer, name in zip([measurer_change, measurer_t1, measurer_t2], ['diff', 'pop_t1', 'pop_t2']):
        rmse = measurer.root_mean_square_error()
        print(f'RMSE {run_type} {name} {rmse:.3f}')
        wandb.log({
            f'{run_type} {name} rmse': rmse,
            'step': step,
            'epoch': epoch,
        })