import sys
import os
import timeit

import torch
from torch import optim
from torch.utils import data as torch_data

import wandb
import numpy as np

from utils import networks, datasets, loss_functions, evaluation, experiment_manager, parsers


def run_training(cfg: experiment_manager.CfgNode):
    cfg.MODEL.OUTPUT_PATH = cfg.PATHS.OUTPUT
    net = networks.PopulationDualTaskNet(cfg.MODEL)
    net.to(device)

    pretraining = cfg.CHANGE_DETECTION.PRETRAINING
    if pretraining.ENABLED:
        net.load_pretrained_encoder(pretraining.CFG_NAME, cfg.PATHS.OUTPUT, device, verbose=True)
        if pretraining.FREEZE_ENCODER:
            net.freeze_encoder()

    optimizer = optim.SGD(net.parameters(), lr=cfg.TRAINER.LR)
    criterion = loss_functions.get_criterion(cfg.MODEL.LOSS_TYPE)

    # unpacking cfg
    epochs = cfg.TRAINER.EPOCHS
    train_units = datasets.get_units(cfg.PATHS.DATASET, 'train')
    steps_per_epoch = len(train_units)

    # tracking variables
    global_step = epoch_float = 0

    # early stopping
    best_rmse_change_val, trigger_times = None, 0
    stop_training = False

    # evaluation
    eval_kwargs = {'max_units': cfg.TRAINER.EVAL_MAX_UNITS, 'verbose': False}

    _ = evaluation.model_change_evaluation_units(net, cfg, 'train', epoch_float, **eval_kwargs)
    _ = evaluation.model_change_evaluation_units(net, cfg, 'val', epoch_float, **eval_kwargs)

    for epoch in range(1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')

        start = timeit.default_timer()
        loss_set = []

        np.random.shuffle(train_units)
        for i_unit, train_unit in enumerate(train_units):
            dataset = datasets.BitemporalCensusUnitDataset(cfg=cfg, unit_nr=train_unit)
            dataloader_kwargs = {
                'batch_size': len(dataset),
                'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
                'shuffle': cfg.DATALOADER.SHUFFLE,
                'drop_last': False,
                'pin_memory': True,
            }
            dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)
            batch = next(iter(dataloader))

            net.train()
            optimizer.zero_grad()

            x_t1 = batch['x_t1'].to(device)
            x_t2 = batch['x_t2'].to(device)
            pred_change, pred_pop_t1, pred_pop_t2 = net(x_t1, x_t2)
            # need to be detached for backprop to work
            pred_pop_t1.detach(), pred_pop_t2.detach()
            pred_change = torch.sum(pred_change, dim=0)

            y_change, *_ = dataset.get_unit_labels()
            loss = criterion(pred_change, y_change.to(device).float())
            loss.backward()
            optimizer.step()

            loss_set.append(loss.item())

            global_step += 1
            epoch_float = global_step / steps_per_epoch

        # logging loss
        time = timeit.default_timer() - start
        wandb.log({
            'loss': np.mean(loss_set),
            'time': time,
            'step': global_step,
            'epoch': epoch_float,
        })

        # logging at the end of each epoch
        _ = evaluation.model_change_evaluation_units(net, cfg, 'train', epoch_float, **eval_kwargs)
        rmse_change_val = evaluation.model_change_evaluation_units(net, cfg, 'val', epoch_float, **eval_kwargs)

        if best_rmse_change_val is None or rmse_change_val < best_rmse_change_val:
            best_rmse_change_val = rmse_change_val
            wandb.log({
                'best val change rmse': best_rmse_change_val,
                'step': global_step,
                'epoch': epoch_float,
            })
            print(f'saving network (RMSE change {rmse_change_val:.3f})', flush=True)
            networks.save_checkpoint(net, optimizer, epoch, cfg)
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= cfg.TRAINER.PATIENCE:
                stop_training = True

        if stop_training:
            break  # end of training by early stopping

    net, *_ = networks.load_checkpoint(cfg, device)
    _ = evaluation.model_change_evaluation_units(net, cfg, 'test', epoch_float, **eval_kwargs)


if __name__ == '__main__':
    args = parsers.training_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)

    # make training deterministic
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('=== Runnning on device: p', device)

    wandb.init(
        name=cfg.NAME,
        config=cfg,
        entity='population_mapping',
        project=args.project,
        tags=['population', ],
        mode='online' if not cfg.DEBUG else 'disabled',
    )

    try:
        run_training(cfg)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
