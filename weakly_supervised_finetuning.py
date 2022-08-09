import sys
import os
import timeit

import torch
from torch import optim
from torch.utils import data as torch_data

import wandb
import numpy as np

from utils import networks, datasets, loss_functions, evaluation, experiment_manager, parsers, dataset_helpers


def run_training(cfg):
    cfg.MODEL.OUTPUT_PATH = cfg.PATHS.OUTPUT
    net = networks.PopulationChangeNet(cfg.MODEL)
    net.to(device)

    if cfg.PRETRAINING.ENABLED:
        # TODO: load optimizer parameters for pretrained network part
        pretrained_weights = networks.load_weights_finetuning(cfg.PATHS.OUTPUT, cfg.PRETRAINING.CFG_NAME,
                                                              cfg.PRETRAINING.EPOCH, device)
        net.encoder.load_state_dict(pretrained_weights)

    optimizer = optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)

    criterion = loss_functions.get_criterion(cfg.MODEL.LOSS_TYPE)

    # unpacking cfg
    epochs = cfg.TRAINER.EPOCHS
    save_checkpoints = cfg.SAVE_CHECKPOINTS

    training_units = dataset_helpers.get_units(cfg.PATHS.DATASET, 'training')

    # tracking variables
    global_step = epoch_float = 0
    steps_per_epoch = len(training_units)

    if not cfg.DEBUG:
        evaluation.model_evaluation_units(net, cfg, 'training', epoch_float, global_step)
        evaluation.model_evaluation_units(net, cfg, 'test', epoch_float, global_step)

    for epoch in range(1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')

        start = timeit.default_timer()
        loss_set, loss_set_change, loss_set_t1, loss_set_t2 = [], [], [], []

        np.random.shuffle(training_units)
        for i_unit, training_unit in enumerate(training_units):
            dataset = datasets.BitemporalCensusUnitDataset(cfg=cfg, unit_nr=int(training_unit))
            print(f'{i_unit + 1:03d}/{len(training_units)}: Unit {training_unit} ({len(dataset)})')

            dataloader_kwargs = {
                'batch_size': len(dataset),
                'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
                'shuffle': cfg.DATALOADER.SHUFFLE,
                'drop_last': False,
                'pin_memory': True,
            }
            dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)
            pred_change = pred_t1 = pred_t2 = 0

            for batch in dataloader:

                net.train()
                optimizer.zero_grad()

                x_t1 = batch['x_t1'].to(device)
                x_t2 = batch['x_t2'].to(device)
                pred_change, pred_t1, pred_t2 = net(x_t1, x_t2)

            y = dataset.get_label()
            y_change, y_t1, y_t2 = y['y_diff'].to(device), y['y_t1'].to(device), y['y_t2'].to(device)
            pred_change = torch.sum(pred_change, dim=0)
            pred_t1 = torch.sum(pred_t1, dim=0)
            pred_t2 = torch.sum(pred_t2, dim=0)

            loss_change = criterion(pred_change, y_change.float())
            loss_t1 = criterion(pred_t1, y_t1.float())
            loss_t2 = criterion(pred_t2, y_t2.float())
            loss = loss_change + loss_t1 + loss_t2
            loss.backward()
            optimizer.step()

            loss_set_change.append(loss_change.item())
            loss_set_t1.append(loss_t1.item())
            loss_set_t2.append(loss_t2.item())
            loss_set.append(loss.item())

            global_step += 1
            epoch_float = global_step / steps_per_epoch
            # end of unit (step)

        assert (epoch == epoch_float)
        print(f'epoch float {epoch_float} (step {global_step}) - epoch {epoch}')

        # logging at the end of each epoch
        evaluation.model_evaluation_units(net, cfg, 'training', epoch_float, global_step)
        evaluation.model_evaluation_units(net, cfg, 'test', epoch_float, global_step)

        # logging
        time = timeit.default_timer() - start
        wandb.log({
            'loss_diff': np.mean(loss_set_change),
            'loss_pop_t1': np.mean(loss_set_t1),
            'loss_pop_t2': np.mean(loss_set_t2),
            'loss': np.mean(loss_set),
            'time': time,
            'step': global_step,
            'epoch': epoch_float,
        })

        if epoch in save_checkpoints and not cfg.DEBUG:
            print(f'saving network', flush=True)
            networks.save_checkpoint(net, optimizer, epoch, global_step, cfg)


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
        project='paper2_debug',
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
