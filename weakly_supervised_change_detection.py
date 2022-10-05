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
    net = networks.PopulationDualTaskNet(cfg.MODEL)
    net.to(device)

    pretraining = cfg.CHANGE_DETECTION.PRETRAINING
    if pretraining.ENABLED:
        pretrained_weights = networks.load_weights_finetuning(cfg.PATHS.OUTPUT, pretraining.CFG_NAME,
                                                              pretraining.EPOCH, device)
        net.encoder.load_state_dict(pretrained_weights)

    if pretraining.FREEZE_ENCODER:
        for layer_name, param in net.encoder.named_parameters():
            # print(layer_name)
            param.requires_grad = False

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
        # evaluation.model_evaluation_units(net, cfg, 'training', epoch_float, global_step)
        evaluation.model_evaluation_units(net, cfg, 'test', epoch_float, global_step)

    # dummy_tensor = torch.rand((2, 4, 10, 10)).to(device)
    # with torch.no_grad():
    #     a, b, c = net(dummy_tensor, dummy_tensor)
    #     print(f'{torch.sum(a).cpu().item():.5f} - {torch.sum(b).cpu().item():.5f} - {torch.sum(c).cpu().item():.5f}')

    for epoch in range(1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')

        start = timeit.default_timer()
        loss_set = []

        np.random.shuffle(training_units)
        for i_unit, training_unit in enumerate(training_units):

            dataset = datasets.BitemporalCensusUnitDataset(cfg=cfg, unit_nr=int(training_unit))

            dataloader_kwargs = {
                'batch_size': len(dataset),
                'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
                'shuffle': cfg.DATALOADER.SHUFFLE,
                'drop_last': False,
                'pin_memory': True,
            }
            dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)
            y = dataset.get_label()
            y_change = y['y_diff'].to(device)

            for batch in dataloader:

                net.train()
                optimizer.zero_grad()

                x_t1 = batch['x_t1'].to(device)
                x_t2 = batch['x_t2'].to(device)
                pred_change, pred_pop_t1, pred_pop_t2 = net(x_t1, x_t2)
                # need to be detached for backprop to work
                pred_pop_t1.detach(), pred_pop_t2.detach()

                pred_change = torch.sum(pred_change, dim=0)

                unit_str = f'{i_unit + 1:03d}/{len(training_units)}: Unit {training_unit} ({len(dataset)})'
                results_str = f'Pred: {pred_change.cpu().item():.0f}; GT: {y_change.cpu().item():.0f}'
                sys.stdout.write("\r%s" % 'Train' + ' ' + unit_str + ' ' + results_str)
                sys.stdout.flush()

                loss = criterion(pred_change, y_change.float())
                loss.backward()
                optimizer.step()

                # with torch.no_grad():
                #     a, b, c = net(dummy_tensor, dummy_tensor)
                #     print(
                #         f'{torch.sum(a).cpu().item():.5f} - {torch.sum(b).cpu().item():.5f} - {torch.sum(c).cpu().item():.5f}')

                loss_set.append(loss.item())

                global_step += 1
                epoch_float = global_step / steps_per_epoch
            # if i_unit == 5:
            #     break
            # end of unit (step)

        # assert (epoch == epoch_float)

        epoch_str = f'Train Loss - {np.mean(loss_set):.0f}'
        sys.stdout.write("\r%s" % epoch_str + '\n')
        sys.stdout.flush()

        # logging at the end of each epoch
        # evaluation.model_evaluation_units(net, cfg, 'training', epoch_float, global_step)
        evaluation.model_evaluation_units(net, cfg, 'test', epoch_float, global_step)

        # logging
        time = timeit.default_timer() - start
        wandb.log({
            'loss': np.mean(loss_set),
            'time': time,
            'step': global_step,
            'epoch': epoch_float,
        })

        if epoch in save_checkpoints and not cfg.DEBUG:
            print(f'saving network', flush=True)
            networks.save_checkpoint(net, optimizer, epoch, global_step, cfg)

        loss_set = []


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
