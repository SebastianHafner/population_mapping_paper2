import sys
import os
import timeit

import torch
from torch import optim
from torch.utils import data as torch_data

import wandb
import numpy as np

from utils import networks, datasets, loss_functions, evaluation, experiment_manager, parsers


def run_training(cfg):

    net = networks.PopulationNet(cfg.MODEL)
    net.to(device)

    optimizer = optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)

    criterion = loss_functions.get_criterion(cfg.MODEL.LOSS_TYPE)

    # reset the generators
    dataset = datasets.PopDataset(cfg=cfg, run_type='training')
    print(dataset)

    dataloader_kwargs = {
        'batch_size': cfg.TRAINER.BATCH_SIZE,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle': cfg.DATALOADER.SHUFFLE,
        'drop_last': True,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    # unpacking cfg
    epochs = cfg.TRAINER.EPOCHS
    save_checkpoints = cfg.SAVE_CHECKPOINTS
    steps_per_epoch = len(dataloader)

    # tracking variables
    global_step = epoch_float = 0

    evaluation.model_evaluation(net, cfg, 'training', epoch_float, global_step, cfg.LOGGING.MAX_SAMPLES)
    evaluation.model_evaluation(net, cfg, 'validation', epoch_float, global_step, cfg.LOGGING.MAX_SAMPLES)

    for epoch in range(1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')

        start = timeit.default_timer()
        loss_set, pop_set = [], []

        for i, batch in enumerate(dataloader):

            net.train()
            optimizer.zero_grad()

            x = batch['x'].to(device)
            y_gts = batch['y'].to(device)
            y_pred = net(x)

            loss = criterion(y_pred, y_gts.float())
            loss.backward()
            optimizer.step()

            loss_set.append(loss.item())
            pop_set.append(y_gts.flatten())

            global_step += 1
            epoch_float = global_step / steps_per_epoch

            if global_step % cfg.LOGGING.FREQUENCY == 0:
                print(f'Logging step {global_step} (epoch {epoch_float:.2f}).')
                # evaluation on sample of training and validation set
                evaluation.model_evaluation(net, cfg, 'training', epoch_float, global_step, cfg.LOGGING.MAX_SAMPLES)
                evaluation.model_evaluation(net, cfg, 'validation', epoch_float, global_step, cfg.LOGGING.MAX_SAMPLES)

                # logging
                time = timeit.default_timer() - start
                wandb.log({
                    'loss': np.mean(loss_set),
                    'labeled_percentage': 100,
                    'time': time,
                    'step': global_step,
                    'epoch': epoch_float,
                })
                start = timeit.default_timer()
                loss_set = []
            # end of batch

        assert (epoch == epoch_float)
        print(f'epoch float {epoch_float} (step {global_step}) - epoch {epoch}')
        # evaluation at the end of an epoch
        evaluation.model_evaluation(net, cfg, 'training', epoch_float, global_step)
        evaluation.model_evaluation(net, cfg, 'validation', epoch_float, global_step)

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
