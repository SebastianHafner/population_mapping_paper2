import torch
import matplotlib.pyplot as plt
from matplotlib import lines
from scipy import stats
import numpy as np
from pathlib import Path
from utils import experiment_manager, networks, datasets, parsers, geofiles


def total_population(cfg: experiment_manager.CfgNode, run_type: str = 'all'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()
    ds = datasets.PopDataset(cfg, run_type, no_augmentations=True)
    y_gt, y_pred = 0, 0

    with torch.no_grad():
        for item in ds:
            x = item['x'].to(device)
            y = item['y']
            pred = net(x.unsqueeze(0))
            y_gt += y.item()
            y_pred += pred.cpu().item()

    print(y_gt, y_pred)


def unit_stats_bitemporal(cfg):
    file = Path(cfg.PATHS.OUTPUT) / 'inference' / 'unit_stats' / f'pop_kigali_{cfg.NAME}.json'
    assert(file.exists())
    pred_data = geofiles.load_json(file)

    metadata_file = Path(cfg.PATHS.DATASET) / f'metadata.json'
    metadata = geofiles.load_json(metadata_file)

    census = metadata['census']
    units = list(census.keys())
    fig_data = {
        'training': {
            'pred': {
                '2016': [],
                '2020': [],
            },
            'gt': {
                '2016': [],
                '2020': [],
            },
        },
        'test': {
            'pred': {
                '2016': [],
                '2020': [],
            },
            'gt': {
                '2016': [],
                '2020': [],
            },
        },
    }
    for unit in units:
        split = census[str(unit)]['split']
        for year in [2016, 2020]:
            # ground truth
            fig_data[split]['gt'][str(year)].append(census[str(unit)][f'pop{year}'])
            # prediction
            fig_data[split]['pred'][str(year)].append(pred_data[str(unit)][f'pred_pop{year}'])

    print(fig_data)
    fig, axs = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    plt.tight_layout()
    for i, split in enumerate(['training', 'test']):
        for j, year in enumerate([2016, 2020]):
            gt = fig_data[split]['gt'][str(year)]
            pred = fig_data[split]['pred'][str(year)]
            axs[i, j].scatter(gt, pred)
            _, _, r_value, *_ = stats.linregress(gt, pred)
            textstr = r'$R^2 = {r_value:.2f}$'.format(r_value=r_value)
            axs[i, j].text(0.05, 0.95, textstr, transform=axs[i, j].transAxes,verticalalignment='top')


        # difference
        gt_diff = np.array(fig_data[split]['gt']['2020']) - np.array(fig_data[split]['gt']['2016'])
        pred_diff = np.array(fig_data[split]['pred']['2020']) - np.array(fig_data[split]['pred']['2016'])
        print(split)
        print('Ground Truth')
        print(gt_diff)
        print(np.sum(gt_diff))
        print('Predicted')
        print(pred_diff)
        print(np.sum(pred_diff))
        axs[i, 2].scatter(gt_diff, pred_diff)
        _, _, r_value, *_ = stats.linregress(gt_diff, pred_diff)
        textstr = r'$R^2 = {r_value:.2f}$'.format(r_value=r_value)
        axs[i, 2].text(0.05, 0.95, textstr, transform=axs[i, 2].transAxes, verticalalignment='top')

    for index, ax in np.ndenumerate(axs):
        i, j = index
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Predicted')
        if i < 2 and j < 2:
            axs[i, j].set_xlim(0, 50_000)
            axs[i, j].set_ylim(0, 50_000)
            line = ax.plot([0, 50_000], [0, 50_000], c='r', zorder=-1, label='1:1 line')
        else:
            line = ax.plot([-1_000, 10_000], [-1_000, 10_000], c='r', zorder=-1, label='1:1 line')
            axs[i, j].set_xlim(-1_000, 10_000)
            axs[i, j].set_ylim(-1_000, 10_000)

    legend_elements = [
        lines.Line2D([0], [0], color='r', lw=1, label='1:1 Line'),
        lines.Line2D([0], [0], marker='.', color='w', markerfacecolor='#1f77b4', label='Census Unit', markersize=15),
    ]
    axs[0, 0].legend(handles=legend_elements, frameon=False, loc='upper center')


    cols = ['Population 2016', 'Population 2020', 'Difference']
    rows = ['Training', 'Test']


    pad = 5  # in points

    for ax, col in zip(axs[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    for ax, row in zip(axs[:, 0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')

    fig.tight_layout()
    # tight_layout doesn't take these labels into account. We'll need
    # to make some room. These numbers are are manually tweaked.
    # You could automatically calculate them, but it's a pain.
    fig.subplots_adjust(left=0.15, top=0.95)

    file = Path(cfg.PATHS.OUTPUT) / 'plots' / 'test.png'
    plt.savefig(file, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    args = parsers.deployment_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    unit_stats_bitemporal(cfg)
    # total_population(cfg, run_type=args.run_type)
