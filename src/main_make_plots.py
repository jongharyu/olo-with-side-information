import argparse
import shelve
from itertools import cycle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = (10, 6)
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markeredgecolor'] = mpl.colors.colorConverter.to_rgba('k', alpha=.45)

# markers = [(i, j, 0) for i in range(2, 10) for j in range(1, 3)]
markers = cycle([',', 'o', 'v', 's', 'p', '*', 'P', 'X', 'D'])


def main(config):
    dataset_name = config.dataset
    is_data_metro = dataset_name.lower() == 'MetroInterStateTrafficVolume'.lower()

    # load results
    filename = "results/{}-cum-losses.out".format(dataset_name)
    my_shelf = shelve.open(filename)
    cum_losses = my_shelf['cum_losses']
    dim = my_shelf['dim']
    my_shelf.close()

    # Figure 1. OGDs
    fig, ax = plt.subplots(nrows=1,
                           ncols=dim,
                           figsize=(14, 2),
                           sharey='all')

    lines = []
    for i in range(dim):
        lines = []
        ax[i].set_prop_cycle('color', [plt.cm.Set1(k) for k in np.linspace(0, 1, 9)])
        depths = cum_losses['ogd'][i].keys()
        for (j, depth) in enumerate(depths):
            lr_scales = cum_losses['ogd'][i][depth].keys()
            lines.append(
                ax[i].plot(lr_scales,
                           [cum_losses['ogd'][i][depth][lr_scale] for lr_scale in lr_scales],
                           marker=None, label='D={}'.format(depth))[0]
            )

        # ax[i].legend(ncol=1)
        ax[i].ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        ax[i].set_title('quantizer {}'.format(i + 1))
        ax[i].set_xscale('log')
        ax[i].set_yscale('log')
        if is_data_metro:
            ax[i].set_ylim([3.5 * 10 ** 7, 1.5 * 10 ** 8])
        else:
            ax[i].set_ylim([10 ** 6, 3 * 10 ** 6])
    labels = ['D={}'.format(i) for i in range(dim)]
    fig.legend(
        handles=lines,  # The line objects
        labels=labels,  # The labels for each line
        loc='center right',  # Position of legend
        ncol=1,
        borderaxespad=0.1,  # Small spacing around legend box
    )
    plt.subplots_adjust(right=0.925)

    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.title('OGD w/ Markov'.format(dataset_name), pad=30)
    plt.xlabel("learning rate scale")
    plt.ylabel("cumulative loss")

    if config.save:
        plt.savefig('figs/{}-ogd.pdf'.format(dataset_name),
                    bbox_inches='tight',
                    transparent=False)
    plt.show()

    # Figure 2. Coordinatewise
    fig, ax = plt.subplots(nrows=1,
                           ncols=dim,
                           figsize=(14, 2),
                           sharey='all')
    lines = []
    for i in range(dim):
        ax[i].set_prop_cycle('color', [plt.cm.Set1(k) for k in np.linspace(0, 1, 9)])

        depths_markov = cum_losses['ogd'][i].keys()
        l1 = ax[i].plot(depths_markov,
                        [np.min([cum_losses['ogd'][i][depth][lr_scale] for lr_scale in lr_scales]) for depth in depths_markov],
                        marker=None,
                        linestyle='--')[0]
        l2 = ax[i].plot(depths_markov,
                        [cum_losses['dfeg_markov'][i][depth] for depth in depths_markov],
                        marker='v')[0]
        l3 = ax[i].plot(depths_markov,
                        [cum_losses['adanorm_markov'][i][depth] for depth in depths_markov],
                        marker='P')[0]
        l4 = ax[i].plot(depths_markov,
                        [cum_losses['kt_markov'][i][depth] for depth in depths_markov],
                        marker='o')[0]

        max_depths_ctw = cum_losses['ctw'][i].keys()
        l5 = ax[i].plot(max_depths_ctw,
                        [cum_losses['ctw'][i][max_depth] for max_depth in max_depths_ctw],
                        marker='X')[0]

        max_depths_cta = cum_losses['cta'][i].keys()
        l6 = ax[i].plot(max_depths_cta,
                        [cum_losses['cta'][i][max_depth] for max_depth in max_depths_ctw],
                        marker='X')[0]

        lines = [l1, l2, l3, l4, l5, l6]
        ax[i].set_title('quantizer {}'.format(i + 1))
        ax[i].set_yscale('log')
        if is_data_metro:
            ax[i].set_ylim([3.5 * 10 ** 7, 1.5 * 10 ** 8])
        else:
            ax[i].set_ylim([10 ** 6, 3 * 10 ** 6])

    labels = ['OGD w/ Markov (best)',
              'DFEG w/ Markov', 'AdaNorm w/ Markov',
              'KT w/ Markov',
              # 'KT w/ hint',
              'CTW', 'CTA']
    fig.legend(
        handles=lines,  # The line objects
        labels=labels,  # The labels for each line
        loc="center right",  # Position of legend
        borderaxespad=0.1,  # Small spacing around legend box
    )
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.subplots_adjust(right=0.85)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

    plt.xlabel("suffix length")
    if config.print_legend:
        plt.ylabel("cumulative loss", labelpad=20)
    if config.save:
        plt.savefig('figs/{}-statewise.pdf'.format(dataset_name),
                    pad_inches=0.2,
                    bbox_inches='tight',
                    transparent=False)
    plt.show()

    # Figure 3. Combined algorithms
    max_depths_ctw = range(1, 13, 2)
    max_depths_cta = range(1, 13, 2)

    figwidth = 3.75 if is_data_metro else 6
    figwidth = 6
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(figwidth, 2.5))
    ax.set_prop_cycle('color', [plt.cm.Set1(k) for k in np.linspace(0, 1, 9)])
    l0 = ax.plot(depths_markov, [np.min([np.min([cum_losses['ogd'][i][depth][lr_scale]
                                                 for lr_scale in lr_scales])
                                         for i in range(dim)])
                                 for depth in depths_markov],
                 linestyle='--',
                 marker=next(markers))[0]
    l1 = ax.plot(depths_markov, [np.min([cum_losses['dfeg_markov'][i][depth]
                                         for i in range(dim)])
                                 for depth in depths_markov],
                 linestyle='--',
                 marker=next(markers))[0]
    l2 = ax.plot(depths_markov, [np.min([cum_losses['adanorm_markov'][i][depth]
                                         for i in range(dim)])
                                 for depth in depths_markov],
                 linestyle='--',
                 marker=next(markers))[0]
    l3 = ax.plot(depths_markov, [np.min([cum_losses['kt_markov'][i][depth]
                                         for i in range(dim)])
                                 for depth in depths_markov],
                 linestyle='--',
                 marker=next(markers))[0]
    l4 = ax.plot(max_depths_ctw, [np.min([cum_losses['ctw'][i][max_depth]
                                          for i in range(dim)])
                                  for max_depth in max_depths_ctw],
                 marker=next(markers))[0]
    l5 = ax.plot(max_depths_cta, [np.min([cum_losses['cta'][i][max_depth]
                                          for i in range(dim)])
                                  for max_depth in max_depths_ctw],
                 marker=next(markers))[0]

    depths_combine_markovs_dims = cum_losses['add_markovs_over_dims'].keys()
    l6 = ax.plot(depths_combine_markovs_dims, [cum_losses['add_markovs_over_dims'][depth]
                                               for depth in depths_combine_markovs_dims],
                 linestyle='-.',
                 marker=next(markers))[0]
    l7 = ax.plot(depths_combine_markovs_dims, [cum_losses['mix_markovs_over_dims'][depth]
                                               for depth in depths_combine_markovs_dims],
                 marker=next(markers),
                 linestyle='solid')[0]

    max_depths_combine_ctws_dims = cum_losses['add_ctws_over_dims'].keys()
    l8 = ax.plot(max_depths_combine_ctws_dims, [cum_losses['add_ctws_over_dims'][depth]
                                                for depth in depths_combine_markovs_dims],
                 marker=next(markers),
                 linestyle='-.')[0]
    l9 = ax.plot(max_depths_combine_ctws_dims, [cum_losses['mix_ctws_over_dims'][depth]
                                                for depth in depths_combine_markovs_dims],
                 marker=next(markers),
                 linestyle='solid')[0]
    l10 = ax.plot(max_depths_combine_ctws_dims, [cum_losses['add_ctas_over_dims'][depth]
                                                 for depth in depths_combine_markovs_dims],
                  marker=next(markers),
                  linestyle='-.')[0]

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    lines = [l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10]
    labels = ['OGD w/ Markov',
              'DFEG w/ Markov', 'AdaNorm w/ Markov',
              'KT w/ Markov',
              # 'KT w/ hint',
              'CTW', 'CTA',
              'Add. KTs w/ Markov (dim)', 'Mix. KTs w/ Markov (dim)',
              # 'Add. KTs w/ Markov (dim,depth)', 'Mix. KTs w/ Markov (dim,depth)',
              'Add. CTWs (dim)', 'Mix. CTWs (dim)', 'Add. CTAs (dim)']

    if config.print_legend:
        legend = fig.legend(
            handles=lines,  # The line objects
            labels=labels,  # The labels for each line
            loc='center right',  # Position of legend
            ncol=1,
            borderaxespad=0.1,  # Small spacing around legend box
        )
        plt.subplots_adjust(right=0.6)

        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_alpha(0.0)

    plt.title(dataset_name, pad=15)
    plt.xlabel("suffix length")
    if config.print_legend:
        plt.ylabel("cumulative loss", labelpad=0)
    if config.save:
        plt.savefig('figs/{}-summary.pdf'.format(dataset_name),
                    bbox_inches='tight',
                    transparent=False)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make plots',)
    parser.add_argument('--dataset', type=str, choices=['BeijingPM2pt5', 'MetroInterstateTrafficVolume'])
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--print-legend', type=bool, default=True)
    config = parser.parse_args()
    main(config)
