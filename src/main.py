import argparse
import shelve
from collections import defaultdict
from timeit import default_timer as timer

import numpy as np

import datasets
from learners import OnlineGradientDescent, DimensionFreeExponentiatedGradient, AdaptiveNormal, \
    KTOLO, Addition, Mixture, ContextTreeWeightingOLO, ContextTreeAdditionOLO
from problems import LinearRegressionWithAbsoluteLoss, Linear
from quantizer import Quantizer, get_standard_vector
from side_information import MarkovSideInformation


def main(config):
    problem = LinearRegressionWithAbsoluteLoss()
    dataset = getattr(datasets, config.dataset)(
        root='.',
        standardize=True,
        rescale=False,
        bias=True,
        normalize=True,
        batch_normalize=False)
    dim = dataset.X.shape[1]
    
    cum_losses = dict()
    
    # data
    X, y = dataset.X, dataset.y
    data = X, y

    init_wealth = 10 ** (-5.)

    print('(1) Markov side information with single coordinate-quantization')
    depths_markov = [0] + list(range(1, 13, 2))
    # print('\t(1-1) OGD with Markov side information')
    # ogds = defaultdict(lambda: defaultdict(dict))
    # cum_losses['ogd'] = defaultdict(lambda: defaultdict(dict))
    # lr_scales = np.array([1.5 ** n for n in np.arange(5, 25)])
    # for i in range(dim):
    #     for depth in depths_markov:
    #         get_side_information = lambda: MarkovSideInformation(depth=depth, quantizer=Quantizer(quantizer_vector=get_standard_vector(dim, i)))
    #         for lr_scale in lr_scales:
    #             ogds[i][depth][lr_scale] = OnlineGradientDescent(dim=dim,
    #                                                              lr_scale=lr_scale,
    #                                                              problem=problem,
    #                                                              side_information=get_side_information()).fit(data)
    #             cum_losses['ogd'][i][depth][lr_scale] = ogds[i][depth][lr_scale].cum_loss
    #         print(depth, end=' ')
    #     print()

    print('\t(1-2) KT, DFEG, AdaNormal with Markov side information')
    kt_markovs = defaultdict(dict)
    cum_losses['kt_markov'] = defaultdict(dict)

    dfeg_markovs = defaultdict(dict)
    cum_losses['dfeg_markov'] = defaultdict(dict)

    adanorm_markovs = defaultdict(dict)
    cum_losses['adanorm_markov'] = defaultdict(dict)

    for i in range(dim):
        for depth in depths_markov:
            get_side_information = lambda: MarkovSideInformation(depth=depth, quantizer=Quantizer(quantizer_vector=get_standard_vector(dim, i)))

            dfeg_markovs[i][depth] = DimensionFreeExponentiatedGradient(dim=dim,
                                                                        problem=problem,
                                                                        side_information=get_side_information()).fit(data)
            cum_losses['dfeg_markov'][i][depth] = dfeg_markovs[i][depth].cum_loss

            adanorm_markovs[i][depth] = AdaptiveNormal(dim=dim,
                                                       problem=problem,
                                                       side_information=get_side_information()).fit(data)
            cum_losses['adanorm_markov'][i][depth] = adanorm_markovs[i][depth].cum_loss

            kt_markovs[i][depth] = KTOLO(dim=dim,
                                         init_wealth=init_wealth,
                                         problem=problem,
                                         side_information=get_side_information()).fit(data)
            cum_losses['kt_markov'][i][depth] = kt_markovs[i][depth].cum_loss
            print(depth, end=' ')
        print()

    print('\t(1-3) Context Tree Weighting')
    times = []

    max_depths_ctw = range(1, 13, 2)

    ctws = defaultdict(dict)
    cum_losses['ctw'] = defaultdict(dict)

    for i in range(dim):
        print("quantizer {}".format(i))
        for max_depth in max_depths_ctw:
            start = timer()
            ctws[i][max_depth] = ContextTreeWeightingOLO(dim=dim,
                                                         init_wealth=init_wealth,
                                                         max_depth=max_depth,
                                                         problem=problem,
                                                         quantizer=Quantizer(quantizer_vector=get_standard_vector(dim, i))).fit(data)
            cum_losses['ctw'][i][max_depth] = ctws[i][max_depth].cum_loss
            times.append(timer() - start)
            print(max_depth, "{:.2f}s".format(times[-1]), "{:.3e}".format(ctws[i][max_depth].cum_loss))

    print('\t(1-4) Context Tree Addition')
    times = []

    max_depths_cta = range(1, 13, 2)

    ctas = defaultdict(dict)
    cum_losses['cta'] = defaultdict(dict)

    for i in range(dim):
        print("quantizer {}".format(i))
        for max_depth in max_depths_cta:
            start = timer()
            get_base_algorithm = lambda: KTOLO(dim, problem=Linear(), side_information=None, init_wealth=init_wealth)
            ctas[i][max_depth] = ContextTreeAdditionOLO(dim=dim,
                                                        max_depth=max_depth,
                                                        problem=problem,
                                                        quantizer=Quantizer(quantizer_vector=get_standard_vector(dim, i), ),
                                                        get_base_algorithm=get_base_algorithm).fit(data)
            cum_losses['cta'][i][max_depth] = ctas[i][max_depth].cum_loss
            times.append(timer() - start)
            print(max_depth, "{:.2f}s".format(times[-1]), "{:.3e}".format(ctas[i][max_depth].cum_loss))

    print('(2) Combine Markov side information with multiple coordinate-quantizations')

    print('\t(2-1) Combine Markov side information over coordinates')
    depths_combine_markovs_dims = range(1, 13, 2)

    add_markovs_over_dims = dict()
    cum_losses['add_markovs_over_dims'] = dict()

    mix_markovs_over_dims = dict()
    cum_losses['mix_markovs_over_dims'] = dict()

    for depth in depths_combine_markovs_dims:
        algorithms = [KTOLO(dim=dim,
                            init_wealth=init_wealth,
                            problem=problem,
                            side_information=MarkovSideInformation(depth, quantizer=Quantizer(quantizer_vector=get_standard_vector(dim, i))))
                      for i in range(dim)]

        add_markovs_over_dims[depth] = Addition(dim=dim, algorithms=algorithms).fit(data)
        cum_losses['add_markovs_over_dims'][depth] = add_markovs_over_dims[depth].cum_loss

        mix_markovs_over_dims[depth] = Mixture(dim, algorithms, init_wealth=init_wealth, weights=None).fit(data)
        cum_losses['mix_markovs_over_dims'][depth] = mix_markovs_over_dims[depth].cum_loss
        print(depth, end=' ')

    print('\t(2-2) Combine Markov side information over depths and coordinates')
    max_depths_combine_markovs_depths_dims = range(1, 13, 2)

    add_markovs_over_depths_dims = dict()
    cum_losses['add_markovs_over_depths_dims'] = dict()

    mix_markovs_over_depths_dims = dict()
    cum_losses['mix_markovs_over_depths_dims'] = dict()

    for depth in max_depths_combine_markovs_depths_dims:
        algorithms = [KTOLO(dim=dim,
                            init_wealth=init_wealth,
                            problem=problem,
                            side_information=MarkovSideInformation(depth, quantizer=Quantizer(quantizer_vector=get_standard_vector(dim, i))))
                      for _ in range(depth) for i in range(dim)]

        add_markovs_over_depths_dims[depth] = Addition(dim, algorithms).fit(data)
        cum_losses['add_markovs_over_depths_dims'][depth] = add_markovs_over_depths_dims[depth].cum_loss

        mix_markovs_over_depths_dims[depth] = Mixture(dim, algorithms, init_wealth=init_wealth, weights=None).fit(data)
        cum_losses['mix_markovs_over_depths_dims'][depth] = mix_markovs_over_depths_dims[depth].cum_loss

        print(depth, end=' ')

    print('\t(2-3) Combine CTWs over depths and coordinates')
    max_depths_combine_ctws_dims = range(1, 13, 2)

    add_ctws_over_dims = dict()
    cum_losses['add_ctws_over_dims'] = dict()

    mix_ctws_over_dims = dict()
    cum_losses['mix_ctws_over_dims'] = dict()

    for max_depth in max_depths_combine_ctws_dims:
        algorithms = [ContextTreeWeightingOLO(dim=dim,
                                              init_wealth=init_wealth,
                                              quantizer=Quantizer(get_standard_vector(dim, i)),
                                              max_depth=max_depth,
                                              problem=problem)
                      for i in range(dim)]

        add_ctws_over_dims[max_depth] = Addition(dim, algorithms).fit(data)
        cum_losses['add_ctws_over_dims'][max_depth] = add_ctws_over_dims[max_depth].cum_loss

        mix_ctws_over_dims[max_depth] = Mixture(dim, algorithms, init_wealth=init_wealth, weights=None).fit(data)
        cum_losses['mix_ctws_over_dims'][max_depth] = mix_ctws_over_dims[max_depth].cum_loss
        print(max_depth, end=' ')

    print('\t(2-4) Combine CTAs over depths and coordinates')
    max_depths_combine_ctas_dims = range(1, 13, 2)

    add_ctas_over_dims = dict()
    cum_losses['add_ctas_over_dims'] = dict()

    for max_depth in max_depths_combine_ctas_dims:
        get_side_information = lambda: KTOLO(dim, problem=Linear(), side_information=None, init_wealth=init_wealth)
        algorithms = [ContextTreeAdditionOLO(dim=dim,
                                             max_depth=max_depth,
                                             problem=problem,
                                             quantizer=Quantizer(quantizer_vector=get_standard_vector(dim, i), ),
                                             get_base_algorithm=get_side_information).fit(data)
                      for i in range(dim)]

        add_ctas_over_dims[max_depth] = Addition(dim, algorithms).fit(data)
        cum_losses['add_ctas_over_dims'][max_depth] = add_ctas_over_dims[max_depth].cum_loss

        print(max_depth, end=' ')

    # Convert defaultdict to normal dict for pickling
    for key in cum_losses:
        cum_losses[key] = dict(cum_losses[key])
    cum_losses_new = dict(cum_losses)

    # Pickle variables
    filename = "results/{}-cum-losses.out".format(dataset.name)
    my_shelf = shelve.open(filename, 'n')  # 'n' for new

    dataset_name = dataset.name
    var_names = ['dataset_name', 'dim', 'cum_losses']

    for key in var_names:
        try:
            my_shelf[key] = globals()[key]
        except TypeError:
            #
            # __builtins__, my_shelf, and imported modules can not be shelved.
            #
            print('ERROR shelving: {0}'.format(key))
    my_shelf.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Online linear regression with absolute loss and side information: '
                                                 'a script for reproducing experimental results of the paper '
                                                 '``Parameter-free Online Linear Optimization with Side Information '
                                                 'via Universal Coin Betting',)
    parser.add_argument('--dataset', type=str, choices=['BeijingPM2pt5', 'MetroInterstateTrafficVolume'])
    config = parser.parse_args()
    main(config)
