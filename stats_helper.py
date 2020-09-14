import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler
import scipy.stats
import matplotlib.pyplot as plt

def scale_me(y):
    sc = StandardScaler()
    yy = y.reshape(-1, 1)
    sc.fit(yy)
    y_std = sc.transform(yy).flatten()
    del yy
    return y_std

def test_distribution(distribution, y_std, cum_observed_frequency, percentile_bins, percentils_cutoffs, size):
    # setup distribution and get fitted distribution parameters
    dist = getattr(scipy.stats, distribution)
    param = dist.fit(y_std)
    # Obtain the KS test stats, round to 5 decimal places
    p = np.around(scipy.stats.kstest(y_std, distribution, args=param)[1], 5)

    # Get expected counts in percentile bins
    # this is based on a CDF
    cdf_fitted = dist.cdf(percentils_cutoffs, *param[:-2], loc=param[-2], scale=param[-1])
    expected_frequency = []
    for bin in range(len(percentile_bins)-1):
        expected_cdf_area = cdf_fitted[bin+1] - cdf_fitted[bin]
        expected_frequency.append(expected_cdf_area)

    # calculate chi-squared
    expected_frequency = np.array(expected_frequency) * size
    cum_expected_frequency = np.cumsum(expected_frequency)
    ss = sum(((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency)
    return p, ss

def test_distributions(y_std, cum_observed_frequency, percentile_bins, percentils_cutoffs, size):
    import warnings
    warnings.filterwarnings('ignore')

    dist_names = ['beta', 'expon', 'gamma', 'lognorm', 'norm', 'triang', 'uniform']

    chi_square = []
    p_values = []
    for distribution in dist_names:
        p, ss = test_distribution(distribution, y_std, cum_observed_frequency, percentile_bins, percentils_cutoffs, size)
        p_values.append(p)
        chi_square.append(ss)

    results = pd.DataFrame({'Distribution': dist_names,
                        'chi_square': chi_square,
                        'p_value': p_values}).sort_values(['chi_square'])

    print('\nDistrubtions sorted by goodness of fit')
    print('---------------------------------------')
    print(results)
    return results

def get_top_n_distributions(y, number_to_plot, results):
    number_of_bins = 50
    x = np.arange(len(y))
    bin_cutoffs = np.linspace(np.percentile(y, 0), np.percentile(y, 99), number_of_bins)

    # create plot
    h = plt.hist(y, bins=bin_cutoffs, color='0.75')

    dist_names = results['Distribution'].iloc[0:number_to_plot]

    parameters = []

    for dist_name in dist_names:
        dist = getattr(scipy.stats, dist_name)
        param = dist.fit(y)
        parameters.append(param)

        # Get line for each distribution and scale to match observed data
        pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1])
        scale_pdf = np.trapz(h[0], h[1][:-1]) / np.trapz(pdf_fitted, x)
        pdf_fitted *= scale_pdf

        # add line to plot
        plt.plot(pdf_fitted, label=dist_name)

        plt.xlim(0, np.percentile(y, 99))

    plt.legend()
    plt.show()

    for dist_name, params in zip(dist_names, parameters):
        print(f'Distribution {dist_name}')
        print(f'Parameters {params}')

    return dist_names

def get_quantiles(y):
    return np.quantile(y, [0.01, 0.5, 0.95, 0.99])

def fit_distributions(y):
    y_std = scale_me(y)
    size = len(y)
    percentile_bins = np.linspace(0, 100, 51)
    percentile_cutoffs = np.percentile(y_std, percentile_bins)
    observed_frequency, _ = (np.histogram(y_std, bins=percentile_cutoffs))
    cum_observed_frequency = np.cumsum(observed_frequency)
    results = test_distributions(y_std, cum_observed_frequency, percentile_bins, percentile_cutoffs, size)
    top_n_distributions = get_top_n_distributions(y, 3, results)
    plot_qq_plots(y_std, top_n_distributions, size)


def plot_qq_plots(y_std, top_n_distributions, size):
    data = y_std.copy()
    data.sort()

    for distribution in top_n_distributions:
        dist = getattr(scipy.stats, distribution)
        param = dist.fit(y_std)

        norm = dist.rvs(*param[0:-2], loc=param[-2], scale=param[-1], size=size)
        norm.sort()

        fig = plt.figure(figsize=(8,5))

        # qq plot
        ax1 = fig.add_subplot(121)
        ax1.plot(norm, data, 'o')
        min_value = np.floor(min(min(norm), min(data)))
        max_value = np.ceil(max(max(norm), max(data)))
        ax1.plot([min_value, max_value], [min_value, max_value], 'r--')
        ax1.set_xlim(min_value, max_value)
        ax1.set_xlabel('Theoretical quantiles')
        ax1.set_ylabel('Observed quantiles')
        ax1.set_title(f'qq plot for {distribution} distribution')

        # pp plot
        ax2 = fig.add_subplot(122)

        bins = np.percentile(norm, range(0, 101))
        data_counts, _ = np.histogram(data, bins)
        norm_counts, _ = np.histogram(norm, bins)
        cum_data = np.cumsum(data_counts)
        cum_norm = np.cumsum(norm_counts)
        cum_data = cum_data / max(cum_data)
        cum_norm = cum_norm / max(cum_norm)

        ax2.plot(cum_norm, cum_data, 'o')
        min_value = np.floor(min(min(cum_norm), min(cum_data)))
        max_value = np.ceil(max(max(cum_norm), max(cum_data)))
        ax2.plot([min_value, max_value], [min_value, max_value], 'r--')
        ax2.set_xlim(min_value, max_value)
        ax2.set_xlabel('Theoretical cumulative distribution')
        ax2.set_ylabel('Observed cumulative distribution')
        ax2.set_title(f'pp plot for {distribution} distribution')

        plt.tight_layout(pad=4)
        plt.show()
