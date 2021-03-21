import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler
import scipy.stats
import matplotlib.pyplot as plt

def scale_me(y):
    sc=StandardScaler() 
    yy = y.reshape (-1,1)
    sc.fit(yy)
    y_std =sc.transform(yy)
    y_std = y_std.flatten()
    del yy
    return y_std

def test_distribution(distribution, y_std, cum_observed_frequency, percentile_bins, percentile_cutoffs, size):
     # Set up distribution and get fitted distribution parameters
    dist = getattr(scipy.stats, distribution)
    param = dist.fit(y_std)
    
    # Obtain the KS test P statistic, round it to 5 decimal places
    p = scipy.stats.kstest(y_std, distribution, args=param)[1]
    p = np.around(p, 5)  
    
    # Get expected counts in percentile bins
    # This is based on a 'cumulative distrubution function' (cdf)
    cdf_fitted = dist.cdf(percentile_cutoffs, *param[:-2], loc=param[-2], 
                          scale=param[-1])
    expected_frequency = []
    for bin in range(len(percentile_bins)-1):
        expected_cdf_area = cdf_fitted[bin+1] - cdf_fitted[bin]
        expected_frequency.append(expected_cdf_area)
    
    # calculate chi-squared
    expected_frequency = np.array(expected_frequency) * size
    cum_expected_frequency = np.cumsum(expected_frequency)
    ss = sum (((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency)
    return p, ss

def test_distributions(y_std, cum_observed_frequency, percentile_bins, percentile_cutoffs, size):
    import warnings
    warnings.filterwarnings("ignore")
    
    """ dist_names = ['beta',
              'expon',
              'gamma',
              'laplace',
              'lognorm',
              'norm',
              'pareto',
              'pearson3',
              'triang',
              'uniform',
              'weibull_min', 
              'weibull_max'] """

    dist_names = [
                'beta',
                'expon',
                'gamma',
                'lognorm',
                'norm',
                'triang',
                'uniform']

    # Set up empty lists to stroe results
    chi_square = []
    p_values = []
    for distribution in dist_names:
        p, ss = test_distribution(distribution, y_std, cum_observed_frequency, percentile_bins, percentile_cutoffs, size)
        p_values.append(p)
        chi_square.append(ss)
    
    results = pd.DataFrame()
    results['Distribution'] = dist_names
    results['chi_square'] = chi_square
    results['p_value'] = p_values
    results.sort_values(['chi_square'], inplace=True)
        
    # Report results

    print ('\nDistributions sorted by goodness of fit:')
    print ('----------------------------------------')
    print (results)
    return results

def get_top_n_distributions(y, number_distributions_to_plot, results):
    number_of_bins = 50
    x = np.arange(len(y))
    bin_cutoffs = np.linspace(np.percentile(y,0), np.percentile(y,99),number_of_bins)

    # Create the plot
    h = plt.hist(y, bins = bin_cutoffs, color='0.75')

    # Get the top three distributions from the previous phase
    dist_names = results['Distribution'].iloc[0:number_distributions_to_plot]

    # Create an empty list to stroe fitted distribution parameters
    parameters = []

    # Loop through the distributions ot get line fit and paraemters

    for dist_name in dist_names:
        # Set up distribution and store distribution paraemters
        dist = getattr(scipy.stats, dist_name)
        param = dist.fit(y)
        parameters.append(param)
        
        # Get line for each distribution (and scale to match observed data)
        pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1])
        scale_pdf = np.trapz (h[0], h[1][:-1]) / np.trapz (pdf_fitted, x)
        pdf_fitted *= scale_pdf
        
        # Add the line to the plot
        plt.plot(pdf_fitted, label=dist_name)
        
        # Set the plot x axis to contain 99% of the data
        # This can be removed, but sometimes outlier data makes the plot less clear
        plt.xlim(0,np.percentile(y,99))

    # Add legend and display plot

    plt.legend()
    plt.show()

    # Store distribution paraemters in a dataframe (this could also be saved)
    dist_parameters = pd.DataFrame()
    dist_parameters['Distribution'] = (
            results['Distribution'].iloc[0:number_distributions_to_plot])
    dist_parameters['Distribution parameters'] = parameters

    # Print parameter results
    print ('\nDistribution parameters:')
    print ('------------------------')

    for _, row in dist_parameters.iterrows():
        print ('\nDistribution:', row[0])
        print ('Parameters:', row[1] )
    return dist_names

def plot_qq_plots(y_std, top_n_distributions, size):
    data = y_std.copy()
    data.sort()

    # Loop through selected distributions (as previously selected)

    for distribution in top_n_distributions:
        # Set up distribution
        dist = getattr(scipy.stats, distribution)
        param = dist.fit(y_std)
        
        # Get random numbers from distribution
        norm = dist.rvs(*param[0:-2],loc=param[-2], scale=param[-1],size = size)
        norm.sort()
        
        # Create figure
        fig = plt.figure(figsize=(8,5)) 
        
        # qq plot
        ax1 = fig.add_subplot(121) # Grid of 2x2, this is suplot 1
        ax1.plot(norm,data,"o")
        min_value = np.floor(min(min(norm),min(data)))
        max_value = np.ceil(max(max(norm),max(data)))
        ax1.plot([min_value,max_value],[min_value,max_value],'r--')
        ax1.set_xlim(min_value,max_value)
        ax1.set_xlabel('Theoretical quantiles')
        ax1.set_ylabel('Observed quantiles')
        title = 'qq plot for ' + distribution +' distribution'
        ax1.set_title(title)
        
        # pp plot
        ax2 = fig.add_subplot(122)
        
        # Calculate cumulative distributions
        bins = np.percentile(norm,range(0,101))
        data_counts, bins = np.histogram(data,bins)
        norm_counts, bins = np.histogram(norm,bins)
        cum_data = np.cumsum(data_counts)
        cum_norm = np.cumsum(norm_counts)
        cum_data = cum_data / max(cum_data)
        cum_norm = cum_norm / max(cum_norm)
        
        # plot
        ax2.plot(cum_norm,cum_data,"o")
        min_value = np.floor(min(min(cum_norm),min(cum_data)))
        max_value = np.ceil(max(max(cum_norm),max(cum_data)))
        ax2.plot([min_value,max_value],[min_value,max_value],'r--')
        ax2.set_xlim(min_value,max_value)
        ax2.set_xlabel('Theoretical cumulative distribution')
        ax2.set_ylabel('Observed cumulative distribution')
        title = 'pp plot for ' + distribution +' distribution'
        ax2.set_title(title)
        
        # Display plot    
        plt.tight_layout(pad=4)
        plt.show()

def fit_distributions(y):
    y_std = scale_me(y)
    size = len(y)
    percentile_bins = np.linspace(0,100,51)
    percentile_cutoffs = np.percentile(y_std, percentile_bins)
    observed_frequency, _ = (np.histogram(y_std, bins=percentile_cutoffs))
    cum_observed_frequency = np.cumsum(observed_frequency)
    results = test_distributions(y_std, cum_observed_frequency, percentile_bins, percentile_cutoffs, size)
    top_n_distributions = get_top_n_distributions(y, 3, results)
    plot_qq_plots(y_std, top_n_distributions, size)

def get_quantiles(y):
    return np.quantile(y, [0.01, 0.5, 0.95, 0.99])

def bayesian_blocks(t):
    """Bayesian Blocks Implementation

    By Jake Vanderplas.  License: BSD
    Based on algorithm outlined in http://adsabs.harvard.edu/abs/2012arXiv1207.5578S

    Parameters
    ----------
    t : ndarray, length N
        data to be histogrammed

    Returns
    -------
    bins : ndarray
        array containing the (N+1) bin edges

    Notes
    -----
    This is an incomplete implementation: it may fail for some
    datasets.  Alternate fitness functions and prior forms can
    be found in the paper listed above.
    """
    # copy and sort the array
    t = np.sort(t)
    N = t.size

    # create length-(N + 1) array of cell edges
    edges = np.concatenate([t[:1],
                            0.5 * (t[1:] + t[:-1]),
                            t[-1:]])
    block_length = t[-1] - edges

    # arrays needed for the iteration
    nn_vec = np.ones(N)
    best = np.zeros(N, dtype=float)
    last = np.zeros(N, dtype=int)

    #-----------------------------------------------------------------
    # Start with first data cell; add one cell at each iteration
    #-----------------------------------------------------------------
    for K in range(N):
        # Compute the width and count of the final bin for all possible
        # locations of the K^th changepoint
        width = block_length[:K + 1] - block_length[K + 1]
        count_vec = np.cumsum(nn_vec[:K + 1][::-1])[::-1]

        # evaluate fitness function for these possibilities
        fit_vec = count_vec * (np.log(count_vec) - np.log(width))
        fit_vec -= 4  # 4 comes from the prior on the number of changepoints
        fit_vec[1:] += best[:K]

        # find the max of the fitness: this is the K^th changepoint
        i_max = np.argmax(fit_vec)
        last[K] = i_max
        best[K] = fit_vec[i_max]

    #-----------------------------------------------------------------
    # Recover changepoints by iteratively peeling off the last block
    #-----------------------------------------------------------------
    change_points =  np.zeros(N, dtype=int)
    i_cp = N
    ind = N
    while True:
        i_cp -= 1
        change_points[i_cp] = ind
        if ind == 0:
            break
        ind = last[ind - 1]
    change_points = change_points[i_cp:]

    return edges[change_points]

def _lnL(n_bins, x, y, a=10):
    """
    Log likelihood from Hogg 2008
    """
    N, e, _ = scipy.stats.binned_statistic(x, statistic="sum", values=y, bins=n_bins)
    d = abs(e[0] - e[1])

    if any((N + a) < 1):
        return np.nan

    s = np.sum(N + a)
    L = np.sum(N * np.log((N + a - 1) / (d * (s - 1))))

    return L

def _optimal_bin_no(x, y):
    """
    Step through the bins in 2's and evaluate the log-like
    """
    bins = np.arange(2, 100, 2).astype(int)

    ls = []
    for b in bins:
        l = _lnL(b, x, y)
        # if you get a nan you'll get nans for all
        # subsequent bins
        if np.isnan(l):
            break

        ls.append(l)
    return bins[np.argmax(ls)]

def get_optimal_bins(x, y):
    bin_no = _optimal_bin_no(x, y)
    bins = np.histogram(x, bins=bin_no)[1]

    pdf = lambda a: np.sum(a) / np.sum(y)
    mu = scipy.stats.binned_statistic(x, statistic=pdf, values=y, bins=bins)
    print(mu)
    return bins

if __name__ == "__main__":
    a = [1 ,2 ,3 ,4]
    fit_distributions(a)
