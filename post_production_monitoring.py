import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display, Markdown
import ipywidgets as widgets
import scipy
import scipy.stats

def get_drift_stats(original, current):
    p = scipy.stats.kstest(original, current)[1]
    p = np.around(p, 5)
    return p

def plot_density(original, current, label):
    plt.figure(figsize=(15, 10))
    sns.distplot(original, hist=False, kde=True,
        kde_kws={'linewidth': 3, 'shade': True},
        label='Original')
    sns.distplot(current, hist=False, kde=True,
        kde_kws={'linewidth': 3, 'shade': True},
        label='Current')

    plt.title(f'Density chart for {label}')
    plt.xlabel(label)
    plt.ylabel('Density')
    plt.show()

def plot_cdfs(original, current, label):
    plt.figure(figsize=(15, 10))
    sns.distplot(original, hist=False,
        hist_kws={'cumulative': True, 'density': True},
        kde_kws={'cumulative': True, 'shade': True},
        label='Original')
    sns.distplot(current, hist=False,
        hist_kws={'cumulative': True, 'density': True},
        kde_kws={'cumulative': True, 'shade': True},
        label='Current')

    plt.title(f'CDF chart for {label}')
    plt.xlabel(label)
    plt.ylabel('Cumulative Density')
    plt.show()

def show_drift(original_df, current_df):
    numerical_columns = list(original_df.select_dtypes(include=[np.number]).columns)

    outputs = [widgets.Output() for col in numerical_columns]
    item_layout = widgets.Layout(margin='0 0 50px 0')
    tab = widgets.Tab(outputs, layout=item_layout)
    index = 0
    for col in numerical_columns:
        p = get_drift_stats(original_df[col], current_df[col])
        with outputs[index]:
            print(f'P-value for Kolmogorov-Smirnov Test: {p}')
            plot_density(original_df[col], current_df[col], col)
            #plot_cdfs(original_df[col], current_df[col], col)
        
        tab.set_title(index, f'Drift for {col}')
        index = index + 1
    
    display(tab)
