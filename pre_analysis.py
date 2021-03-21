import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display, Markdown
import ipywidgets as widgets

def printmd(string):
    display(Markdown(string))

def get_iv_text(iv):
    if iv < 0.02:
        return 'useless'
    elif iv < 0.1:
        return 'weak'
    elif iv < 0.3:
        return 'medium'
    elif iv < 0.5:
        return 'strong'
    else:
        return 'suspicious'

def get_woe_and_iv(feature_values, target_values):
    df_woe_iv = np.round((pd.crosstab(feature_values, target_values, normalize='columns')
             .assign(woe=lambda dfx: np.log(dfx[1] / dfx[0]))
             .replace({'woe': {np.inf: 0, -np.inf: 0}})
             .assign(iv=lambda dfx: np.sum(dfx['woe']*
                                           (dfx[1]-dfx[0])))).reset_index(), 3)
    iv = df_woe_iv.iloc[0, 4]
    iv_text = get_iv_text(iv)
    print(f'IV: {iv} ({iv_text} predictor)')
    print(' ')
    for _, row in df_woe_iv.iterrows():
        print('{:22} {:>10}'.format(row.iloc[0], row.iloc[3]))

def calculate_bins_for_numeric(feature_values, target_values):
    bins = np.histogram(feature_values)[1]
    df = pd.DataFrame({'feature_values': feature_values,
                        'target_values': target_values,
                        'bucket': pd.cut(feature_values, bins=bins, include_lowest=True),
                        'bucket_number': pd.cut(feature_values, bins=bins, include_lowest=True, labels=False)}).sort_values('bucket_number')
    # make sure bucket is a string
    df.loc[:,'bucket'] = df.loc[:,'bucket'].astype('str')
    # clean bucket name
    df.loc[:,'bucket'] = df.loc[:,'bucket'].str.replace(r'[()\[\]]', '').str.replace(',', ' to')
    return df

def summary_stats_numeric(feature_values):
    n_values = len(feature_values)
    n_distinct = len(feature_values.unique())
    mean = np.round(np.mean(feature_values), 2)
    median = np.median(feature_values)
    std_dev = np.round(np.std(feature_values), 2)
    min_ = np.min(feature_values)
    max_ = np.max(feature_values)

    plt.figure(figsize=(15,5))
    sns.distplot(feature_values)
    plt.show()

    df = pd.DataFrame({'stat': ['N values','N distinct','Mean','Median', 'Std Dev', 'Min', 'Max'], 'value': [n_values, n_distinct, mean, median, std_dev, min_, max_]})
    for _, row in df.iterrows():
        print('{:22} {:>10}'.format(row['stat'], row['value']))


def summary_stats_categorical(feature_values):
    n_values = len(feature_values)
    distincts_with_counts = feature_values.value_counts().reset_index()
    distincts_with_counts.columns = ['value', 'count']
    distincts_with_counts.loc[:, 'count_as_perc'] = np.round(distincts_with_counts['count'] / np.sum(distincts_with_counts['count']) * 100, 0)
    n_distinct = distincts_with_counts.shape[0]
    distincts_with_counts.sort_values('count', ascending=False, inplace=True)
    mode = distincts_with_counts['value'][0]
    n_empty = 0 # TBD

    # show historgram
    plt.figure(figsize=(15,5))
    sns.barplot(x='value', y='count', data=distincts_with_counts, order=distincts_with_counts['value'])
    plt.show()
    printmd('**Summary Stats**')
    df = pd.DataFrame({'stat': ['N values','N distinct','Mode','N empty'], 'value': [n_values, n_distinct, mode, n_empty]})
    for _, row in df.iterrows():
        print('{:32} {:>10}'.format(row['stat'], row['value']))
    #print(distincts_with_counts[['value','count_as_perc','count']].head().to_string(index=False, header=False))
    printmd('**Frequency Stats**')
    for _, row in distincts_with_counts.iterrows():
        print('{:20} {:>10}% {:>10}'.format(row['value'], row['count_as_perc'], row['count']))
        

def get_all_summary_stats(X, y):
    numerical_columns = list(X.select_dtypes(include=[np.number]).columns)
    categorical_columns = list(X.select_dtypes(include=['object']).columns)

    # setup widgets
    # fix layout
    item_layout = widgets.Layout(margin='0 0 50px 0')

    # numeric feature widget
    num_outputs = [widgets.Output() for numerical_column in numerical_columns]
    # create tab widget
    num_tab = widgets.Tab(num_outputs, layout=item_layout)
    index = 0
    for numerical_column in numerical_columns:
        with num_outputs[index]:
            summary_stats_numeric(X[numerical_column])
            printmd('**Weight of Evidence**')
            binned_df = calculate_bins_for_numeric(X[numerical_column], y)
            get_woe_and_iv(binned_df['bucket'], binned_df['target_values'])

        num_tab.set_title(index, f'{numerical_column}')
        index = index + 1
    
    # categorical feature widget
    cat_outputs = [widgets.Output() for categorical_column in categorical_columns]
    # create tab widget
    cat_tab = widgets.Tab(cat_outputs, layout=item_layout)
    index = 0
    for categorical_column in categorical_columns:
        with cat_outputs[index]:
            summary_stats_categorical(X[categorical_column])
            printmd('**Weight of Evidence**')
            get_woe_and_iv(X[categorical_column], y)
        
        cat_tab.set_title(index, f'{categorical_column}')
        index = index + 1
    
    accordion = widgets.Accordion(children=[num_tab, cat_tab])
    accordion.set_title(0, 'Numeric Features')
    accordion.set_title(1, 'Categorical Features')

    display(accordion)