import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import scipy.stats

import matplotlib.pyplot as plt
import seaborn as sns

import ipywidgets as widgets
from IPython.display import display

def show_drift_in_widget(original_df, current_df, numerical_columns):
    # create outputs for each numerical_columns
    outputs = [widgets.Output() for numerical_column in numerical_columns]
    # fix layout
    item_layout = widgets.Layout(margin='0 0 50px 0')
    # create tab widget
    tab = widgets.Tab(outputs, layout=item_layout)
    index = 0
    for numerical_column in numerical_columns:
        p = get_drift_stats(original_df[numerical_column], current_df[numerical_column])
        with outputs[index]:
            print(f'P-value for Kolmogorov-Smirnov Test: {p}')
            plot_density(original_df[numerical_column], current_df[numerical_column], numerical_column)
        
        tab.set_title(index, f'Drift for {numerical_column}')
        index = index + 1
    
    display(tab)


def plot_density(original, current, label):
    x0 = original
    x1 = current
    plt.figure(figsize=[15,10])
    sns.distplot(x0, hist = False, kde = True,
                     kde_kws = {'linewidth': 3, 'shade': True},
                     label = 'Original')
    sns.distplot(x1, hist = False, kde = True,
                     kde_kws = {'linewidth': 3, 'shade': True},
                     label = 'Current')
    plt.title(f'Density Chart for {label}')
    plt.xlabel(label)
    plt.ylabel('Density')
    plt.show()

def plot_cdfs(original, current, label):

    plt.figure(figsize=[15,10])
    sns.distplot(original, hist=False, 
        hist_kws={'cumulative': True, 'density': True}, 
        kde_kws={'cumulative': True, 'shade': True})
    sns.distplot(current, hist=False, 
        hist_kws={'cumulative': True, 'density': True}, 
        kde_kws={'cumulative': True, 'shade': True})
    plt.title(f'CDF Chart for {label}')
    plt.xlabel(label)
    plt.ylabel('Cumulative Density')
    plt.show()
    #orig_sorted = np.sort(original)
    #orig_p = 1. * np.arange(len(orig_sorted)) / (len(data) - 1)

def get_drift_stats(original, current):

    # Obtain the KS test P statistic, round it to 5 decimal places
    p = scipy.stats.kstest(original, current)[1]
    p = np.around(p, 5)
    return p

def domain_classifier_covariate_shift_test(original, current):

    # create dataframe union of both
    original.loc[:, 'set'] = 0
    current.loc[:, 'set'] = 1
    model_data = pd.concat([original, current], ignore_index=True)

    print(model_data.shape)

    X = model_data.drop(columns=['set'])
    y = model_data['set']

    a = []

    linear_model = LogisticRegression(random_state=42)

    for i in range(1, 2):
        # create split
        X_train, X_test, y_train, y_test = train_test_split(
                                                            X, 
                                                            y, 
                                                            test_size=0.3)

        # fit model
        linear_model.fit(X_train, y_train)

        # determine accuracy
        y_lr_test = linear_model.predict(X_test)
        lr_accuracy = accuracy_score(y_test, y_lr_test)
        print(pd.DataFrame(confusion_matrix(y_test, y_lr_test), 
                columns=['Predicted Not Churn', 'Predicted to Churn'],
                index=['Actual Not Churn', 'Actual Churn']))
        bt = scipy.stats.binom_test(lr_accuracy, n=1, alternative='greater')
        #print('Accuracy: %.3f, Binomial Test: %.3f' % (lr_accuracy, bt))
        a.append({'accuracy': lr_accuracy, 'bt': bt})

    
    m_df = pd.DataFrame(a)
    m_df.plot()


def test_for_drift(original_df, current_df):

    # just do numerical columns for now
    numerical_columns = list(original_df.select_dtypes(include=[np.number]).columns)
    show_drift_in_widget(original_df, current_df, numerical_columns)
    '''for column in numerical_columns:
        p = get_drift_stats(original_df[column], current_df[column])
        print(f'P-value for Kolmogorov-Smirnov Test: {p}')
        plot_density(original_df[column], current_df[column], column)
        plot_cdfs(original_df[column], current_df[column], column)'''
