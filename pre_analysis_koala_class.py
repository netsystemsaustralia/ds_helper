import warnings

import numpy as np
import pandas as pd
import databricks.koalas as ks


import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display, Markdown
import ipywidgets as widgets


class PreAnalysis:
    
    def __init__(self, features, target, use_optimal_binning=True):
        self.features = features
        self.target = target
        self.use_optimal_binning = use_optimal_binning
        self.target_type = 'categorical' if self.target.dtype == 'object' else 'numeric'
        self._infer_use_case()
        self.numerical_columns = list(self.features.select_dtypes(include=['float64', 'float32', 'int32']).columns)
        self.categorical_columns = list(self.features.select_dtypes(include=['object']).columns)
        
    def _printmd(self, string):
        display(Markdown(string))

    def _label_to_numeric(self, as_series=True):
        if self.target_type == 'numeric':
            return self.target
        else:
            if as_series:
                return pd.Series(np.where(self.target == 'Yes', 1, 0))
            else:
                return np.where(self.target == 'Yes', 1, 0)
    
    def _get_iv_text(self, iv):
        if iv < 0.02:
            return 'useless'
        elif iv < 0.1:
            return 'weak'
        elif iv < 0.3:
            return 'medium'
        elif iv < 0.5:
            return 'strong'
        else:
            return 'suspiciously good...'

    def _get_woe_and_iv(self, feature_values, target_values):
        df_woe_iv = np.round((pd.crosstab(feature_values, target_values, normalize='columns')
                .assign(woe=lambda dfx: np.log(dfx[1] / dfx[0]))
                .replace({'woe': {np.inf: 0, -np.inf: 0}})
                .assign(iv=lambda dfx: np.sum(dfx['woe']*
                                            (dfx[1]-dfx[0])))).reset_index(), 3)
        iv = df_woe_iv.iloc[0, 4]
        iv_text = self._get_iv_text(iv)
        print(f'IV: {iv} ({iv_text} predictor)')
        print(' ')
        for _, row in df_woe_iv.iterrows():
            print('{:22} {:>10}'.format(row.iloc[0], row.iloc[3]))

    def _calculate_bins_for_numeric(self, feature_values, target_values):
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

    def _summary_stats_numeric(self, feature_name):
        n_values = self.features.loc[:, feature_name].count()
        n_distinct = self.features.loc[:, feature_name].nunique()
        mean = np.round(self.features.loc[:, feature_name].mean(), 2)
        median = self.features.loc[:, feature_name].quantile()
        std_dev = np.round(self.features.loc[:, feature_name].std(), 2)
        min_ = self.features.loc[:, feature_name].min()
        max_ = self.features.loc[:, feature_name].max()
        n_empty = self.features.loc[:, feature_name].isna().sum()

        # get sample data set for graphs
        sample_np = self.features.loc[:, feature_name].sample(frac=0.2, random_state=42, replace=False).to_numpy()

        plt.figure(figsize=(15,5))
        sns.distplot(sample_np)
        
        plt.show()

        plt.figure(figsize=(15,5))
        sns.boxplot(x=sample_np, orient='v')
        plt.show()

        df = pd.DataFrame({'stat': ['N values','N distinct','Mean','Median', 'Std Dev', 'Min', 'Max', 'N empty'], 'value': [n_values, n_distinct, mean, median, std_dev, min_, max_, n_empty]})
        for _, row in df.iterrows():
            print('{:22} {:>10}'.format(row['stat'], row['value']))


    def _summary_stats_categorical(self, feature_name):
        n_values = self.features.loc[:, feature_name].count()
        distincts_with_counts = self.features.loc[:, feature_name].value_counts().reset_index().to_pandas()
        distincts_with_counts.columns = ['value', 'count']
        distincts_with_counts.loc[:, 'count_as_perc'] = np.round(distincts_with_counts['count'] / np.sum(distincts_with_counts['count']) * 100, 0)
        n_distinct = distincts_with_counts.shape[0]
        distincts_with_counts.sort_values('count', ascending=False, inplace=True)
        mode = distincts_with_counts['value'][0]
        n_empty = self.features.loc[:, feature_name].isna().sum()

        # show historgram
        plt.figure(figsize=(25,5))
        sns.barplot(x='value', y='count', data=distincts_with_counts, order=distincts_with_counts['value'])
        plt.show()
        self._printmd('**Summary Stats**')
        df = pd.DataFrame({'stat': ['N values','N distinct','Mode','N empty'], 'value': [n_values, n_distinct, mode, n_empty]})
        for _, row in df.iterrows():
            print('{:32} {:>10}'.format(row['stat'], row['value']))
        #print(distincts_with_counts[['value','count_as_perc','count']].head().to_string(index=False, header=False))
        self._printmd('**Frequency Stats**')
        for _, row in distincts_with_counts.iterrows():
            print('{:40} {:>10}% {:>10}'.format(row['value'], row['count_as_perc'], row['count']))
            
    def _infer_use_case(self):
        c1 = self.target.dtype == 'int64'
        c2 = len(self.target.unique()) <= 20
        c3 = self.target.dtype == 'object'
        
        if ( ( (c1) & (c2) ) | (c3)   ):
            if (len(self.target.unique()) > 2):
                self.ml_usecase = 'multi'
            else:
                self.ml_usecase = 'binary'
        else:
            self.ml_usecase = 'regression'

    def get_all_summary_stats(self):
        # setup widgets
        # fix layout
        item_layout = widgets.Layout(margin='0 0 50px 0')

        # target feature widget
        targetoutput = widgets.Output()
        # create tab widget
        with targetoutput:
            if self.target_type == 'categorical':
                self._summary_stats_categorical(self._label_to_numeric())
            else: # TBD 
                self._summary_stats_numeric(self.target)

        # numeric feature widget
        num_outputs = [widgets.Output() for numerical_column in self.numerical_columns]
        # create tab widget
        num_tab = widgets.Tab(num_outputs, layout=item_layout)
        index = 0
        for numerical_column in self.numerical_columns:
            with num_outputs[index]:
                self._summary_stats_numeric(self.features[numerical_column])
                self._printmd('**Weight of Evidence**')
                binned_df = self._calculate_bins_for_numeric(self.features[numerical_column], self._label_to_numeric())
                self._get_woe_and_iv(binned_df['bucket'], binned_df['target_values'])

            num_tab.set_title(index, f'{numerical_column}')
            index = index + 1
        
        # categorical feature widget
        cat_outputs = [widgets.Output() for categorical_column in self.categorical_columns]
        # create tab widget
        cat_tab = widgets.Tab(cat_outputs, layout=item_layout)
        index = 0
        for categorical_column in self.categorical_columns:
            with cat_outputs[index]:
                self._summary_stats_categorical(self.features[categorical_column])
                self._printmd('**Weight of Evidence**')
                self._get_woe_and_iv(self.features[categorical_column], self._label_to_numeric())
            
            cat_tab.set_title(index, f'{categorical_column}')
            index = index + 1
        
        accordion = widgets.Accordion(children=[targetoutput, num_tab, cat_tab])
        accordion.set_title(0, 'Target Label')
        accordion.set_title(1, 'Numeric Features')
        accordion.set_title(2, 'Categorical Features')

        the_box = widgets.HBox([accordion])

        display(the_box)

    def show_correlation_matrix(self, with_target=False, remove_duplicate_half=False):

        df = self.features[self.numerical_columns]
        if with_target:
            df.loc[:, 'target'] = self._label_to_numeric()
        
        my_corr = df.corr()

        params = {
            'annot': True,
            'vmin': -1,
            'vmax': 1,
            'center': 0,
            'cmap': 'coolwarm'
        }

        if remove_duplicate_half:
            mask = np.triu(np.ones_like(my_corr, dtype=bool))
            params['mask'] = mask

        plt.figure(figsize=(15,5))
        sns.heatmap(my_corr, **params)
        plt.yticks(rotation = 0)
        plt.show()