import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import phik

from IPython.display import display, Markdown
import ipywidgets as widgets

from stats_helper import fit_distributions

class PreAnalysis:

    def __init__(self, features, target):
        self.features = features
        self.target = target
        self.target_type = 'categorical' if self.target.dtype == 'object' else 'numerical'
        self._infer_use_case()
        self.numerical_columns = list(self.features.select_dtypes(include=[np.number]).columns)
        self.categorical_columns = list(self.features.select_dtypes(include=['object','category']).columns)

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

    def _infer_use_case(self):
        c1 = self.target.dtype == 'int64'
        c2 = len(self.target.unique()) <= 20
        c3 = self.target.dtype == 'object'

        if (((c1) & (c2)) | (c3)):
            if (len(self.target.unique()) > 2):
                self.ml_usecase = 'multi'
            else:
                self.ml_usecase = 'binary'
        else:
            self.ml_usecase = 'regression'

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
        df.loc[:, 'bucket'] = df.loc[:, 'bucket'].astype('str')
        # clean bucket name
        df.loc[:, 'bucket'] = df.loc[:, 'bucket'].str.replace(r'[()\[\]]', '').str.replace(',', 'to')
        return df

    def _summary_stats_numeric(self, feature_values):
        n_values = len(feature_values)
        n_distinct = len(feature_values.unique())
        mean = np.round(np.mean(feature_values), 2)
        median = np.round(np.median(feature_values), 2)
        std_dev = np.round(np.std(feature_values), 2)
        _min = np.round(np.min(feature_values), 2)
        _max = np.round(np.max(feature_values), 2)
        n_empty = feature_values.isnull().sum()

        plt.figure(figsize=(15, 5))
        sns.distplot(feature_values)
        plt.show()

        print('{:22} {:>10}'.format('N values', n_values))
        print('{:22} {:>10}'.format('N distinct', n_distinct))
        print('{:22} {:>10}'.format('Mean', mean))
        print('{:22} {:>10}'.format('Median', median))
        print('{:22} {:>10}'.format('Std Dev', std_dev))
        print('{:22} {:>10}'.format('Min', _min))
        print('{:22} {:>10}'.format('Max', _max))
        print('{:22} {:>10}'.format('N empty', n_empty))

    def _summary_stats_categorical(self, feature_values):
        n_values = len(feature_values)
        distincts_with_counts = feature_values.value_counts().reset_index()
        distincts_with_counts.columns = ['value', 'count']
        distincts_with_counts.loc[:, 'count_as_perc'] = np.round(distincts_with_counts['count'] / np.sum(distincts_with_counts['count']) * 100, 0)
        n_distinct = distincts_with_counts.shape[0]
        mode = distincts_with_counts['value'][0]
        n_empty = feature_values.isnull().sum()

        plt.figure(figsize=(15,5))
        sns.barplot(x='value', y='count', data=distincts_with_counts, order=distincts_with_counts['value'])
        plt.show()
        self._printmd('**Summary Stats**')
        print('{:22} {:>10}'.format('N values', n_values))
        print('{:22} {:>10}'.format('N distinct', n_distinct))
        print('{:22} {:>10}'.format('Mode', mode))
        print('{:22} {:>10}'.format('N empty', n_empty))
        self._printmd('**Frequency Stats**')
        for _, row in distincts_with_counts.iterrows():
            print('{:20} {:>10}% {:>10}'.format(row['value'], row['count_as_perc'], row['count']))


    def get_all_summary_stats(self):

        item_layout = widgets.Layout(margin='0 0 50px 0')

        # target widget
        targetoutput = widgets.Output()
        with targetoutput:
            if self.target_type == 'categorical':
                self._summary_stats_categorical(self._label_to_numeric())
            else:
                self._summary_stats_numeric(self.target)

        # numeric feature widget
        num_outputs = [widgets.Output() for col in self.numerical_columns]
        num_tab = widgets.Tab(num_outputs, layout=item_layout)
        index = 0
        for col in self.numerical_columns:
            with num_outputs[index]:
                self._summary_stats_numeric(self.features[col])
                self._printmd('**Weight of Evidence**')
                binned_df = self._calculate_bins_for_numeric(self.features[col], self._label_to_numeric())
                self._get_woe_and_iv(binned_df['bucket'], binned_df['target_values'])
            
            num_tab.set_title(index, f'{col}')
            index = index + 1
        
        # numeric feature widget
        cat_outputs = [widgets.Output() for col in self.categorical_columns]
        cat_tab = widgets.Tab(cat_outputs, layout=item_layout)
        index = 0
        for col in self.categorical_columns:
            with cat_outputs[index]:
                self._summary_stats_categorical(self.features[col])
                self._printmd('**Weight of Evidence**')
                self._get_woe_and_iv(self.features[col], self._label_to_numeric())
            
            cat_tab.set_title(index, f'{col}')
            index = index + 1

        accordion = widgets.Accordion(children=[targetoutput, num_tab, cat_tab])
        accordion.set_title(0, 'Target Label')
        accordion.set_title(1, 'Numerical Features')
        accordion.set_title(2, 'Categorical Features')

        the_box = widgets.HBox([accordion])
        display(the_box)

    
    def show_correlation_matrix(self, correlation_type='normal', with_target=False, remove_duplicate_half=False):

        if correlation_type != 'phik':
            df = self.features[self.numerical_columns]
        else: # phik correlation can deal with categorical columns
            df = self.features

        if with_target:
            df.loc[:, 'target'] = self._label_to_numeric()

        if correlation_type == 'normal':
            my_corr = df.corr()
        elif correlation_type == 'phik':
            my_corr = df.phik_matrix(interval_cols=self.numerical_columns)

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
        plt.yticks(rotation=0)
        plt.show()

    def check_distribution_of_feature(self, features):

        if isinstance(features, list) == False:
            features = [features]

        item_layout = widgets.Layout(margin='0 0 50px 0')
        outputs = [widgets.Output() for feature in features]
        tab = widgets.Tab(children=outputs, layout=item_layout)
        index = 0
        for feature in features:
            with outputs[index]:
                fit_distributions(self.features[feature].values)
            tab.set_title(index, f'{feature}')
            index = index + 1
        display(tab)
