import warnings

import numpy as np
import pandas as pd

from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.pca import PCA as PCA_od

import matplotlib.pyplot as plt
import seaborn as sns

import phik

from IPython.display import display, Markdown
import ipywidgets as widgets

from .stats_helper import fit_distributions, _optimal_bin_no

class PreAnalysis:
    """PreAnalysis object handles data analysis prior to modelling"""
    
    def __init__(self, features, target, use_optimal_binning=True):
        self.features = features
        self.target = target
        self.use_optimal_binning = use_optimal_binning
        self.target_type = 'categorical' if self.target.dtype == 'object' else 'numeric'
        self._infer_use_case()
        self.numerical_columns = list(self.features.select_dtypes(include=[np.number]).columns)
        self.categorical_columns = list(self.features.select_dtypes(include=['object']).columns)
        
    def _printmd(self, string: str):
        """Prints out markdown string as formatted text.
    
        Parameters
        ----------
        string : string
            The markdown string to be printed out.
        """
        display(Markdown(string))

    def _label_to_numeric(self, as_series: bool = True) -> pd.Series:
        """Converts target values to numeric.
    
        Parameters
        ----------
        as_series : bool
            Whether to return as a pdSeries or a numpy.ndarray, is optional, defaults to True

        Returns
        -------
        pd.Series or np.ndarray
            Returns numeric representation of the target values as either a pd.Series or np.ndarray.
    
        """
        if self.target_type == 'numeric':
            return self.target
        else:
            if as_series:
                return pd.Series(np.where(self.target == 'Yes', 1, 0))
            else:
                return np.where(self.target == 'Yes', 1, 0)
    
    def _get_iv_text(self, iv: float) -> str:
        """Retrieves the related text for an IV value.
    
        Parameters
        ----------
        iv : float
            The markdown string to be printed out.

        Returns
        -------
        str
            The related text of the provided IV value.
        """
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

    def _get_woe_and_iv(self, feature_values: pd.Series, target_values: pd.Series):
        """Calculates and prints out the IV (Information Value) and WOE (Weight of Evidence) values.

        WOE is a calculation of the predictive power of an independent variable in relation to the dependent variable.
        IV is calculated from the individual WOE scores that gives an overall importance value and can be used to rank
        the importance of this variable vs others.
    
        Parameters
        ----------
        feature_values : pd.Series
            The values of the feature in question.
        target_values : pd.Series
            The values of the target.
        """
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

    def _calculate_bins_for_numeric(self, feature_values: pd.Series, target_values: pd.Series) -> pd.DataFrame:
        """Creates bins for numeric feaure values.
    
        Parameters
        ----------
        feature_values : pd.Series
            The values of the feature in question.
        target_values : pd.Series
            The values of the target.

        Returns
        -------
        pd.DataFrame
            The resulting DataFrame with columns:
                - original feature values
                - original target values
                - the bucket label as a string
                - the bucket number
        """
        if self.use_optimal_binning:
            optimal_bin_no = _optimal_bin_no(feature_values, target_values)
            bins = np.histogram(feature_values, bins=optimal_bin_no)[1]
        else:
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

    def _summary_stats_numeric(self, feature_values: pd.Series):
        """Calculates metrics for the numeric feature.
    
        Parameters
        ----------
        feature_values : pd.Series
            The values of the feature in question.
        """
        n_values = len(feature_values)
        n_distinct = len(feature_values.unique())
        mean = np.round(np.mean(feature_values), 2)
        median = np.median(feature_values)
        std_dev = np.round(np.std(feature_values), 2)
        min_ = np.min(feature_values)
        max_ = np.max(feature_values)
        n_empty = feature_values.isnull().sum()

        plt.figure(figsize=(15,5))
        if self.use_optimal_binning:
            optimal_bin_no = _optimal_bin_no(feature_values, self._label_to_numeric())
            sns.displot(feature_values, bins=optimal_bin_no)
        else:
            sns.displot(feature_values)
        
        plt.show()

        #plt.figure(figsize=(15,5))
        #sns.boxplot(x=feature_values, orient='v')
        #plt.show()

        df = pd.DataFrame({'stat': ['N values','N distinct','Mean','Median', 'Std Dev', 'Min', 'Max', 'N empty'], 'value': [n_values, n_distinct, mean, median, std_dev, min_, max_, n_empty]})
        for _, row in df.iterrows():
            print('{:22} {:>10}'.format(row['stat'], row['value']))


    def _summary_stats_categorical(self, feature_values):
        """Calculates metrics for the categorical feature.
    
        Parameters
        ----------
        feature_values : pd.Series
            The values of the feature in question.
        """
        n_values = len(feature_values)
        distincts_with_counts = feature_values.value_counts().reset_index()
        distincts_with_counts.columns = ['value', 'count']
        distincts_with_counts.loc[:, 'count_as_perc'] = np.round(distincts_with_counts['count'] / np.sum(distincts_with_counts['count']) * 100, 0)
        n_distinct = distincts_with_counts.shape[0]
        distincts_with_counts.sort_values('count', ascending=False, inplace=True)
        mode = distincts_with_counts['value'][0]
        n_empty = feature_values.isnull().sum()

        # show historgram
        plt.figure(figsize=(15,5))
        sns.barplot(x='value', y='count', data=distincts_with_counts, order=distincts_with_counts['value'])
        plt.show()
        self._printmd('**Summary Stats**')
        df = pd.DataFrame({'stat': ['N values','N distinct','Mode','N empty'], 'value': [n_values, n_distinct, mode, n_empty]})
        for _, row in df.iterrows():
            print('{:32} {:>10}'.format(row['stat'], row['value']))
        #print(distincts_with_counts[['value','count_as_perc','count']].head().to_string(index=False, header=False))
        self._printmd('**Frequency Stats**')
        for _, row in distincts_with_counts.iterrows():
            print('{:20} {:>10}% {:>10}'.format(row['value'], row['count_as_perc'], row['count']))
            
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

    def _find_zero_or_near_zero_variance_columns(self, percent_unique_threshold=0.0, frequency_ratio_threshold=20, majority_threshold=0.95):
        data = self.features.copy()
        to_drop = []
        total_length = data.shape[0]
        for column in data.columns:
            unique_counts = data[column].value_counts() 
            percent_unique = len(unique_counts) / total_length
            if len(unique_counts) == 1:
                print("Column {0} has zero variance".format(column))
                to_drop.append(column)
            elif unique_counts.iloc[0] / total_length >= majority_threshold:
                print("Column {0} has near-zero variance of {1} (over the threshold of {2})".format(column, (unique_counts.iloc[0] / total_length), majority_threshold))
                to_drop.append(column)
            else:
                frequency_ratio = unique_counts.iloc[0] / unique_counts.iloc[1]
                if percent_unique <= percent_unique_threshold:
                    print("Column {0} has a percent unique of {1} (under the threshold of {2})".format(column, percent_unique, percent_unique_threshold))
                    to_drop.append(column)
                elif frequency_ratio >= frequency_ratio_threshold:
                    print("Column {0} has a frequency ration of {1} (over the threshold of {2})".format(column, frequency_ratio, frequency_ratio_threshold))
                    to_drop.append(column)
        
        return to_drop

    def _detect_outliers(self, contamination=0.05, methods=['knn','iso','pca']):
        # only do this on the numeric columns
        data = self.features[self.numerical_columns].copy()

        if 'knn' in methods:
            knn = KNN(contamination=contamination)
            knn.fit(data)
            knn_predict = knn.predict(data)
            data['knn'] = knn_predict
        
        if 'iso' in methods:
            iso = IForest(contamination=contamination, random_state=42, behaviour='new')
            iso.fit(data)
            iso_predict = iso.predict(data)
            data['iso'] = iso_predict

        if 'pca' in methods:
            pca = PCA_od(contamination=contamination, random_state=42)
            pca.fit(data)
            pca_predict = pca.predict(data)
            data['pca'] = pca_predict

        data['vote_outlier'] = 0
        
        for i in methods:
            data['vote_outlier'] = data['vote_outlier'] + data[i]
        

        # only select if all methods agree on outliers
        outliers = data[data['vote_outlier']== len(methods)]
        print('Outlier size {0}'.format(len(outliers)))
        
        print(data[[True if i in outliers.index else False for i in data.index]])

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

    def check_distribution_of_feature(self, features):

        # check if list of features or single feature
        if isinstance(features, list) == False:
            features = [features]

        # fix layout
        item_layout = widgets.Layout(margin='0 0 50px 0')
        # numeric feature widget
        outputs = [widgets.Output() for feature in features]
        # create tab widget
        tab = widgets.Tab(outputs, layout=item_layout)
        index = 0
        for feature in features:
            with outputs[index]:
                fit_distributions(self.features[feature].values)
            tab.set_title(index, f'{feature}')
            index = index + 1
        
        display(tab)

    def show_correlation_matrix(self, correlation_type='normal', with_target=False, remove_duplicate_half=False):

        if correlation_type != 'phik':
            df = self.features[self.numerical_columns]
        else:
            df = self.features

        if with_target:
            df.loc[:, 'target'] = self._label_to_numeric()
        
        if correlation_type == 'normal':
            my_corr = df.corr()
        else:
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
        plt.yticks(rotation = 0)
        plt.show()