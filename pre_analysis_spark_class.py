import warnings

import numpy as np
import pandas as pd

from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.pca import PCA as PCA_od

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display, Markdown
import ipywidgets as widgets

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, IntegerType

from pyspark.mllib.stat import Statistics

class PreAnalysis:
    
    def __init__(self, spark_dataframe, target_name, sampling_size=0.5, sampling_random_seed=42):
        self.spark_dataframe = spark_dataframe
        self.target_name = target_name
        self.sampling_size = sampling_size
        self.sampling_random_seed = sampling_random_seed
        self.feature_names = [col for col in self.spark_dataframe.columns if col != self.target_name]
        self.numerical_columns = [t[0] for t in self.spark_dataframe.select(self.feature_names).dtypes if t[1] == 'int' or t[1] == 'double']
        self.categorical_columns = [col for col in self.spark_dataframe.select(self.feature_names).columns if col not in self.numerical_columns]

        # run initialisation functions
        self._add_numeric_target_columns()
        self._create_stats_dataframe()

    def _add_numeric_target_columns(self):
        _udf = udf(lambda x: 1 if x == 'Yes' else 0, IntegerType())
        self.spark_dataframe = self.spark_dataframe.withColumn('numeric_target', _udf(col(self.target_name)))

    def _create_stats_dataframe(self):
        stats_df = self.spark_dataframe.select(self.numerical_columns).describe().toPandas().T
        new_header = stats_df.iloc[0]
        new_df = stats_df.iloc[1:]
        new_df.columns = new_header
        self.stats_df = new_df.astype(np.float)

    def _printmd(self, string):
        display(Markdown(string))
    
    def _sample_for_pandas(self, column_name):
        return self.spark_dataframe.select(column_name).sample(False, self.sampling_size, self.sampling_random_seed).toPandas()
    
    def _summary_stats_numeric(self, feature_name):
        n_values = self.stats_df.loc[feature_name, 'count']
        n_distinct = self.spark_dataframe.select(feature_name).distinct().count()
        mean = np.round(self.stats_df.loc[feature_name, 'mean'], 2)
        median = self.spark_dataframe.approxQuantile(feature_name, [0.5], 0.05)[0]
        std_dev = np.round(self.stats_df.loc[feature_name, 'stddev'], 2)
        min_ = self.stats_df.loc[feature_name, 'min']
        max_ = self.stats_df.loc[feature_name, 'max']
        n_empty = self.spark_dataframe.where(col(feature_name).isNull()).count()

        # sample for Plotting
        sample_df = self._sample_for_pandas(feature_name)

        plt.figure(figsize=(15,5))
        sns.distplot(sample_df)
        plt.show()

        plt.figure(figsize=(15,5))
        sns.boxplot(x=sample_df, orient='v')
        plt.show()

        df = pd.DataFrame({'stat': ['N values','N distinct','Mean','Median (Approx.)', 'Std Dev', 'Min', 'Max', 'N empty'], 'value': [n_values, n_distinct, mean, median, std_dev, min_, max_, n_empty]})
        for _, row in df.iterrows():
            print('{:22} {:>10}'.format(row['stat'], row['value']))

    def _summary_stats_categorical(self, feature_name):
        n_values = self.spark_dataframe.select(feature_name).count()
        distincts_with_counts = self.spark_dataframe.groupBy(feature_name).count().toPandas()
        distincts_with_counts.columns = ['value', 'count']
        distincts_with_counts.loc[:, 'count_as_perc'] = np.round(distincts_with_counts['count'] / np.sum(distincts_with_counts['count']) * 100, 0)
        distincts_with_counts.sort_values('count', ascending=False, inplace=True)
        n_distinct = distincts_with_counts.shape[0]
        mode = distincts_with_counts['value'][0]
        n_empty = self.spark_dataframe.where(col(feature_name).isNull()).count()

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

    def get_all_summary_stats(self):
        # setup widgets
        # fix layout
        item_layout = widgets.Layout(margin='0 0 50px 0')

        # target feature widget
        targetoutput = widgets.Output()
        # create tab widget
        with targetoutput:
            #if self.target_type == 'categorical':
                #self._summary_stats_categorical(self._label_to_numeric())
            self._summary_stats_categorical(self.target_name)
            #else: # TBD 
            #    self._summary_stats_numeric(self.target)

        # numeric feature widget
        num_outputs = [widgets.Output() for numerical_column in self.numerical_columns]
        # create tab widget
        num_tab = widgets.Tab(num_outputs, layout=item_layout)
        index = 0
        for numerical_column in self.numerical_columns:
            with num_outputs[index]:
                self._summary_stats_numeric(numerical_column)
                self._printmd('**Weight of Evidence**')
                #binned_df = self._calculate_bins_for_numeric(self.features[numerical_column], self._label_to_numeric())
                #self._get_woe_and_iv(binned_df['bucket'], binned_df['target_values'])

            num_tab.set_title(index, f'{numerical_column}')
            index = index + 1
        
        # categorical feature widget
        cat_outputs = [widgets.Output() for categorical_column in self.categorical_columns]
        # create tab widget
        cat_tab = widgets.Tab(cat_outputs, layout=item_layout)
        index = 0
        for categorical_column in self.categorical_columns:
            with cat_outputs[index]:
                self._summary_stats_categorical(categorical_column)
                self._printmd('**Weight of Evidence**')
                #self._get_woe_and_iv(self.features[categorical_column], self._label_to_numeric())
            
            cat_tab.set_title(index, f'{categorical_column}')
            index = index + 1
        
        accordion = widgets.Accordion(children=[targetoutput, num_tab, cat_tab])
        accordion.set_title(0, 'Target Label')
        accordion.set_title(1, 'Numeric Features')
        accordion.set_title(2, 'Categorical Features')

        the_box = widgets.HBox([accordion])

        display(the_box)

    def show_correlation_matrix(self, with_target=False, remove_duplicate_half=False):

        features_for_correlation = self.numerical_columns.copy()
        
        
        if with_target:
            features_for_correlation.append('numeric_target')

        df_features = self.spark_dataframe.select(features_for_correlation)
        rdd_table = df_features.rdd.map(lambda row: row[0:])

        my_corr = Statistics.corr(rdd_table, method='pearson')
        df_corr = pd.DataFrame(my_corr, index=features_for_correlation, columns=features_for_correlation)

        params = {
            'annot': True,
            'vmin': -1,
            'vmax': 1,
            'center': 0,
            'cmap': 'coolwarm'
        }

        if remove_duplicate_half:
            mask = np.triu(np.ones_like(df_corr, dtype=bool))
            params['mask'] = mask

        plt.figure(figsize=(15,5))
        sns.heatmap(df_corr, **params)
        plt.yticks(rotation = 0)
        plt.show()