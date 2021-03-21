import warnings
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import shap

from sklearn.metrics import (
    roc_curve, auc, precision_score, recall_score, 
    f1_score, precision_recall_curve, confusion_matrix
)

from IPython.display import display, Markdown
import ipywidgets as widgets

from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType, FloatType

from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler, StandardScaler, QuantileDiscretizer
from pyspark.ml import Pipeline

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

class PostAnalysis:

    def __init__(self, clf, preprocessor, test_df):
        self.clf = clf
        self.preprocessor = preprocessor # do we need that one??
        self.test_df = test_df # assume is already transformed? bring that one in latter
        self.optimal_threshold = 0.5 # TBD determine best threshold later
        self.shap_values = None # for caching
        self.shap_explainer = None # for caching

        # other initialisation steps
        #self._create_predictions()
        self._create_predictions_rdd() # stores prediction values in DataFrame
        self._get_precision_recall_curve() # this is for all, no sub populations yet...
        self._set_best_threshold()
        shap.initjs()

    def _set_best_threshold(self, method='f1opt'):

        if method == 'f1opt':
            #precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_pred_pos_prob)
            #precision = precision[:-1]
            #recall = recall[:-1]
            #f1 = 2 * ((precision * recall) / (precision + recall))
            ix = np.argmax(self.stats_df['f1'].values)
            self.optimal_threshold = self.stats_df['threshold'].values[ix]
            self.optimal_threshold_f1 = self.stats_df['f1'].values[ix]
            self.optimal_threshold_precision = self.stats_df['precision'].values[ix]
            self.optimal_threshold_recall = self.stats_df['recall'].values[ix]
        else:
            pass
    
    def _printmd(self, string):
        display(Markdown(string))
    
    def _get_mean_log_odds(self, prob):
        return np.mean(np.log(prob / (1 - prob)))

    def _get_precision_recall_curve(self):

        a = []

        print('Calculating metrics for different thresholds...')
        i = 0
        for threshold in np.arange(0.05, 1.0, 0.01):
            temp_df = self.predictions_df\
                .withColumn('pred', F.when(F.col('pos_prob') >= threshold, 1).otherwise(0))\
                .withColumn('TP', F.when((F.col('label') == 1)  & (F.col('pred') == 1), 1).otherwise(0))\
                .withColumn('FP', F.when((F.col('label') == 0)  & (F.col('pred') == 1), 1).otherwise(0))\
                .withColumn('TN', F.when((F.col('label') == 0)  & (F.col('pred') == 0), 1).otherwise(0))\
                .withColumn('FN', F.when((F.col('label') == 1)  & (F.col('pred') == 0), 1).otherwise(0))
            
            #temp_df.show()
                        
            stats = temp_df.agg(F.sum('TP').alias('TP_TOT'),
                F.sum('FP').alias('FP_TOT'),
                F.sum('TN').alias('TN_TOT'),
                F.sum('FN').alias('FN_TOT')).toPandas()

            # check for 'empty' values

            if (stats.iloc[0, 0] > 0) | (stats.iloc[0, 1] > 0):
                a.append({
                    'threshold': threshold,
                    'TP': stats.iloc[0, 0],
                    'FP': stats.iloc[0, 1],
                    'TN': stats.iloc[0, 2],
                    'FN': stats.iloc[0, 3]
                })

            i += 1
            if i % 10 == 0:
                print('.')

        stats_df = pd.DataFrame(a)

        # now add precision, etc

        stats_df.loc[:, 'precision'] = stats_df.loc[:, 'TP'] / (stats_df.loc[:, 'TP'] + stats_df.loc[:, 'FP'])
        stats_df.loc[:, 'recall'] = stats_df.loc[:, 'TP'] / (stats_df.loc[:, 'TP'] + stats_df.loc[:, 'FN'])
        stats_df.loc[:, 'tpr'] = stats_df.loc[:, 'recall']
        stats_df.loc[:, 'fpr'] = stats_df.loc[:, 'FP'] / (stats_df.loc[:, 'FP'] + stats_df.loc[:, 'TN'])
        stats_df.loc[:, 'f1'] = 2 * ((stats_df.loc[:, 'precision'] * stats_df.loc[:, 'recall']) / (stats_df.loc[:, 'precision'] + stats_df.loc[:, 'recall']))

        
        self.stats_df = stats_df.copy()
        return stats_df.copy()
            

    def _create_predictions_rdd(self):
        predictions = self.clf.transform(self.test_df).select('label', 'rawPrediction', 'prediction', 'probability')
        _udf = F.udf(lambda x: float(x[1]), FloatType())
        predictions = predictions.withColumn('pos_prob', _udf(predictions.probability))
        
        self.predictions_df = predictions

        evaluator = BinaryClassificationEvaluator()
        self.auc_roc = evaluator.evaluate(predictions)
        self.auc_pr = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"})

    def _plot_prob_dens(self):
        sample_df = self.predictions_df.sample(False, 0.5, 42)
        x0 = sample_df.select('pos_prob').where(F.col('label') == 0).toPandas()
        x1 = sample_df.select('pos_prob').where(F.col('label') == 1).toPandas()
        plt.figure(figsize=(4, 4))
        sns.distplot(x0, hist = False, kde = True,
                        kde_kws = {'linewidth': 3, 'shade': True},
                        label = 'class False')
        sns.distplot(x1, hist = False, kde = True,
                        kde_kws = {'linewidth': 3, 'shade': True},
                        label = 'class True')
        plt.title(f'Probability Density Chart')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Probability Density')
        plt.show()
    
    def _show_model_performance(self):
        plt.figure(figsize=(20, 5))
        sns.lineplot(x='threshold', y='value', hue='variable', data=pd.melt(self.stats_df[['threshold', 'precision', 'recall', 'f1']], ['threshold']))
        plt.scatter(self.optimal_threshold, self.optimal_threshold_f1, marker='o', color='black', label='Best')
        plt.show()

    def _show_auc_curve(self):
        plt.figure(figsize=(4,4))
        sns.lineplot(x=self.stats_df['fpr'].values, y=self.stats_df['tpr'].values)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (AUC: %.3f)' % self.auc_roc)
        plt.show()

    def _show_pr_curve(self):
        plt.figure(figsize=(4,4))
        sns.lineplot(x=self.stats_df['recall'].values, y=self.stats_df['precision'].values)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'PR Curve (AUC: %.3f)' % self.auc_pr)
        plt.show()
    
    def _get_metrics(self):
        roc_auc = np.round(self.auc_roc, 4)
        precision = np.round(self.optimal_threshold_precision, 4)
        recall = np.round(self.optimal_threshold_recall, 4)
        f1 = np.round(self.optimal_threshold_f1, 4)
        return (roc_auc, precision, recall, f1)
        
    def _get_confusion_matrix(self):
        t_df = self.stats_df[self.stats_df['threshold'] == self.optimal_threshold].iloc[0]
        total_size = t_df['TP'] + t_df['FP'] + t_df['TN'] + t_df['FN']
        df = pd.DataFrame([[t_df['TN'], t_df['FP']], [t_df['FN'], t_df['TP']]], 
                columns=['Predicted Not Churn', 'Predicted to Churn'],
                index=['Actual Not Churn', 'Actual Churn'])
        df.loc['Total Actual'] = df.sum(numeric_only=True, axis=0)
        df.loc[:, 'Total Predicted'] = df.sum(numeric_only=True, axis=1)
        df.loc[:, 'Predicted Not Churn'] = df.loc[:, 'Predicted Not Churn'].apply(lambda x: '%.1f%% (%d)' % (x / total_size * 100, x))
        df.loc[:, 'Predicted to Churn'] = df.loc[:, 'Predicted to Churn'].apply(lambda x: '%.1f%% (%d)' % (x / total_size * 100, x))
        df.loc[:, 'Total Predicted'] = df.loc[:, 'Total Predicted'].apply(lambda x: '%.1f%% (%d)' % (x / total_size * 100, x))
        display(df)

    def show_performance(self):
        
        item_layout = widgets.Layout(margin='0 0 50px 0')
        output = widgets.Output(layout=item_layout)

        with output:
            _metrics = self._get_metrics()
            print('AUC: %.3f - Precision: %.3f - Recall: %.3f - F1: %.3f' % (_metrics[0], _metrics[1], _metrics[2], _metrics[3]))
            self._show_model_performance()
            mini_output = [widgets.Output() for _ in range(1, 5)]
            with mini_output[0]:
                self._plot_prob_dens()
            with mini_output[1]:
                self._show_auc_curve()
            with mini_output[2]:
                self._show_pr_curve()
            with mini_output[3]:
                self._printmd('**Confusion Matrix (threshold set to %.2f)**' % self.optimal_threshold)
                self._get_confusion_matrix()
            display(widgets.HBox(mini_output))
        
        display(output)

    def get_lift_per_bin(self):
        # TBD move this to _create_predictions_rdd function
        discretizer = QuantileDiscretizer(numBuckets=10, inputCol="pos_prob", outputCol="pos_prob_q")
        df_r = self.predictions_df.select('label', 'pos_prob')
        df_r = discretizer.fit(df_r).transform(df_r)
        # calculate lift per bin
        
        avg_perc = 0.2 # TBD later!

        t_df = df_r.groupBy(F.col('pos_prob_q')).agg(F.mean(F.col('label')).alias('pred')).toPandas()
        print(t_df)
        
        t_df.loc[:, 'lift'] = t_df.loc[:, 'pred'] / avg_perc
        plt.figure(figsize=(15,5))
        graph = sns.barplot(x='pos_prob_q', y='lift', palette='Blues_d', data=t_df.sort_values('pos_prob_q', ascending=True))
        plt.axhline(1.0, linewidth=2, color='r', linestyle='dashed')
        for p in graph.patches:
            graph.annotate('{:.2f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()),
                        ha='center', va='bottom',
                        color= 'black')
        plt.show()