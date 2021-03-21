import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_curve, auc, precision_score, 
    recall_score, f1_score
)

class EthicsTest:

    def __init__(self, features, y_pred_pos_prob, y_true, threshold):
        self.features = features
        self.y_pred_pos_prob = y_pred_pos_prob
        self.y_true = y_true
        self.threshold = threshold

        # set base metrics
        self._calc_base_metrics()

        # infer column types
        self.numerical_columns = list(self.features.select_dtypes(include=[np.number]).columns)
        self.categorical_columns = list(self.features.select_dtypes(include=['object']).columns)

    def _calc_base_metrics(self):
        self.base_metrics = self._get_metrics(self.y_true, self.y_pred_pos_prob)

    def _get_auc(self, y, pred_prob):
        false_positive_rate, true_positive_rate, _ = roc_curve(y, pred_prob)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        return roc_auc
    
    def _get_metrics(self, y_true, y_pred_prob):
        roc_auc = np.round(self._get_auc(y_true, y_pred_prob), 4)
        pred = y_pred_prob >= self.threshold
        precision = np.round(precision_score(y_true, pred), 4)
        recall = np.round(recall_score(y_true, pred), 4)
        f1 = np.round(f1_score(y_true, pred), 4)
        return {'auc': roc_auc, 'precision': precision, 'recall': recall, 'f1': f1}
    
    def _get_subpopulation_report(self, _feature):
        features = self.features[_feature].unique()
        stats = []

        for feature in features:
            l = (self.features[_feature] == feature).tolist()
            ypp = self.y_pred_pos_prob[l]
            yt = self.y_true[l]
            metrics = self._get_metrics(yt, ypp)
            #metrics['feature'] = feature
            stats.append(metrics)
        
        # how to test for imbalance? check standard deviation
        df = pd.DataFrame(stats)
        df.index = features.astype('str')
        _std_dev = np.round(df.iloc[:, 0:4].std(), 4)
        _mean = np.round(df.iloc[:, 0:4].mean(), 4)
        _cov = np.round(df.iloc[:, 0:4].std() / df.iloc[:, 0:4].mean(), 4)
        df.loc['mean'] = _mean
        df.loc['std'] = _std_dev
        df.loc['cov'] = _cov # coefficient of variation / relative standard deviation
        
        # for now check individually
        test_result_ok = True
        if _cov['auc'] >= 0.05:
            print('Possible imbalance in AUC for Feature {} (Coefficient of Variation. {})'.format(_feature, _cov['auc']))
            test_result_ok = False
        """ if _cov['precision'] >= 0.1:
            print('Possible imbalance in Precision for Feature {} (Coefficient of Variation. {})'.format(_feature, _cov['precision']))
            test_result_ok = False
        if _cov['recall'] >= 0.1:
            print('Possible imbalance in Recall for Feature {} (Coefficient of Variation. {})'.format(_feature, _cov['recall']))
            test_result_ok = False
        if _cov['f1'] >= 0.1:
            print('Possible imbalance in F1 for Feature {} (Coefficient of Variation. {})'.format(_feature, _cov['f1']))
            test_result_ok = False """

        if test_result_ok == False:
            print('----')
            print(df)


    def get_full_report(self):

        # for now just do categorical columns

        for col in self.categorical_columns:
            print('Ethics test on columns "{}"'.format(col))
            self._get_subpopulation_report(col)
            print('\n................................................\n')
        