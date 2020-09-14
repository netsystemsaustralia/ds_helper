import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display, Markdown
import ipywidgets as widgets

import shap

from sklearn.metrics import (
    roc_curve, auc, precision_score, recall_score,
    f1_score, precision_recall_curve, confusion_matrix
)

from sklearn.inspection import plot_partial_dependence, partial_dependence, permutation_importance
from sklearn.pipeline import _name_estimators, Pipeline

class PostAnalysis:

    def __init__(self, clf, preprocessor, features, y_true):
        self.clf = clf
        self.preprocessor = preprocessor
        self.features = features
        self.y_true = y_true
        self.transformed_features = self._get_post_processed_dataframe()
        self.optimal_threshold = 0.5
        self.shap_values = None
        self.shap_explainer = None
        self._create_predictions()
        shap.initjs()

    def _create_predictions(self):
        self.y_pred = self.clf.predict(self.transformed_features.values)
        self.y_pred_prob = self.clf.predict_proba(self.transformed_features.values)
        self.y_pred_pos_prob = self.y_pred_prob[:, 1]

    def _get_post_processed_dataframe(self):
        X_test_t = self.preprocessor.transform(self.features)
        all_names = self._get_names()
        X_test_df = pd.DataFrame(data=X_test_t, columns=all_names)
        return X_test_df

    def _get_names(self):
        all_names = []
        for t in self.preprocessor.transformers_:
            names = []
            if hasattr(t[1], 'named_steps'):
                for _, v in t[1].named_steps.items():
                    if hasattr(v, 'get_feature_names'):
                        names = v.get_feature_names(input_features=[t[0]])
                if len(names) == 0:
                    names = t[2]
                all_names.extend(names)
        
        return all_names
    
    def _plot_prob_dens(self, y_true, y_pred_prob, _feature, feature):
        x0 = y_pred_prob[(y_true == 0).tolist()][:, np.newaxis]
        x1 = y_pred_prob[(y_true == 1).tolist()][:, np.newaxis]
        plt.figure(figsize=(15, 10))
        sns.distplot(x0, hist=False, kde=True, kde_kws={'linewidth': 3, 'shade': True}, label='class False')
        sns.distplot(x1, hist=False, kde=True, kde_kws={'linewidth': 3, 'shade': True}, label='class True')
        plt.title(f'Probability Density Chart: {_feature}={feature}')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Probability Density')
        plt.show()

    def _show_model_performance(self, y_true, y_pred_prob):
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
        precision = precision[:-1]
        recall = recall[:-1]
        f1 = 2 * ((precision * recall) / (precision + recall))
        ix = np.argmax(f1)
        print('Best Threshold=%f, F_Score=%.3f' % (thresholds[ix], f1[ix]))
        plt.figure(figsize=(15, 5))
        d = pd.DataFrame({'Threshold': thresholds,
                        'Recall': recall,
                        'Precisioin': precision,
                        'F1': f1})
        sns.lineplot(x='Threshold', y='value', hue='variable', data=pd.melt(d, ['Threshold']))
        plt.scatter(thresholds[ix], f1[ix], marker='o', color='black', label='best')
        plt.show()

    def _get_auc(self, y, pred_prob):
        false_positive_rate, true_positive_rate, _ = roc_curve(y, pred_prob)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        return roc_auc

    def _get_metrics(self, y_true, y_pred_prob, threshold):
        roc_auc = np.round(self._get_auc(y_true, y_pred_prob), 4)
        pred = y_pred_prob >= threshold
        precision = np.round(precision_score(y_true, pred), 4)
        recall = np.round(recall_score(y_true, pred), 4)
        f1 = np.round(f1_score(y_true, pred), 4)
        return (roc_auc, precision, recall, f1)

    def _get_confusion_matrix(self, y_true, y_pred_prob, threshold):
        pred = y_pred_prob >= threshold
        display(pd.DataFrame(confusion_matrix(y_true, pred),
                columns=['Predicted Not Churn', 'Predicted to Churn'],
                index=['Actual Not Churn', 'Actual Churn']))

    def get_subpopulation_report_in_widget(self, _feature):
        features = self.features[_feature].unique()
        outputs = [widgets.Output() for feature in features]
        item_layout = widgets.Layout(margin='0 0 50px 0')
        tab = widgets.Tab(outputs, layout=item_layout)
        index = 0
        for feature in features:
            l = (self.features[_feature] == feature).tolist()
            ypp = self.y_pred_pos_prob[l]
            yt = self.y_true[l]
            with outputs[index]:
                print(f'Feature Instance: {feature}')
                print(self._get_metrics(yt, ypp, self.optimal_threshold))
                self._plot_prob_dens(yt, ypp, _feature, feature)
                self._get_confusion_matrix(yt, ypp, self.optimal_threshold)

            tab.set_title(index, f'Report for {feature}')
            index = index + 1

        display(tab)

    def get_lift_per_bin(self):
        lift_df = pd.DataFrame({'pred': self.y_true, 'prob': self.y_pred_pos_prob})
        lift_df.loc[:, 'prob_bin'] = pd.qcut(lift_df.loc[:, 'prob'], q=10, labels=False)
        lift_df.loc[:, 'prob_bin'] = (10 - lift_df.loc[:, 'prob_bin']).astype(pd.Int32Dtype())

        avg_perc = np.mean(self.y_true)
        t_df = lift_df.groupby(['prob_bin'])[['pred']].mean().reset_index()
        t_df.loc[:, 'lift'] = t_df.loc[:, 'pred'] / avg_perc

        plt.figure(figsize=(15, 5))
        graph = sns.barplot(x='prob_bin', y='lift', palette='Blues_d', data=t_df.sort_values('prob_bin', ascending=True))
        plt.axhline(1.0, linewidth=2, color='r', linestyle='dashed')
        for p in graph.patches:
            graph.annotate('{:.2f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()),
                    ha='center', va='bottom', color='black')

        plt.show()

    