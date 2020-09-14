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
        self.transformer_dictionary = self._get_transformer_dictionary()
        shap.initjs()

    def _get_transformer_dictionary(self):
        if self.preprocessor != None:
            transformer_dictionary = dict()

            for field, transformer, _ in self.preprocessor.transformers_:
                transformer_dictionary.setdefault(field, []).append(transformer)

            return transformer_dictionary
        else:
            return dict()

    def _create_predictions(self):
        self.y_pred = self.clf.predict(self.transformed_features.values)
        self.y_pred_prob = self.clf.predict_proba(self.transformed_features.values)
        self.y_pred_pos_prob = self.y_pred_prob[:, 1]
        self.mean_log_odds = np.mean(np.log(self.y_pred_prob[:, 1] / self.y_pred_prob[:, 0]))

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

    def _get_feature_importance_permutation(self):
        result = permutation_importance(self.clf, self.transformed_features.values, self.y_true, n_repeats=10,
                                random_state=42, n_jobs=2)
        importance_df = pd.DataFrame({'feature': self.transformed_features.columns,
                                    'importance': result.importances_mean})
        plt.figure(figsize=(15, 10))
        sns.set_color_codes('pastel')
        sns.barplot(x='importance', y='feature', 
            data=importance_df[importance_df['importance'] != 0.0].sort_values('importance', ascending=False),
            label='Importance', color='b')
        plt.show()

    def _get_feature_importance_from_model(self):
        importance_df = pd.DataFrame({'feature': self.transformed_features.columns,
                                    'importance': self.clf.feature_importances_})
        plt.figure(figsize=(15, 10))
        sns.set_color_codes('pastel')
        sns.barplot(x='importance', y='feature', 
            data=importance_df[importance_df['importance'] != 0.0].sort_values('importance', ascending=False),
            label='Importance', color='b')
        plt.show()

    def _get_feature_importance_from_shap(self):

        if self.shap_explainer is None:
            self.shap_explainer = shap.TreeExplainer(self.clf)
        if self.shap_values is None:
            self.shap_values = self.shap_explainer.shap_values(self.transformed_features)

        shap.summary_plot(self.shap_values, self.transformed_features)

    def get_feature_importance(self):
        outputs = [widgets.Output() for i in range(1, 4)]
        item_layout = widgets.Layout(margin='0 0 50px 0')
        tab = widgets.Tab(outputs, layout=item_layout)
        tab.set_title(0, 'From Model')
        tab.set_title(1, 'From Permutation')
        tab.set_title(2, 'From Shap')

        with outputs[0]:
            self._get_feature_importance_from_model()

        with outputs[1]:
            self._get_feature_importance_permutation()

        with outputs[2]:
            self._get_feature_importance_from_shap()

        display(tab)

    def plot_pdp(self, feature, y_pct=True, norm_hist=True, dec=0.5):
        feature_index = self.transformed_features.columns.tolist().index(feature)

        pardep = partial_dependence(self.clf, self.transformed_features.values, [feature_index])

        pardep_x = pardep[1][0]
        pardep_y = pardep[0][0]

        if feature in self.transformer_dictionary:
            transformer = self.transformer_dictionary[feature][0].named_steps['scaler']
            pardep_x = transformer.inverse_transform(pardep_x)
            x = transformer.inverse_transform(self.transformed_features[feature])
        else:
            x = self.transformed_features[feature]

        pardep_y = np.log(pardep_y / (1 - pardep_y))
        pardep_y = pardep_y - self.mean_log_odds

        xmin = pardep_x.min()
        xmax = pardep_x.max()
        ymin = pardep_y.min()
        ymax = pardep_y.max()

        fig, ax1 = plt.subplots(figsize=(15,10))
        ax1.grid(alpha=0.5, linewidth=1)

        color = 'tab:blue'
        ax1.plot(pardep_x, pardep_y, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xlabel(feature, fontsize=14)
        ax1.set_ylabel('Partial Dependence', color=color, fontsize=14)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.hist(x, bins=50, range=(xmin, xmax), alpha=0.25, color=color, density=norm_hist)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylabel('Distribution', color=color, fontsize=14)

        plt.show()

        