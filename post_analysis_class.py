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
from sklearn.inspection import plot_partial_dependence, partial_dependence, permutation_importance
from sklearn.pipeline import _name_estimators, Pipeline

from IPython.display import display, Markdown
import ipywidgets as widgets

from .utils import _printmd

class PostAnalysis:

    def __init__(self, clf, preprocessor, features, y_true):
        self.clf = clf
        self.preprocessor = preprocessor
        self.features = features
        self.y_true = y_true
        self.transformed_features = self._get_post_processed_dataframe()
        self.optimal_threshold = 0.5 # TBD determine best threshold later
        self.shap_values = None # for caching
        self.shap_explainer = None # for caching

        # other initialisation steps
        self._create_predictions()
        self.transformer_dictionary = self._get_transformer_dictionary()
        self._set_best_threshold()
        shap.initjs()

    def _get_transformer_dictionary(self):
        if self.preprocessor != None:
            transformer_dictionary = dict()

            for field, transformer, _ in self.preprocessor.transformers_:
                transformer_dictionary.setdefault(field, []).append(transformer)

            return transformer_dictionary
        else:
            return dict()
    
    def _get_mean_log_odds(self, prob):
        return np.mean(np.log(prob / (1 - prob)))
    
    def _create_predictions(self):
        self.y_pred = self.clf.predict(self.transformed_features.values)
        self.y_pred_prob = self.clf.predict_proba(self.transformed_features.values)
        self.y_pred_pos_prob = self.y_pred_prob[:,1]
        self.mean_log_odds = self._get_mean_log_odds(self.y_pred_prob[:, 1])
        self.mean_pred = np.mean(self.y_pred)
        self.mean_pred_prob = np.mean(self.y_pred_pos_prob)
        self.mean_log_pred_prob = self._get_mean_log_odds(self.y_pred_pos_prob)
        self.mean_true = np.mean(self.y_true)
        self.mean_log_true = self._get_mean_log_odds(self.y_true)
    
    def _set_best_threshold(self, method='f1opt'):

        if method == 'f1opt':
            precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_pred_pos_prob)
            precision = precision[:-1]
            recall = recall[:-1]
            f1 = 2 * ((precision * recall) / (precision + recall))
            ix = np.argmax(f1)
            self.optimal_threshold = thresholds[ix]
        else:
            pass
    
    def _get_post_processed_dataframe(self):
        """
        Take the preprocessor from the Pipeline and create a post_processed_dataframe on the test features
        """
        X_test_t = self.preprocessor.transform(self.features)
        all_names = self._get_names()
        X_test_df = pd.DataFrame(data = X_test_t, columns = all_names)
        return X_test_df

    def _get_names(self):
        """
        Return verbose names for the transformed columns.
        """
        all_names = []
        for t in self.preprocessor.transformers_:
            names = []
            if hasattr(t[1],'named_steps'):
                for _, v in t[1].named_steps.items():
                    #print(v)
                    if hasattr(v,'get_feature_names'):
                        names = v.get_feature_names(input_features=[t[0]])
                if len(names) == 0:
                    names = t[2]
                all_names.extend(names)
        
        return all_names

    # TBD move to new class
    def show_values_on_bars(self, axs, h_v="v", space=0.4):
        def _show_on_single_plot(ax):
            if h_v == "v":
                for p in ax.patches:
                    _x = p.get_x() + p.get_width() / 2
                    _y = p.get_y() + p.get_height()
                    value = int(p.get_height())
                    ax.text(_x, _y, value, ha="center") 
            elif h_v == "h":
                for p in ax.patches:
                    _x = p.get_x() + p.get_width() + float(space)
                    _y = p.get_y() + p.get_height()
                    value = int(p.get_width())
                    ax.text(_x, _y, value, ha="left")

        if isinstance(axs, np.ndarray):
            for _, ax in np.ndenumerate(axs):
                _show_on_single_plot(ax)
        else:
            _show_on_single_plot(axs)

    def _plot_prob_dens(self, y_true, y_pred_prob, _feature, feature):
        x0 = y_pred_prob[(y_true == 0).tolist()][:, np.newaxis]
        x1 = y_pred_prob[(y_true == 1).tolist()][:, np.newaxis]
        plt.figure(figsize=(4, 4))
        sns.distplot(x0, hist = False, kde = True,
                        kde_kws = {'linewidth': 3, 'shade': True},
                        label = 'class False')
        sns.distplot(x1, hist = False, kde = True,
                        kde_kws = {'linewidth': 3, 'shade': True},
                        label = 'class True')
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
        print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], f1[ix]))
        plt.figure(figsize=(20, 5))
        d = pd.DataFrame({'Threshold': thresholds, 
                        'Recall': recall, 
                        'Precision': precision,
                        'F1': f1})
        sns.lineplot(x='Threshold', y='value', hue='variable', data=pd.melt(d, ['Threshold']))
        plt.scatter(thresholds[ix], f1[ix], marker='o', color='black', label='Best')
        plt.show()

    def _get_auc(self, y, pred_prob):
        false_positive_rate, true_positive_rate, _ = roc_curve(y, pred_prob)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        return roc_auc

    def _show_auc_curve(self, y_true, y_pred_prob):
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_pred_prob)
        _auc = auc(false_positive_rate, true_positive_rate)
        plt.figure(figsize=(4,4))
        sns.lineplot(x=false_positive_rate, y=true_positive_rate)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (AUC: %.3f)' % _auc)
        plt.show()

    def _show_pr_curve(self, y_true, y_pred_prob):
        precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
        _auc = auc(recall, precision)
        plt.figure(figsize=(4,4))
        sns.lineplot(x=recall, y=precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'PR Curve (AUC: %.3f)' % _auc)
        plt.show()
    
    def _get_metrics(self, y_true, y_pred_prob, threshold):
        roc_auc = np.round(self._get_auc(y_true, y_pred_prob), 4)
        pred = y_pred_prob >= threshold
        precision = np.round(precision_score(y_true, pred), 4)
        recall = np.round(recall_score(y_true, pred), 4)
        f1 = np.round(f1_score(y_true, pred), 4)
        return (roc_auc, precision, recall, f1)
        
    def _get_confusion_matrix(self, y_true, y_pred_prob, threshold):
        pred = y_pred_prob >= threshold
        total_size = len(y_true)
        df = pd.DataFrame(confusion_matrix(y_true, pred), 
                columns=['Predicted Not Churn', 'Predicted to Churn'],
                index=['Actual Not Churn', 'Actual Churn'])
        df.loc['Total Actual'] = df.sum(numeric_only=True, axis=0)
        df.loc[:, 'Total Predicted'] = df.sum(numeric_only=True, axis=1)
        df.loc[:, 'Predicted Not Churn'] = df.loc[:, 'Predicted Not Churn'].apply(lambda x: '%.1f%% (%d)' % (x / total_size * 100, x))
        df.loc[:, 'Predicted to Churn'] = df.loc[:, 'Predicted to Churn'].apply(lambda x: '%.1f%% (%d)' % (x / total_size * 100, x))
        df.loc[:, 'Total Predicted'] = df.loc[:, 'Total Predicted'].apply(lambda x: '%.1f%% (%d)' % (x / total_size * 100, x))
        display(df)

    def show_performance(self):
        ypp = self.y_pred_pos_prob
        yt = self.y_true
        item_layout = widgets.Layout(margin='0 0 50px 0')
        output = widgets.Output(layout=item_layout)

        with output:
            _metrics = self._get_metrics(yt, ypp, self.optimal_threshold)
            print('AUC: %.3f - Precision: %.3f - Recall: %.3f - F1: %.3f' % (_metrics[0], _metrics[1], _metrics[2], _metrics[3]))
            self._show_model_performance(yt, ypp)
            mini_output = [widgets.Output() for _ in range(1, 5)]
            with mini_output[0]:
                self._plot_prob_dens(yt, ypp, 'ALL', 'ALL')
            with mini_output[1]:
                self._show_auc_curve(yt, ypp)
            with mini_output[2]:
                self._show_pr_curve(yt, ypp)
            with mini_output[3]:
                _printmd('**Confusion Matrix (threshold set to %.2f)**' % self.optimal_threshold)
                self._get_confusion_matrix(yt, ypp, self.optimal_threshold)
            display(widgets.HBox(mini_output))
        
        display(output)

    
    def get_subpopulation_report_in_widget(self, _feature):
        features = self.features[_feature].unique()
        # create outputs for each numerical_columns
        outputs = [widgets.Output() for feature in features]
        # fix layout
        item_layout = widgets.Layout(margin='0 0 50px 0')
        # create tab widget
        tab = widgets.Tab(outputs, layout=item_layout)
        index = 0

        for feature in features:
            l = (self.features[_feature] == feature).tolist()
            ypp = self.y_pred_pos_prob[l]
            yt = self.y_true[l]
            with outputs[index]:
                print(f'Feature Instance: {feature}')
                _metrics = self._get_metrics(yt, ypp, self.optimal_threshold)
                print('AUC: %.3f - Precision: %.3f - Recall: %.3f - F1: %.3f' % (_metrics[0], _metrics[1], _metrics[2], _metrics[3]))
                mini_output = [widgets.Output() for _ in range(1, 5)]
                with mini_output[0]:
                    self._plot_prob_dens(yt, ypp, _feature, feature)
                with mini_output[1]:
                    self._show_auc_curve(yt, ypp)
                with mini_output[2]:
                    self._show_pr_curve(yt, ypp)
                with mini_output[3]:
                    _printmd('**Confusion Matrix (threshold set to %.2f)**' % self.optimal_threshold)
                    self._get_confusion_matrix(yt, ypp, self.optimal_threshold)
                display(widgets.HBox(mini_output))
            
            tab.set_title(index, f'Report for {feature}')
            index = index + 1

        display(tab)
    
    def get_lift_per_bin(self):
        # calculate lift per bin
        lift_df = pd.DataFrame({'pred': self.y_true, 'prob': self.y_pred_pos_prob})
        lift_df.loc[:,'prob_bin'] = pd.qcut(lift_df.loc[:,'prob'], q=10, labels=False)
        # reverse bin numbers
        lift_df.loc[:,'prob_bin'] = (10 - lift_df.loc[:,'prob_bin']).astype(pd.Int32Dtype())
        avg_perc = np.mean(self.y_true)
        t_df = lift_df.groupby(['prob_bin'])[['pred']].mean().reset_index()
        t_df.loc[:, 'lift'] = t_df.loc[:, 'pred'] / avg_perc
        plt.figure(figsize=(15,5))
        graph = sns.barplot(x='prob_bin', y='lift', palette='Blues_d', data=t_df.sort_values('prob_bin', ascending=True))
        plt.axhline(1.0, linewidth=2, color='r', linestyle='dashed')
        for p in graph.patches:
            graph.annotate('{:.2f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()),
                        ha='center', va='bottom',
                        color= 'black')
        plt.show()

    def _get_feature_importance_permutation(self):
        result = permutation_importance(self.clf, self.transformed_features.values, self.y_true, n_repeats=10,
                                    random_state=42, n_jobs=2)
        importance_df = pd.DataFrame({'feature': self.transformed_features.columns, 'importance': result.importances_mean})
        plt.figure(figsize=(15, 10))
        sns.set_color_codes("pastel")
        sns.barplot(x="importance", y="feature", 
                data=importance_df[importance_df['importance'] != 0.0].sort_values('importance', ascending=False),
                label="Importance", color="b")
        plt.show()

    def _get_feature_importance_from_model(self):
        # feature importance
        importance_df = pd.DataFrame({'feature': self.transformed_features.columns, 'importance': self.clf.feature_importances_})
        # summarize feature importance
        plt.figure(figsize=(15,10))
        sns.set_color_codes("pastel")
        sns.barplot(x="importance", y="feature", data=importance_df[importance_df['importance'] > 0.0].sort_values('importance', ascending=False),
                    label="Importance", color="b")
        plt.show()

    def _get_feature_importance_use_shapley(self):
        
        # get these values the first time otherwise get it from the 'cache'
        
        if self.shap_explainer is None:
            self.shap_explainer = shap.TreeExplainer(self.clf)
        if self.shap_values is None:
            self.shap_values = self.shap_explainer.shap_values(self.transformed_features)
        
        shap.summary_plot(self.shap_values, self.transformed_features)

    def get_feature_importance(self):
        outputs = [widgets.Output() for i in range(1,4)]
        # fix layout
        item_layout = widgets.Layout(margin='0 0 50px 0')
        # create tab widget
        tab = widgets.Tab(outputs, layout=item_layout)
        tab.set_title(0, 'From Model')
        tab.set_title(1, 'From Permutation')
        tab.set_title(2, 'From SHAP')

        with outputs[0]:
            self._get_feature_importance_from_model()
        
        with outputs[1]:
            self._get_feature_importance_permutation()

        with outputs[2]:
            self._get_feature_importance_use_shapley()

        display(tab)

    def save_results(self):
        import datetime
        # gather information first
        lines = []
        ts = datetime.datetime.now()
        file_name = 'results_{}_{}_{}_{}_{}_{}.txt'.format(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second)

        model_type = type(self.clf)
        model_params = self.clf.get_params()
        (roc_auc, precision, recall, f1) = self._get_metrics(self.y_true, self.y_pred_pos_prob, self.optimal_threshold)

        lines.append(f'Timestamp,{ts}\n')
        lines.append(f'Model,{model_type}\n')
        lines.append(f'Model Parameters\n')

        for key in model_params:
            lines.append('{},{}\n'.format(key, model_params[key]))

        lines.append('\nMetrics\n')

        lines.append(f'ROC,{roc_auc}\n')
        lines.append(f'Precision,{precision}\n')
        lines.append(f'Recall,{recall}\n')
        lines.append(f'F1,{f1}\n')

        fout = open(file_name,'w')
        fout.writelines(lines)
        fout.close()

    def plot_pdp(self, feature, y_pct=True, norm_hist=True, log_odds=False):
        # get feature_index
        feature_index = self.transformed_features.columns.tolist().index(feature)
        
        # Get partial dependence
        pardep = partial_dependence(self.clf, self.transformed_features.values, [feature_index])

        # x-values
        pardep_x = pardep[1][0]

        # y-values
        pardep_y = pardep[0][0]

        # translate X-axis values back to original values if have transformer
        if feature in self.transformer_dictionary:
            transformer = self.transformer_dictionary[feature][0].named_steps['scaler']
            pardep_x = transformer.inverse_transform(pardep_x)
            x = transformer.inverse_transform(self.transformed_features[feature])
        else:
            x = self.transformed_features[feature]
        
        
        if log_odds:
            # transform y-axis values to difference of log-odds to average
            pardep_y = np.log(pardep_y / (1 - pardep_y))
            avg_y = self.mean_log_odds
            pardep_y = pardep_y - self.mean_log_odds
        else:
            avg_y = self.mean_pred
            pardep_y = pardep_y - self.mean_pred

        # Get min & max values
        xmin = pardep_x.min()
        xmax = pardep_x.max()
        #ymin = pardep_y.min()
        #ymax = pardep_y.max()

            
        # Create figure
        _, ax1 = plt.subplots(figsize=(15,10))
        ax1.grid(alpha=.5, linewidth=1)
        
        # Plot partial dependence
        color = 'tab:blue'
        ax1.plot(pardep_x, pardep_y, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xlabel(feature, fontsize=14)
        ax1.set_ylabel('Partial Dependence', color=color, fontsize=14)
        
        #ax1.set_title('Relationship Between {} and Target Variable'.format(feature), fontsize=16)
        
        # Plot line for decision boundary
        #ax1.hlines(avg_y, xmin=xmin, xmax=xmax, color='black', linewidth=2, linestyle='--', label='Avg. Prediction')
        #ax1.legend()

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.hist(x, bins=50, range=(xmin, xmax), alpha=.25, color=color, density=norm_hist)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylabel('Distribution', color=color, fontsize=14)
        
        plt.show()

    def _calc_ohe_pdp(self, feature_name):

        # get all OHE columns with feature_name in the column name
        feature_array = [col for col in self.transformed_features.columns.tolist() if feature_name in col]

        total_size = self.transformed_features.shape[0]

        a = []
        for feature in feature_array:
            # create copy of transformed_features as this will mutate the data
            _data = self.transformed_features.copy()

            # get population size before resetting it
            pop_size = np.sum(_data[feature])

            # get the other features related to the OHE columns
            other_features = [other_feature for other_feature in feature_array if other_feature != feature]
            # set the column feature to calculate to 1
            _data[feature] = 1

            # set the other features related to the OHE column to 0
            for other_feature in other_features:
                _data[other_feature] = 0

            # calculate prediction probabilities, only getting the positive probability
            preds = self.clf.predict_proba(_data.values)[:, 1]
            # calculate mean log odds for prediction and take away the mean log probability with the base data
            pdp_score = self._get_mean_log_odds(preds) - self.mean_log_pred_prob

            a.append({
                'instance': feature.replace(feature_name, '').replace('_', ''),
                'pop_size': pop_size / total_size,
                'pdp_score': pdp_score
                })
            #print('Feature %s: %.2f' % (feature, np.mean(preds)))
            #print('Feature %s: %.2f' % (feature, self._get_mean_log_odds(preds)))
        
        df = pd.DataFrame(a)
        df = df.sort_values('pop_size', ascending=False)

        fig, axes = plt.subplots(1, 2, sharey=True, figsize=(15, 5))
        fig.suptitle('PDP Values for {}'.format(feature_name))
        sns.set_color_codes("pastel")
        sns.barplot(x="pdp_score", y="instance", data=df,
                    label="PDP", color="b", ax=axes[0])
        sns.barplot(x="pop_size", y="instance", data=df,
                    label="Population Size", color="b", ax=axes[1])
        axes[1].set_ylabel('')
        plt.show()

        