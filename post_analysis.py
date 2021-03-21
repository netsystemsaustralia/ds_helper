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

def show_values_on_bars(axs, h_v="v", space=0.4):
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

def plot_prob_dens(y_true, y_pred_prob, _feature, feature):
    x0 = y_pred_prob[(y_true == 0).tolist()][:, np.newaxis]
    x1 = y_pred_prob[(y_true == 1).tolist()][:, np.newaxis]
    plt.figure(figsize=[15,10])
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
    
def show_model_performance(y_true, y_pred_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
    precision = precision[:-1]
    recall = recall[:-1]
    f1 = 2 * ((precision * recall) / (precision + recall))
    ix = np.argmax(f1)
    print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], f1[ix]))
    plt.figure(figsize=(15,5))
    d = pd.DataFrame({'Threshold': thresholds, 
                      'Recall': recall, 
                      'Precision': precision,
                     'F1': f1})
    sns.lineplot(x='Threshold', y='value', hue='variable', data=pd.melt(d, ['Threshold']))
    plt.scatter(thresholds[ix], f1[ix], marker='o', color='black', label='Best')
    plt.show()

def get_auc(y, pred_prob):
    false_positive_rate, true_positive_rate, _ = roc_curve(y, pred_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    return roc_auc

def get_metrics(y_true, y_pred_prob, threshold):
    roc_auc = np.round(get_auc(y_true, y_pred_prob), 4)
    pred = y_pred_prob >= threshold
    precision = np.round(precision_score(y_true, pred), 4)
    recall = np.round(recall_score(y_true, pred), 4)
    f1 = np.round(f1_score(y_true, pred), 4)
    return (roc_auc, precision, recall, f1)
    
def get_confusion_matrix(y_true, y_pred_prob, threshold):
    pred = y_pred_prob >= threshold
    display(pd.DataFrame(confusion_matrix(y_true, pred), 
             columns=['Predicted Not Churn', 'Predicted to Churn'],
             index=['Actual Not Churn', 'Actual Churn']))
    
def get_subpopulation_report_in_widget(y_true, y_pred_prob, threshold, X, _feature):
    features = X[_feature].unique()
    # create outputs for each numerical_columns
    outputs = [widgets.Output() for feature in features]
    # fix layout
    item_layout = widgets.Layout(margin='0 0 50px 0')
    # create tab widget
    tab = widgets.Tab(outputs, layout=item_layout)
    index = 0

    for feature in features:
        l = (X[_feature] == feature).tolist()
        ypp = y_pred_prob[l]
        yt = y_true[l]
        with outputs[index]:
            print(f'Feature Instance: {feature}')
            print(get_metrics(yt, ypp, threshold))
            plot_prob_dens(yt, ypp, _feature, feature)
            get_confusion_matrix(yt, ypp, threshold)
        
        tab.set_title(index, f'Report for {feature}')
        index = index + 1

    display(tab)

def get_subpopulation_report(y_true, y_pred_prob, threshold, X, _feature):
    features = X[_feature].unique()
    for feature in features:
        l = (X[_feature] == feature).tolist()
        ypp = y_pred_prob[l]
        yt = y_true[l]
        print(f'Feature Instance: {feature}')
        print(get_metrics(yt, ypp, threshold))
        plot_prob_dens(yt, ypp, _feature, feature)
        get_confusion_matrix(yt, ypp, threshold)
        
def plot_pdp(model, X, feature, mean_log_odds, transformer=None, return_pd=False, y_pct=True, norm_hist=True, dec=.5):
    # get feature_index
    feature_index = X.columns.tolist().index(feature)
    
    # Get partial dependence
    pardep = partial_dependence(model, X.values, [feature_index])

    # x-values
    pardep_x = pardep[1][0]

    # y-values
    pardep_y = pardep[0][0]

    # translate X-axis values back to original values if have transformer
    if transformer != None:
        pardep_x = transformer.inverse_transform(pardep_x)
        x = transformer.inverse_transform(X[feature])
    else:
        x = X[feature]
    
    
    # transform y-axis values to difference of log-odds to average
    pardep_y = np.log(pardep_y / (1 - pardep_y))
    pardep_y = pardep_y - mean_log_odds
    
    # Get min & max values
    xmin = pardep_x.min()
    xmax = pardep_x.max()
    ymin = pardep_y.min()
    ymax = pardep_y.max()

        
    # Create figure
    fig, ax1 = plt.subplots(figsize=(15,10))
    ax1.grid(alpha=.5, linewidth=1)
    
    # Plot partial dependence
    color = 'tab:blue'
    ax1.plot(pardep_x, pardep_y, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xlabel(feature, fontsize=14)
    
    ax1.set_ylabel('Partial Dependence', color=color, fontsize=14)
    
    ax1.set_title('Relationship Between {} and Target Variable'.format(feature), fontsize=16)
    
    if y_pct and ymin>=0 and ymax<=1:
        # Display yticks on ax1 as percentages
        fig.canvas.draw()
        labels = [item.get_text() for item in ax1.get_yticklabels()]
        labels = [int(np.float(label)*100) for label in labels]
        labels = ['{}%'.format(label) for label in labels]
        ax1.set_yticklabels(labels)
    
    # Plot line for decision boundary
    ax1.hlines(dec, xmin=xmin, xmax=xmax, color='black', linewidth=2, linestyle='--', label='Decision Boundary')
    ax1.legend()

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.hist(x, bins=50, range=(xmin, xmax), alpha=.25, color=color, density=norm_hist)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylabel('Distribution', color=color, fontsize=14)
    
    if y_pct and norm_hist:
        # Display yticks on ax2 as percentages
        fig.canvas.draw()
        labels = [item.get_text() for item in ax2.get_yticklabels()]
        labels = [int(np.float(label)*100) for label in labels]
        labels = ['{}%'.format(label) for label in labels]
        ax2.set_yticklabels(labels)

    plt.show()
    
    if return_pd:
        return pardep
    
def get_original_values(X_test_df, preprocessor):
    """
    Take the preprocessor from the Pipeline and 'recover' original values
    """
    new_df = X_test_df.copy()
    for t in preprocessor.transformers_:
        if hasattr(t[1],'named_steps'):
            for k, v in t[1].named_steps.items():
                field_name = t[0]
                if k == 'scaler':
                    # easy, just get inverse
                    new_df.loc[:,f'{field_name}_orig'] = v.inverse_transform(X_test_df[field_name])
                elif k == 'onehot':
                    # slight harder, get all derived columns then get inverse
                    derived_fields = v.get_feature_names(input_features=[field_name])
                    new_df.loc[:,f'{field_name}_orig'] = v.inverse_transform(X_test_df[derived_fields])
     
    return new_df
    
def get_post_processed_dataframe(preprocessor, X_test):
    """
    Take the preprocessor from the Pipeline and create a post_processed_dataframe on the test features
    """
    X_test_t = preprocessor.transform(X_test)
    all_names = get_names(preprocessor)
    X_test_df = pd.DataFrame(data = X_test_t, columns = all_names)
    return X_test_df

def get_names(preprocessor):
    """
    Return verbose names for the transformed columns.
    """
    all_names = []
    for t in preprocessor.transformers_:
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

def get_lift_per_bin(y_true, y_pred_prob):
    # calculate lift per bin
    lift_df = pd.DataFrame({'pred': y_true, 'prob': y_pred_prob})
    lift_df.loc[:,'prob_bin'] = pd.qcut(lift_df.loc[:,'prob'], q=10, labels=False)
    # reverse bin numbers
    lift_df.loc[:,'prob_bin'] = (10 - lift_df.loc[:,'prob_bin']).astype(pd.Int32Dtype())
    avg_perc = np.mean(y_true)
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

def get_feature_importance_permutation(model, X, y):
    result = permutation_importance(model, X.values, y, n_repeats=10,
                                random_state=42, n_jobs=2)
    importance_df = pd.DataFrame({'feature': X.columns, 'importance': result.importances_mean})
    plt.figure(figsize=(15, 10))
    sns.set_color_codes("pastel")
    sns.barplot(x="importance", y="feature", 
            data=importance_df[importance_df['importance'] != 0.0].sort_values('importance', ascending=False),
            label="Importance", color="b")
    plt.show()

def get_feature_importance_from_model(model, X):
    # feature importance
    importance_df = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
    # summarize feature importance
    plt.figure(figsize=(15,10))
    sns.set_color_codes("pastel")
    sns.barplot(x="importance", y="feature", data=importance_df[importance_df['importance'] > 0.0].sort_values('importance', ascending=False),
                label="Importance", color="b")
    plt.show()

def get_feature_importance_use_shapley(model, X):
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X)

def get_feature_importance(model, X, y):
    outputs = [widgets.Output() for i in range(1,4)]
    # fix layout
    item_layout = widgets.Layout(margin='0 0 50px 0')
    # create tab widget
    tab = widgets.Tab(outputs, layout=item_layout)
    tab.set_title(0, 'From Model')
    tab.set_title(1, 'From Permutation')
    tab.set_title(2, 'From SHAP')

    with outputs[0]:
        get_feature_importance_from_model(model, X)
    
    with outputs[1]:
        get_feature_importance_permutation(model, X, y)

    with outputs[2]:
        get_feature_importance_use_shapley(model, X)

    display(tab)