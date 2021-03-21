import warnings
from functools import wraps
from time import time
from datetime import datetime

from joblib import dump, load
import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

import seaborn as sns

from sklearn.metrics import (
    roc_curve, auc, precision_score, recall_score, 
    f1_score, precision_recall_curve, confusion_matrix
)
from sklearn.inspection import plot_partial_dependence, partial_dependence, permutation_importance
from sklearn.pipeline import _name_estimators, Pipeline

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap

def logging(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = datetime.now()
        ts_start = '{}/{}/{}/{}_{}_{}'.format(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second)
        print('%r Function %r started' % (ts_start, f.__name__))
        result = f(*args, **kw)
        ts = datetime.now()
        ts_end = '{}/{}/{}/{}_{}_{}'.format(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second)
        print('%r Function %r ended' % (ts_end, f.__name__))
        return result
    return wrap

class MLWrapper:

    def __init__(self, X, y, numeric_features, categorical_features, config={}):
        self.X = X
        self.y = y
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        if not config:
            self.config = {
                'random_state': 42,
                'test_size': 0.2,
                'imputer_numeric_strategy': 'median',
                'scale_numeric': True,
                'imputer_categorical_strategy': 'constant:missing',
                'model': 'xgboost',
                'model_params': {
                    'learning_rate': 0.2, 
                    'n_estimators': 19,
                    'subsample': 0.8,
                    'max_depth': 3
                }
            }
        else:
            self.config = config

        self._initalise()

    def _initalise(self):
        self._get_numeric_label()
        self._create_transformers()
        self._create_preprocessor()
        self._create_pipeline()
        self._split_train_test()

    def _get_numeric_label(self):
        le = LabelEncoder()
        self.y_encoded = le.fit_transform(self.y)
    
    def _split_train_test(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y_encoded, 
                                                    test_size=self.config['test_size'], 
                                                    random_state=self.config['random_state'], stratify=self.y_encoded)
    
    def _get_numeric_transformer(self):
        #TBC make this configurable
        return Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.config['imputer_numeric_strategy'])),
            ('scaler', StandardScaler())])

    def _get_categorical_transformer(self):
        # TBD make this configurable
        return Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    def _create_transformers(self):
        transformers = []

        for feature in self.numeric_features:
            transformers.append((feature, self._get_numeric_transformer(), [feature]))

        for feature in self.categorical_features:
            transformers.append((feature, self._get_categorical_transformer(), [feature]))

        self.transformers = transformers

    def _create_preprocessor(self):
        self.preprocessor = ColumnTransformer(transformers=self.transformers)

    def _get_model(self):
        if self.config['model'] == 'xgboost':
            return XGBClassifier(**self.config['model_params'], random_state=self.config['random_state'])

    def _create_pipeline(self):
        self.clf = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', self._get_model())
        ])

    
    def save_experiment(self):
        ts = datetime.now()
        suffix = '{}_{}_{}_{}_{}_{}'.format(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second)
        # save preprocessor pipeline separate to model
        preprocessor = self.clf['preprocessor']
        dump(preprocessor, 'preprocessor_{}.joblib'.format(suffix))
        clf = self.clf['classifier']
        dump(clf, 'clf_{}.joblib'.format(suffix))

        # save metadata on experiment
        # this will eventually include performance metrics and possibly sanity checks, like overfitting, and model ethics too
        metadata = {
            'ts': suffix,
            'score': self.clf.score(self.X_test, self.y_test)
        }
        with open('metadata_{}.json'.format(suffix), 'w') as outfile:
            json.dump(metadata, outfile)

    def hyperopt_fit(self):

        param_grid = {
            'classifier__n_estimators': [5, 10, 15, 20],
            'classifier__max_depth': [2, 4, 6],
            'classifier__subsample': [0.6, 0.8, 1.0],
            #'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__min_child_weight': [1, 5, 10],
            'classifier__gamma': [0.5, 1, 1.5, 2, 5],
            'classifier__colsample_bytree': [0.6, 0.8, 1.0]
        }

        search = GridSearchCV(self.clf, param_grid, n_jobs=2, cv=2, verbose=2)
        search.fit(self.X_train, self.y_train)
        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        print(search.best_params_)
        return search

    def plot_hyperopt_results_alt(self, search):
        
        # get out parameters
        results = pd.DataFrame(search.cv_results_['params'])
        results.loc[:, 'score'] = search.cv_results_['mean_test_score']
        # fix up column names; remove classifier
        col_names = [col.replace('classifier__', '') for col in results.columns]
        results.columns = col_names

        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

        for col in results.columns:
            if col != 'score':
                df = pd.DataFrame({col: results['score'].values, 'g': results[col].values})

                
                # Initialize the FacetGrid object
                pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
                g = sns.FacetGrid(df, row="g", hue="g", aspect=15, height=.8, palette=pal)

                # Draw the densities in a few steps
                g.map(sns.kdeplot, col,
                    bw_adjust=.5, clip_on=False,
                    fill=True, alpha=1, linewidth=1.5)
                g.map(sns.kdeplot, col, clip_on=False, color="w", lw=2, bw_adjust=.5)
                g.map(plt.axhline, y=0, lw=2, clip_on=False)


                # Define and use a simple function to label the plot in axes coordinates
                def label(x, color, label):
                    ax = plt.gca()
                    ax.text(0, .2, label, fontweight="bold", color=color,
                            ha="left", va="center", transform=ax.transAxes)


                g.map(label, col)

                # Set the subplots to overlap
                g.fig.subplots_adjust(hspace=-.25)

                # Remove axes details that don't play well with overlap
                g.set_titles("")
                g.set(yticks=[])
                g.despine(bottom=True, left=True)
    
    def plot_hyperopt_results(self, search, highlight_top_x=5, color_type='gradient'):
        # get out parameters
        results = pd.DataFrame(search.cv_results_['params'])
        results.loc[:, 'score'] = search.cv_results_['mean_test_score']
        # fix up column names; remove classifier
        col_names = [col.replace('classifier__', '') for col in results.columns]
        results.columns = col_names

        # sort dataframe by score
        results_sorted = results.copy().sort_values(by=['score'], ascending=True)

        # setup data and plot framework
        
        normalised_scores = ((results_sorted.loc[:, 'score'] - np.min(results_sorted.loc[:, 'score'])) / (np.max(results_sorted.loc[:, 'score']) - np.min(results_sorted.loc[:, 'score']))).values
        top_x_min = np.sort(normalised_scores)[-highlight_top_x:][0]
        
        _, host = plt.subplots(figsize=(15,10))

        # organize the data
        #ys = np.dstack([md, ne, lr, ss, sc])[0]
        ys = results_sorted.values
        ymins = ys.min(axis=0)
        ymaxs = ys.max(axis=0)
        dys = ymaxs - ymins
        ymins -= dys * 0.05  # add 5% padding below and above
        ymaxs += dys * 0.05
        dys = ymaxs - ymins

        # transform all data to be compatible with the main axis
        zs = np.zeros_like(ys)
        zs[:, 0] = ys[:, 0]
        zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]


        axes = [host] + [host.twinx() for i in range(ys.shape[1] - 1)]
        n = 0
        for i, ax in enumerate(axes):
            ax.grid(False)
            ax.set_ylim(ymins[i], ymaxs[i])
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_color('grey')
            ax.set_facecolor('white')
            if ax != host:
                #ax.spines['left'].set_visible(False)
                ax.spines['right'].set_color('grey')
                ax.yaxis.set_ticks_position('right')
                ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))

        host.set_xlim(0, ys.shape[1] - 1)
        host.set_xticks(range(ys.shape[1]))
        host.set_xticklabels(results_sorted.columns, fontsize=14)
        host.tick_params(axis='x', which='major', pad=7)
        host.spines['right'].set_visible(False)
        host.xaxis.tick_top()
        host.set_title('Hyperopt Parameter Results', fontsize=18)
        #print(ys.shape)
        for j in range(ys.shape[0]):
            if normalised_scores[j] >= top_x_min:
                color = plt.cm.PRGn(n / highlight_top_x) if color_type == 'gradient' else plt.cm.Paired(n)
                #print(color)
                n = n + 1
            else:
                color = 'whitesmoke'
            
            # to just draw straight lines between the axes:
            # host.plot(range(ys.shape[1]), zs[j,:], c=colors[(category[j] - 1) % len(colors) ])
            # create bezier curves
            # for each axis, there will a control vertex at the point itself, one at 1/3rd towards the previous and one
            #   at one third towards the next axis; the first and last axis have one less control vertex
            # x-coordinate of the control vertices: at each integer (for the axes) and two inbetween
            # y-coordinate: repeat every point three times, except the first and last only twice
            verts = list(zip([x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
                            np.repeat(zs[j, :], 3)[1:-1]))
            # for x,y in verts: host.plot(x, y, 'go') # to show the control points of the beziers
            codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
            path = Path(verts, codes)
            patch = patches.PathPatch(path, 
                                    facecolor='none', 
                                    lw=4 if normalised_scores[j] >= top_x_min else 1, 
                                    edgecolor=color)
            host.add_patch(patch)
        plt.tight_layout()
        plt.show()
    
    
    @logging
    def fit(self):
        self.clf.fit(self.X_train, self.y_train)
        print("model score: %.3f" % self.clf.score(self.X_test, self.y_test))
        self.save_experiment()
