import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_curve, auc, precision_score, 
    recall_score, f1_score, precision_recall_curve
)

class MLTest:

    def __init__(self, train_X, train_y, validate_X, validate_y, model, transformer=None):
        self.train_X = train_X
        self.train_y = train_y
        self.validate_X = validate_X
        self.validate_y = validate_y
        self.model = model
        self.transformer = transformer

        if transformer is not None:
            self.train_X = self.transformer.transform(self.train_X)
            self.validate_X = self.transformer.transform(self.validate_X)
            self.transformed_names = self._get_names()

        self.train_pred, self.train_pred_prob, self.train_pred_pos_prob = self._create_predictions(self.train_X)
        self.validate_pred, self.validate_pred_prob, self.validate_pred_pos_prob = self._create_predictions(self.validate_X)

        self._set_best_threshold()

    def _get_names(self):
        """
        Return verbose names for the transformed columns.
        """
        all_names = []
        for t in self.transformer.transformers_:
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
    
    def _set_best_threshold(self, method='f1opt'):

        if method == 'f1opt':
            precision, recall, thresholds = precision_recall_curve(self.train_y, self.train_pred_pos_prob)
            precision = precision[:-1]
            recall = recall[:-1]
            f1 = 2 * ((precision * recall) / (precision + recall))
            ix = np.argmax(f1)
            self.optimal_threshold = thresholds[ix]
        else:
            pass
    
    def _create_predictions(self, features):  
        y_pred = self.model.predict(features)
        y_pred_prob = self.model.predict_proba(features)
        y_pred_pos_prob = y_pred_prob[:,1]
        return y_pred, y_pred_prob, y_pred_pos_prob
    
    def _get_auc(self, y, pred_prob):
        false_positive_rate, true_positive_rate, _ = roc_curve(y, pred_prob)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        return roc_auc
    
    def _get_metrics(self, y_true, y_pred_prob):
        roc_auc = np.round(self._get_auc(y_true, y_pred_prob), 4)
        pred = y_pred_prob >= self.optimal_threshold
        precision = np.round(precision_score(y_true, pred), 4)
        recall = np.round(recall_score(y_true, pred), 4)
        f1 = np.round(f1_score(y_true, pred), 4)
        return {'auc': roc_auc, 'precision': precision, 'recall': recall, 'f1': f1}

    def test_for_overfitting(self):

        # get training metrics
        train_metrics = self._get_metrics(self.train_y, self.train_pred_pos_prob)
        validate_metrics = self._get_metrics(self.validate_y, self.validate_pred_pos_prob)

        # for now just print them
        print('Training metrics: {}'.format(train_metrics))
        print('Validation metrics: {}'.format(validate_metrics))

    def test_for_feature_importance_overweight(self):
        importance_df = pd.DataFrame({'feature': self.transformed_names, 'importance': self.model.feature_importances_})
        top_3_sum = importance_df.sort_values('importance', ascending=False).iloc[0:3, 1].sum()
        if top_3_sum >= 0.80:
            print(importance_df.sort_values('importance', ascending=False).iloc[0:3,:])
            print('Test Failed: {}'.format(top_3_sum))
        else:
            print('Test Passed: {}'.format(top_3_sum))

    def test_for_class_imbalance(self):
        # just check to see what the ratio is between classes (assume binary for now)
        ratio = np.sum(self.train_y == 0) / np.sum(self.train_y == 1)
        print('Class Ratio: {}'.format(ratio))

        # now check the metrics for each class; again for now just assume binary
        # using the validation metric?

        pred = self.train_pred_pos_prob >= self.optimal_threshold
        l = (self.train_y == 0).tolist()
        yp = pred[l]
        yt = self.train_y[l]
        neg_metric = np.sum(yp == yt) / len(yt)

        l = (self.train_y == 1).tolist()
        yp = pred[l]
        yt = self.train_y[l]
        pos_metric = np.sum(yp == yt) / len(yt)

        print('Class Negative Accuracy: {}'.format(neg_metric))
        print('Class Positive Accuracy: {}'.format(pos_metric))