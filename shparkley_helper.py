import warnings

import numpy as np
import pandas as pd

from affirm.model_interpretation.shparkley.spark_shapley import (
    compute_shapley_for_sample,
    ShparkleyModel
)

class MyShparkleyModel(ShparkleyModel):
    """
    You need to wrap your model with the ShparkleyModel interface.
    """
    def get_required_features(self):
        # type: () -> Set[str]
        """
        Needs to return a set of feature names for the model.
        """
        return ['feature-1', 'feature-2', 'feature-3']

    def predict(self, feature_matrix):
        # type: (List[Dict[str, Any]]) -> List[float]
        """
        Wrapper function to convert the feature matrix into an acceptable format for your model.
        This function should return the predicted probabilities.
        The feature_matrix is a list of feature dictionaries.
        Each dictionary has a mapping from the feature name to the value.
        :return: Model predictions for all feature vectors
        """
        # Convert the feature matrix into an appropriate form for your model object.
        pd_df = pd.DataFrame.from_dict(feature_matrix)
        preds = self._model.my_predict(pd_df)
        return preds

class ShparkleyHelper:

    def __init__(self, fitted_model):
        self.fitted_model = fitted_model
        self.shparkley_wrapped_model = MyShparkleyModel(self.fitted_model)
