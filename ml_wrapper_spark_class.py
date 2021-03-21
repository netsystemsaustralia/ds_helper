import warnings
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, IntegerType

from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

class MLWrapper:

    def __init__(self, spark_dataframe, target_name, numeric_features, categorical_features, config={}):
        self.spark_dataframe = spark_dataframe
        self.target_name = target_name
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.pipeline_stages = []
        self.final_pipeline = None
        self.pipeline_model = None
        self.spark_dataframe_transformed = None
        self.train_df = None
        self.test_df = None
        self.train_df_transformed = None
        self.test_df_transformed = None
        self.clf = None
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

        
    def _get_string_indexers_pipeline(self, df=None):
        indexers = [StringIndexer().setInputCol(column).setOutputCol(column+"_index") for column in self.categorical_features]
        pipeline = Pipeline().setStages(indexers)
        if df is not None:
            df_r = pipeline.fit(df).transform(df)
            print('-- Applying String Indexer to Categorical Columns --')
            print(df_r.toPandas().head())
        else:
            df_r = None
        
        self.pipeline_stages.append(pipeline)
        
        return df_r

    def _get_one_hot_encoder_estimators(self, df=None):
        encoder = OneHotEncoderEstimator()\
            .setInputCols([column+"_index" for column in self.categorical_features])\
            .setOutputCols([column+"_encoded" for column in self.categorical_features])
        if df is not None:
            df_r = encoder.fit(df).transform(df)
            print('-- Applying One Hot Encoder to Indexed Categorical Columns --')
            print(df_r.toPandas().head())
        else:
            df_r = None        
        
        self.pipeline_stages.append(encoder)
        
        return df_r

    def _get_label_encoder(self, df=None):
        label_indexer = StringIndexer().setInputCol(self.target_name).setOutputCol('label')
        if df is not None:
            df_r = label_indexer.fit(df).transform(df)
            print('-- Applying String Indexer to Label --')
            print(df_r.toPandas().head())
        else:
            df_r = None     
        
        self.pipeline_stages.append(label_indexer)
        
        return df_r

    def _get_vector_assembly(self, df=None):
        assembler = VectorAssembler()\
            .setInputCols([column+"_encoded" for column in self.categorical_features] + self.numeric_features)\
            .setOutputCol('vectorised_features')
        
        if df is not None:
            df_r = assembler.transform(df)
            print('-- Applying Vector Assembly --')
            print(df_r.toPandas().head())
        else:
            df_r = None     
        
        self.pipeline_stages.append(assembler)
        
        return df_r

    def _get_standard_scaler(self, df=None):
        scaler = StandardScaler().setInputCol('vectorised_features').setOutputCol('features')

        if df is not None:
            df_r = scaler.fit(df).transform(df)
            print('-- Applying Standard Scaler to Vectorised features --')
            print(df_r.toPandas().head())
        else:
            df_r = None     
        
        self.pipeline_stages.append(scaler)
        
        return df_r

    def _create_end_to_end_pipeline(self, mode='test'):
        
        if mode == 'test':
            df = self.spark_dataframe
        else:
            df = None
        
        df_s = self._get_string_indexers_pipeline(df)
        df_e = self._get_one_hot_encoder_estimators(df_s)
        df_l = self._get_label_encoder(df_e)
        df_v = self._get_vector_assembly(df_l)
        df_f = self._get_standard_scaler(df_v)

        if mode == 'test':
            print(df_f.toPandas().head())
        
        # create final pipeline
        self.final_pipeline = Pipeline().setStages(self.pipeline_stages)
            
    def _split_and_fit(self):
        self.train_df, self.test_df = self.spark_dataframe.randomSplit([0.8, 0.2], seed=42)
        print("Training Dataset Count: " + str(self.train_df.count()))
        print("Test Dataset Count: " + str(self.test_df.count()))
        
        # fit on training set
        self.pipeline_model = self.final_pipeline.fit(self.train_df)

        # transform train and test
        self.train_df_transformed = self.pipeline_model.transform(self.train_df)
        self.test_df_transformed = self.pipeline_model.transform(self.test_df)

    def _run_classification(self):
        lr = LogisticRegression(featuresCol='features', labelCol='label', maxIter=5)
        self.clf = lr.fit(self.train_df_transformed)

        predictions = self.clf.transform(self.test_df_transformed)
        print(predictions.select('label', 'rawPrediction', 'prediction', 'probability').toPandas().head())
        
        evaluator = BinaryClassificationEvaluator()
        print('Test AUC ROC', evaluator.evaluate(predictions))

    def _run_hyperparameter_tuning(self):
        lr = LogisticRegression(featuresCol='features', labelCol='label', maxIter=5)
        evaluator = BinaryClassificationEvaluator()
        paramGrid = (ParamGridBuilder()
                .addGrid(lr.regParam, [0.01, 0.5, 2.0])
                .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
                .addGrid(lr.maxIter, [1, 5, 10])
                .build())

        cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)

        cvModel = cv.fit(self.train_df_transformed)
        
        predictions = cvModel.transform(self.test_df_transformed)
        print('Best Model AUC', evaluator.evaluate(predictions))

        best_model = cvModel.bestModel

        print('Best Params {} {} {}'.format(best_model._java_obj.getRegParam(), best_model._java_obj.getMaxIter(), best_model._java_obj.getElasticNetParam()))
