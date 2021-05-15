from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from disaster_response_pipeline_project.data.tokenizer import  tokenize
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import os
import joblib

class SplitFeatures(BaseEstimator, TransformerMixin):
    """
    Splitt the features in different columns
    """
    def __init__(self,start, end = None):
        self.start = start
        self.end = end

    def fit(self, x, y=None):
        return self

    def transform(self, all_features):
        if self.end:
            return all_features[:, self.start:self.end].copy()
        else:
            return all_features[:, self.start]


def get_model_pipeline(model_name: str):
    """
    Get the model pipeline and the search parameters for the model
    :param model_name: str
        Name of the model to use
    :return: tuple
        scikit-Learn Pipeline of the model, parameters of the model

    """
    if model_name is "random_forest":
        pipline = Pipeline([("feature", FeatureUnion([
            ("nlp", Pipeline([
                ("split", SplitFeatures(start=0)),
                ("vect", CountVectorizer(tokenizer=tokenize)),
                ("tfidf", TfidfTransformer())
            ])),
            ("dummy", Pipeline([
                ("split", SplitFeatures(start=1, end=2)),
                ("encoder", OneHotEncoder(sparse=True))
            ]))
        ])),
                            ("clf", MultiOutputClassifier(RandomForestClassifier()))])
        parameters = {
            "feature__nlp__vect__ngram_range": [(1, 1), (1, 2)],
            "feature__nlp__vect__min_df": [3, 5, 10],
            "clf__estimator__n_estimators": [100],
            "clf__estimator__min_samples_split": [ 4, 6]
        }
        return  pipline, parameters
    elif model_name is "logistic_regression":
        pipline = Pipeline([
            ("split", SplitFeatures(start=0)),
            ("vect", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer()),
            ("clf", MultiOutputClassifier(LogisticRegression()))])

        parameters = {
            "vect__ngram_range": [(1, 1), (1, 2)],
            "vect__min_df": [3, 5, 10]
        }
        return pipline, parameters
    else:
        raise  ValueError("The model {} is not known, please create a model")

def build_model(model_name: str, n_jobs = 1, cv = 5):
    """
    Choose a model from the model factory and create a grid search object.
    :param model_name: str
        model name for the model in the model factory.
    :param n_jobs: int
        number of jobs for multi processing.
    :param cv: int
        number of folds for cross validation.
    :return:
        grid search model object.
    """

    pipeline, parameters = get_model_pipeline(model_name)
    gv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=n_jobs, cv=cv, scoring="f1_macro")
    return gv

def load_model(model_path: str, model_name: str):
    """
    Load the pickle file with a trained scikit-learn pipeline.
    :param model_path:
    :param model_name:
    :return:
    """
    complete_model_filename = os.path.join(model_path, model_name)
    if os.path.isfile(complete_model_filename):
        model = joblib.load(complete_model_filename)
        return model
    else:
        raise ValueError("The provided filepath is not correct.")