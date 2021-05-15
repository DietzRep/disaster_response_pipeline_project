from sqlalchemy import create_engine
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from sklearn.metrics import classification_report
from models.model_factory import  build_model
import joblib
import argparse
import logging
from data.column_utils import TRAININGS_COLUMNS, MESSAGES_COLUMNS, IGNORE_LABELS
logging.basicConfig(level=logging.INFO)


def load_data(database_filepath):
    assert type(database_filepath) is str
    if os.path.isfile(database_filepath):
        engine = create_engine("sqlite:///" + database_filepath)
        data_frame = pd.read_sql("Select * from disaster_response", engine)
        data_frame_new = data_frame.copy()

        X = data_frame_new[TRAININGS_COLUMNS].values
        Y = data_frame_new.drop(MESSAGES_COLUMNS + IGNORE_LABELS, axis = 1)
        column_names = Y.columns
        return X, Y.values, column_names
    else:
        raise ValueError("{} is no valid data base.")



def evaluate_model(model, X_test, Y_test, category_names, save_eval , report_file_path ):
    """
    Evaluate the performance of a pretrained classifier.
    :param model: object
        pretrained scikit learn pipeline
    :param X_test: np.array
        data of test observations.
    :param Y_test: np.array
        True labels of test observations.
    :param category_names: list
        names of the categories.
    :param save_eval: bool
        If true save evaluation results and predictions.
    :param report_file_path: string
        Path to save evaluation results
    """
    #assert Y_test.shape[0] == X_test.shape[0]
    #assert Y_test.shape[1] == len(category_names)
    prediction_results = model.predict(X_test)
    prediction_results = prediction_results.round()
    results = []
    for col in range(len(category_names)):
        result_report = classification_report(y_pred=prediction_results[:,col], y_true=Y_test[:,col])
        print("Evaluation column:", category_names[col])
        print(result_report)

        if save_eval:
            result_report = classification_report(y_pred=prediction_results[:,col], y_true=Y_test[:,col], output_dict=True)
            results.append(result_report["1"])

    if save_eval:
        results_dataframe = pd.DataFrame(results)
        results_dataframe.to_csv(os.path.join(report_file_path , "eval.csv") )

        with open(os.path.join(report_file_path ,"eval_X.pickle"), "wb") as output_file:
            pickle.dump(X_test, output_file)

        with open(os.path.join(report_file_path, "eval_Y_true.pickle"), "wb") as output_file:
            pickle.dump(Y_test, output_file)

        with open(os.path.join(report_file_path, "eval_Y_predict.pickle"), "wb") as output_file:
            pickle.dump(prediction_results, output_file)


def save_model(model, model_filepath: str, model_name: str):
    """
    Save a sklearn model in a pickle file.
    :param model: object
        trained sklearn pipeline.
    :param model_filepath: str
        path to save the model in.
    """
    assert type(model_filepath) is str
    assert type(model_name) is str
    if os.path.isdir(model_filepath):
        joblib.dump(model, os.path.join(model_filepath, "{}.pkl".format(model_name)))
    else:
        raise ValueError("{} is no valid path. Please provide a valid path name.".format(model_filepath))

def main(args):
    database_filepath= args.database_filepath
    model_filepath = args.model_filepath
    model_name = args.model_name
    save_results = args.save_results == "yes"


    logging.info('Loading data...\n    DATABASE: {}'.format(database_filepath))
    X, Y, category_names = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    logging.info('Build model: {}'.format(model_name))
    model = build_model(model_name)

    logging.info('Training model on {} training samples'.format(X_train.shape))
    model.fit(X_train, Y_train)

    logging.info('Evaluating model...')

    evaluate_model(model, X_test, Y_test, category_names, save_results, model_filepath)

    logging.info('Saving model...\n    MODEL: {}'.format(model_name))
    save_model(model, model_filepath, model_name)

    logging.info('Trained model saved!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model training for disaster response pipeline.')
    parser.add_argument("-mp", "--model_filepath", required=True, type= str, help="filepath to save the model" )
    parser.add_argument("-dp", "--database_filepath", required=True, type=str, help = "filepath to the data base.")
    parser.add_argument("-sr", "--save_results", required=True, type=str, help="save results of the model evaluation", choices=["yes", "no"])
    parser.add_argument("--model_name", type= str, help="Name of the model to train", default="random_forest")
    args = parser.parse_args()
    main(args)