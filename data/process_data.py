import os
import pandas as pd
import numpy as np
import argparse
import logging
from data.entity_extractor import EntityExtractor
from sqlalchemy import create_engine
from data.column_utils import CATEGORIES_COLUMNS, MESSAGES_COLUMNS, IDENTIFIER_COLUMN, CATEGORIES_RAW
logging.basicConfig(level=logging.INFO)


def load_data(messages_filepath: str, categories_filepath :str) -> pd.DataFrame:
    """
    Load the pandas data frames with messages and categories and join them into one pandas DataFrame.
    :param messages_filepath: str
        file path to messages.csv file. With columns ["id", "message", "original", "genre"]
    :param categories_filepath: str
        file path to categories.csv file. With columns ["id", "categories"].
    :return: pandas.DataFrame
    """
    assert type(messages_filepath) is str
    assert type(categories_filepath) is str
    # join path names and file names.
    messages_filename = os.path.join(messages_filepath, "disaster_messages.csv")
    categories_filename = os.path.join(categories_filepath, "disaster_categories.csv")
    # check if both file paths are files actual
    if os.path.isfile(messages_filename) and os.path.isfile(categories_filename):
        categories = pd.read_csv(categories_filename)
        messages = pd.read_csv(messages_filename)
        # check if all necessary columns are available
        if all([x in categories.columns for x in CATEGORIES_COLUMNS]) and all([x in messages.columns for x in  MESSAGES_COLUMNS]):
            df = pd.merge(categories, messages, on = IDENTIFIER_COLUMN)
            return df
        else:
            raise ValueError("Not all expected columns available.")
    else:
        raise ValueError("File paths are not correct. Please provide file path to messages.csv and categories.csv")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data set and create labels for the data set.
    :param df: pandas DataFrame
        Loaded DataFrame with the columns ["id", "messages", "original", "genre", "categories"]
    :return: pandas DataFrame
        cleaned pandas DataFrame with binary labels.
    """
    assert isinstance(df, pd.DataFrame)
    # check if data frame includes all columns.
    if all([x in df.columns for x in set(CATEGORIES_COLUMNS + MESSAGES_COLUMNS)]):
        # extract different categories.
        categories_data = df[CATEGORIES_RAW].str.split(";", expand = True)
        # assign category names.
        categories_data.columns  = categories_data.iloc[0].apply(lambda x: x[:-2])
        for col in categories_data.columns:
            # get one hot encoding for each column and cast column to integer.
            categories_data[col] = categories_data[col].apply(lambda x: x[-1]).astype(np.int8).replace(2,1)
        # drop categories and add new categories data frame
        df = df.drop([CATEGORIES_RAW], axis = 1)
        df = pd.concat([df, categories_data], axis = 1)
        df = df.drop_duplicates()
        assert sum(df.duplicated()) == 0

        # extract original language

        # extract entities

        return df
    else:
        raise ValueError("DataFrame includes not all expected columns.")

def save_data_to_db(df: pd.DataFrame, entities_dataframe: pd.DataFrame,  database_filename : str):
    """
    Save the preprocessed DataFrame in a sqlite db.
    :param df: pandas DataFrame
        preprocessed pandas data frame
    :param df: entity DataFrame
        data frame with entities
    :param database_filename: str
        filename of the final database
    """
    assert type(database_filename) is str
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql("disaster_response", engine, index = False, if_exists = "replace")
    entities_dataframe.to_sql("text_entities", engine, index=False, if_exists="replace")

def main(args):
    """
    Run the preprocessing pipeline for the provided csv. First merge messages and categories. Then extract entities of the messages
    and the end save the data to the data base.

    :param args
        parameter for the function. Have to include file path to messages, categories and the path to save the data base to.
    """
    messages_filepath = args.messages_filepath
    categories_filepath = args.categories_filepath
    database_filepath = args.database_filepath

    logging.info('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
          .format(messages_filepath, categories_filepath))
    df = load_data(messages_filepath, categories_filepath)
    logging.info('Cleaning data frame of shape {}'.format(df.shape))
    df = clean_data(df)

    logging.info('Extract entities of {} messages'.format(df.shape[0]))
    entity_extractor = EntityExtractor()
    entity_data = entity_extractor.extract_entities(df)
    logging.info('{} entities extracted.'.format(entity_data.shape[0]))

    logging.info('Saving data...\n    DATABASE: {}'.format(database_filepath))
    save_data_to_db(df, entity_data, database_filepath)

    logging.info('Cleaned data saved to database!')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ETL for disaster response pipeline.')
    parser.add_argument("-mp", "--messages_filepath", required=True, type= str, help="filepath for messages.csv." )
    parser.add_argument("-cp", "--categories_filepath", required=True, type = str, help = "filepath for categories.csv with labels for each message.")
    parser.add_argument("-dp", "--database_filepath", required=True, type=str, help = "filename of sqlite data base.")
    args = parser.parse_args()

    main(args)