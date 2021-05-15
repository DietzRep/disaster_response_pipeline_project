import spacy
import pandas as pd
from data.column_utils import MESSAGES_COLUMNS, MESSAGE_COLUMN, ENTITY_COLUMN, ENTITY_LABEL_COLUMN, IDENTIFIER_COLUMN


class EntityExtractor:

    def __init__(self, embedding_name: str = "en_core_web_lg"):
        self.nlp  = spacy.load(embedding_name, disable=["tok2vec"])

    def extract_entities(self, disaster_df: pd.DataFrame) -> pd.DataFrame:
        """
        extract all entities from the disaster messages.
        :param disaster_df: pd.DataFrame
            Data frame with texts to process
        :return: pd.DataFrame
            Pandas DataFrame with entities of each message.
        """
        assert isinstance(disaster_df, pd.DataFrame)
        assert all(required_column in disaster_df.columns for required_column in MESSAGES_COLUMNS)
        labels = []
        texts = []
        ids = []
        for idx, row in disaster_df.iterrows():
            for ent in self.get_doc_entities(row[MESSAGE_COLUMN]):
                labels.append(ent.label_)
                texts.append(ent.text)
                ids.append(row[IDENTIFIER_COLUMN])
        return pd.DataFrame({IDENTIFIER_COLUMN: ids, ENTITY_LABEL_COLUMN: labels, ENTITY_COLUMN: texts})


    def get_doc_entities(self, doc: str):
        return self.nlp(doc).ents