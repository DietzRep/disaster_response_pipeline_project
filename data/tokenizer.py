from nltk import word_tokenize
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

def tokenize(text):
    """
    Tokenzie and lemmatize text tokens.
    :param text: str
        Text string object.
    :return: list
        list with tokens.
    """
    assert type(text) is str
    lemmatizer = WordNetLemmatizer()
    eng_stopwords = stopwords.words("english")
    clean_tokens = [tok for tok in word_tokenize(text) if tok not in eng_stopwords]
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in clean_tokens]
    return clean_tokens