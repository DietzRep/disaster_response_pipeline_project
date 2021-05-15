import nltk
from nltk import word_tokenize
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', "stopwords"])

def tokenize(text):
    """
    Tokenize and lemmatize a text into token. ALso Removes english stopwords.
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