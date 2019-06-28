from __future__ import absolute_import
import os
import sys
from nltk import TreebankWordTokenizer


from textblob_dz.compat import unicode

try:
    MODULE = os.path.dirname(os.path.abspath(__file__))
except:
    MODULE = ""

sys.path.insert(0, os.path.join(MODULE, "..", "..", "..", ".."))

# Import parser base classes.
from textblob_dz.text import (

    PUNCTUATION
)
# Import parser universal tagset.
from textblob_dz.text import (

    UNIVERSAL,
)


# Import sentiment analysis base classes.
from textblob_dz.text import (
    Sentiment as _Sentiment,

    MOOD, IRONY
)

sys.path.pop(0)

# Constants from pattern.text.tree
SLASH, WORD, POS, CHUNK, PNP, REL, ANCHOR, LEMMA = \
    "&slash;", "word", "part-of-speech", "chunk", "preposition", "relation", "anchor", "lemma"
MBSP = False



class Parser():

    def find_tokens(self, tokens, **kwargs):
        print("tokens "+ tokens)
        return TreebankWordTokenizer().tokenize(tokens)


class Sentiment(_Sentiment):

    def load(self, path=None):
        _Sentiment.load(self, path)
        # Map "précaire" to "precaire" (without diacritics, +1% accuracy).
        if not path:
            for w, pos in list(self.items()):
                w0 = w
                if w != w0:
                    for pos, (p, s, i) in pos.items():
                        self.annotate(w, pos, p, s, i)

parser = Parser()

sentiment = Sentiment(
        path = os.path.join(MODULE, "dz-sentiment.xml"),
        synset = None,
        negations = ("لا", "ماشي", "خاطي", "ماهوش", "ما", "بلا", "والو", "ابدا", "ماهيش", "خاطية", "",""),
        modifiers = ("RB",),
        modifier  = lambda w: w.endswith("ment"),
        tokenizer = parser.find_tokens,
        language = "dz"
)

def tokenize(s, *args, **kwargs):
    """ Returns a list of sentences, where punctuation marks have been split from words.
    """
    return parser.find_tokens(s, *args, **kwargs)


def split(s, token=[WORD, POS, CHUNK, PNP]):
    """ Returns a parsed Text from the given parsed string.
    """
    return Text(s, token)


def polarity(s, **kwargs):
    """ Returns the sentence polarity (positive/negative) between -1.0 and 1.0.
    """
    return sentiment(s, **kwargs)[0]


def subjectivity(s, **kwargs):
    """ Returns the sentence subjectivity (objective/subjective) between 0.0 and 1.0.
    """
    return sentiment(s, **kwargs)[1]


def positive(s, threshold=0.1, **kwargs):
    """ Returns True if the given sentence has a positive sentiment (polarity >= threshold).
    """
    return polarity(s, **kwargs) >= threshold

