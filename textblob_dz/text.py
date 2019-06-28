# -*- coding: utf-8 -*-
'''This file is adapted from the pattern library.
URL: http://www.clips.ua.ac.be/pages/pattern-web
Licence: BSD
'''
from __future__ import unicode_literals
import string
from itertools import chain
import types
import os
import re
from xml.etree import cElementTree

from .compat import text_type, string_types, basestring, imap, unicode

try:
    MODULE = os.path.dirname(os.path.abspath(__file__))
except:
    MODULE = ""

SLASH, WORD, POS, CHUNK, PNP, REL, ANCHOR, LEMMA = \
        "&slash;", "word", "part-of-speech", "chunk", "preposition", "relation", "anchor", "lemma"


# String functions
def decode_string(v, encoding="utf-8"):
    """ Returns the given value as a Unicode string (if possible).
    """
    if type(encoding) in string_types:
        encoding = ((encoding,),) + (("windows-1252",), ("utf-8", "ignore"))
    if type(v) in string_types:
        for e in encoding:
            try: return v.decode(*e)
            except:
                pass
        return v
    return str(v)

def encode_string(v, encoding="utf-8"):
    """ Returns the given value as a Python byte string (if possible).
    """
    if type(encoding) in string_types:
        encoding = ((encoding,),) + (("windows-1252",), ("utf-8", "ignore"))
    if type(v) in string_types:
        for e in encoding:
            try: return v.encode(*e)
            except:
                pass
        return v
    return str(v)

decode_utf8 = decode_string
encode_utf8 = encode_string

def isnumeric(strg):
    try:
        float(strg)
    except ValueError:
        return False
    return True

#--- LAZY DICTIONARY -------------------------------------------------------------------------------
# A lazy dictionary is empty until one of its methods is called.
# This way many instances (e.g., lexicons) can be created without using memory until used.

class lazydict(dict):

    def load(self):
        # Must be overridden in a subclass.
        # Must load data with dict.__setitem__(self, k, v) instead of lazydict[k] = v.
        pass

    def _lazy(self, method, *args):
        """ If the dictionary is empty, calls lazydict.load().
            Replaces lazydict.method() with dict.method() and calls it.
        """
        if dict.__len__(self) == 0:
            self.load()
            setattr(self, method, types.MethodType(getattr(dict, method), self))
        return getattr(dict, method)(self, *args)

    def __repr__(self):
        return self._lazy("__repr__")
    def __len__(self):
        return self._lazy("__len__")
    def __iter__(self):
        return self._lazy("__iter__")
    def __contains__(self, *args):
        return self._lazy("__contains__", *args)
    def __getitem__(self, *args):
        return self._lazy("__getitem__", *args)
    def __setitem__(self, *args):
        return self._lazy("__setitem__", *args)
    def setdefault(self, *args):
        return self._lazy("setdefault", *args)
    def get(self, *args, **kwargs):
        return self._lazy("get", *args)
    def items(self):
        return self._lazy("items")
    def keys(self):
        return self._lazy("keys")
    def values(self):
        return self._lazy("values")
    def update(self, *args):
        return self._lazy("update", *args)
    def pop(self, *args):
        return self._lazy("pop", *args)
    def popitem(self, *args):
        return self._lazy("popitem", *args)

class lazylist(list):

    def load(self):
        # Must be overridden in a subclass.
        # Must load data with list.append(self, v) instead of lazylist.append(v).
        pass

    def _lazy(self, method, *args):
        """ If the list is empty, calls lazylist.load().
            Replaces lazylist.method() with list.method() and calls it.
        """
        if list.__len__(self) == 0:
            self.load()
            setattr(self, method, types.MethodType(getattr(list, method), self))
        return getattr(list, method)(self, *args)

    def __repr__(self):
        return self._lazy("__repr__")
    def __len__(self):
        return self._lazy("__len__")
    def __iter__(self):
        return self._lazy("__iter__")
    def __contains__(self, *args):
        return self._lazy("__contains__", *args)
    def insert(self, *args):
        return self._lazy("insert", *args)
    def append(self, *args):
        return self._lazy("append", *args)
    def extend(self, *args):
        return self._lazy("extend", *args)
    def remove(self, *args):
        return self._lazy("remove", *args)
    def pop(self, *args):
        return self._lazy("pop", *args)

#--- UNIVERSAL TAGSET ------------------------------------------------------------------------------
# The default part-of-speech tagset used in Pattern is Penn Treebank II.
# However, not all languages are well-suited to Penn Treebank (which was developed for English).
# As more languages are implemented, this is becoming more problematic.
#
# A universal tagset is proposed by Slav Petrov (2012):
# http://www.petrovi.de/data/lrec.pdf
#
# Subclasses of Parser should start implementing
# Parser.parse(tagset=UNIVERSAL) with a simplified tagset.
# The names of the constants correspond to Petrov's naming scheme, while
# the value of the constants correspond to Penn Treebank.

UNIVERSAL = "universal"


# Handle common punctuation marks.
PUNCTUATION = \
punctuation = ".,;:!?()[]{}`''\"@#$^&*+-|=~_"

# Handle common abbreviations.
ABBREVIATIONS = ""

RE_ABBR1 = re.compile("^[A-Za-z]\.$")       # single letter, "T. De Smedt"
RE_ABBR2 = re.compile("^([A-Za-z]\.)+$")    # alternating letters, "U.S."
RE_ABBR3 = re.compile("^[A-Z][" + "|".join( # capital followed by consonants, "Mr."
        "bcdfghjklmnpqrstvwxz") + "]+.$")

# Handle emoticons.
EMOTICONS = { # (facial expression, sentiment)-keys
    ("love" , +1.00): set(("<3", "♥")),
    ("grin" , +1.00): set((">:D", ":-D", ":D", "=-D", "=D", "X-D", "x-D", "XD", "xD", "8-D")),
    ("taunt", +0.75): set((">:P", ":-P", ":P", ":-p", ":p", ":-b", ":b", ":c)", ":o)", ":^)")),
    ("smile", +0.50): set((">:)", ":-)", ":)", "=)", "=]", ":]", ":}", ":>", ":3", "8)", "8-)")),
    ("wink" , +0.25): set((">;]", ";-)", ";)", ";-]", ";]", ";D", ";^)", "*-)", "*)")),
    ("gasp" , +0.05): set((">:o", ":-O", ":O", ":o", ":-o", "o_O", "o.O", "°O°", "°o°")),
    ("worry", -0.25): set((">:/",  ":-/", ":/", ":\\", ">:\\", ":-.", ":-s", ":s", ":S", ":-S", ">.>")),
    ("frown", -0.75): set((">:[", ":-(", ":(", "=(", ":-[", ":[", ":{", ":-<", ":c", ":-c", "=/")),
    ("cry"  , -1.00): set((":'(", ":'''(", ";'("))
}

RE_EMOTICONS = [r" ?".join([re.escape(each) for each in e]) for v in EMOTICONS.values() for e in v]
RE_EMOTICONS = re.compile(r"(%s)($|\s)" % "|".join(RE_EMOTICONS))

# Handle sarcasm punctuation (!).
RE_SARCASM = re.compile(r"\( ?\! ?\)")

# Handle common contractions.
replacements = {}

# Handle paragraph line breaks (\n\n marks end of sentence).
EOS = "END-OF-SENTENCE"

def find_tokens(string, punctuation=PUNCTUATION, abbreviations=ABBREVIATIONS, replace=replacements, linebreak=r"\n{2,}"):
    """ Returns a list of sentences. Each sentence is a space-separated string of tokens (words).
        Handles common cases of abbreviations (e.g., etc., ...).
        Punctuation marks are split from other words. Periods (or ?!) mark the end of a sentence.
        Headings without an ending period are inferred by line breaks.
    """
    # Handle periods separately.
    punctuation = tuple(punctuation.replace(".", ""))


    # Handle Unicode quotes.
    if type(string) in string_types:
        string = unicode(string).replace("“", " “ ")\
                                .replace("”", " ” ")\
                                .replace("‘", " ‘ ")\
                                .replace("’", " ’ ")\
                                .replace("'", " ' ")\
                                .replace('"', ' " ')
    # Collapse whitespace.
    string = re.sub("\r\n", "\n", string)
    string = re.sub(linebreak, " %s " % EOS, string)
    string = re.sub(r"\s+", " ", string)
    tokens = []
    for t in TOKEN.findall(string+" "):
        if len(t) > 0:
            tail = []
            while t.startswith(punctuation) and \
              not t in replace:
                # Split leading punctuation.
                if t.startswith(punctuation):
                    tokens.append(t[0]); t=t[1:]
            while t.endswith(punctuation+(".",)) and \
              not t in replace:
                # Split trailing punctuation.
                if t.endswith(punctuation):
                    tail.append(t[-1]); t=t[:-1]
                # Split ellipsis (...) before splitting period.
                if t.endswith("..."):
                    tail.append("..."); t=t[:-3].rstrip(".")
                # Split period (if not an abbreviation).
                if t.endswith("."):
                    if t in abbreviations or \
                      RE_ABBR1.match(t) is not None or \
                      RE_ABBR2.match(t) is not None or \
                      RE_ABBR3.match(t) is not None:
                        break
                    else:
                        tail.append(t[-1]); t=t[:-1]
            if t != "":
                tokens.append(t)
            tokens.extend(reversed(tail))
    sentences, i, j = [[]], 0, 0
    while j < len(tokens):
        if tokens[j] in ("...", ".", "!", "?", EOS):
            # There may be a trailing parenthesis.
            while j < len(tokens) \
              and tokens[j] in ("...", ".", "!", "?", ")", "'", "\"", "”", "’", EOS):
                j += 1
            sentences[-1].extend(t for t in tokens[i:j] if t != EOS)
            sentences.append([])
            i = j
        j += 1
    sentences[-1].extend(tokens[i:j])
    sentences = (" ".join(s) for s in sentences if len(s) > 0)
    sentences = (RE_SARCASM.sub("(!)", s) for s in sentences)
    sentences = [RE_EMOTICONS.sub(
        lambda m: m.group(1).replace(" ", "") + m.group(2), s) for s in sentences]
    return sentences



### SENTIMENT POLARITY LEXICON #####################################################################
# A sentiment lexicon can be used to discern objective facts from subjective opinions in text.
# Each word in the lexicon has scores for:
# 1)     polarity: negative vs. positive    (-1.0 => +1.0)
# 2) subjectivity: objective vs. subjective (+0.0 => +1.0)
# 3)    intensity: modifies next word?      (x0.5 => x2.0)

# For English, adverbs are used as modifiers (e.g., "very good").
# For Dutch, adverbial adjectives are used as modifiers
# ("hopeloos voorspelbaar", "ontzettend spannend", "verschrikkelijk goed").
# Negation words (e.g., "not") reverse the polarity of the following word.

# Sentiment()(txt) returns an averaged (polarity, subjectivity)-tuple.
# Sentiment().assessments(txt) returns a list of (chunk, polarity, subjectivity, label)-tuples.

# Semantic labels are useful for fine-grained analysis, e.g.,
# negative words + positive emoticons could indicate cynicism.

# Semantic labels:
MOOD  = "mood"  # emoticons, emojis
IRONY = "irony" # sarcasm mark (!)

NOUN, VERB, ADJECTIVE, ADVERB = \
    "NN", "VB", "JJ", "RB"

RE_SYNSET = re.compile(r"^[acdnrv][-_][0-9]+$")

def avg(list):
    return sum(list) / float(len(list) or 1)

class Score(tuple):

    def __new__(self, polarity, subjectivity, assessments=[]):
        """ A (polarity, subjectivity)-tuple with an assessments property.
        """
        return tuple.__new__(self, [polarity, subjectivity])

    def __init__(self, polarity, subjectivity, assessments=[]):
        self.assessments = assessments

### SENTIMENT POLARITY LEXICON #####################################################################
# A sentiment lexicon can be used to discern objective facts from subjective opinions in text.
# Each word in the lexicon has scores for:
# 1)     polarity: negative vs. positive    (-1.0 => +1.0)
# 2) subjectivity: objective vs. subjective (+0.0 => +1.0)
# 3)    intensity: modifies next word?      (x0.5 => x2.0)


# Sentiment()(txt) returns an averaged (polarity, subjectivity)-tuple.
# Sentiment().assessments(txt) returns a list of (chunk, polarity, subjectivity, label)-tuples.

# Semantic labels are useful for fine-grained analysis, e.g.,
# negative words + positive emoticons could indicate cynicism.

# Semantic labels:
MOOD  = "mood"  # emoticons, emojis
IRONY = "irony" # sarcasm mark (!)

NOUN, VERB, ADJECTIVE, ADVERB = \
    "NN", "VB", "JJ", "RB"

RE_SYNSET = re.compile(r"^[acdnrv][-_][0-9]+$")

def avg(list):
    return sum(list) / float(len(list) or 1)

class Score(tuple):

    def __new__(self, polarity, subjectivity, assessments=[]):
        """ A (polarity, subjectivity)-tuple with an assessments property.
        """
        return tuple.__new__(self, [polarity, subjectivity])

    def __init__(self, polarity, subjectivity, assessments=[]):
        self.assessments = assessments

class Sentiment(lazydict):

    def __init__(self, path="", language=None, synset=None, confidence=None, **kwargs):
        """ A dictionary of words (adjectives) and polarity scores (positive/negative).
            The value for each word is a dictionary of part-of-speech tags.
            The value for each word POS-tag is a tuple with values for
            polarity (-1.0-1.0), subjectivity (0.0-1.0) and intensity (0.5-2.0).
        """
        self._path = path   # XML file path.
        self._language = None   # XML language attribute ("en", "fr", ...)
        self._confidence = None   # XML confidence attribute threshold (>=).
        self._synset = synset # XML synset attribute ("wordnet_id", "cornetto_id", ...)
        self._synsets    = {}     # {"a-01123879": (1.0, 1.0, 1.0)}
        self.labeler     = {}     # {"dammit": "profanity"}
        self.tokenizer   = kwargs.get("tokenizer", find_tokens)
        self.negations   = kwargs.get("negations", ("no", "not", "n't", "never"))
        self.modifiers   = kwargs.get("modifiers", ("RB",))
        self.modifier    = kwargs.get("modifier" , lambda w: w.endswith("ly"))

    @property
    def path(self):
        return self._path

    @property
    def language(self):
        return self._language

    @property
    def confidence(self):
        return self._confidence

    def load(self, path=None):
        """ Loads the XML-file (with sentiment annotations) from the given path.
            By default, Sentiment.path is lazily loaded.
        """
        # <word form="great" wordnet_id="a-01123879" pos="JJ" polarity="1.0" subjectivity="1.0" intensity="1.0" />
        # <word form="damnmit" polarity="-0.75" subjectivity="1.0" label="profanity" />
        if not path:
            path = self._path
        if not os.path.exists(path):
            return
        words, synsets, labels = {}, {}, {}
        xml = cElementTree.parse(path)
        xml = xml.getroot()
        for w in xml.findall("word"):
            if self._confidence is None \
            or self._confidence <= float(w.attrib.get("confidence", 0.0)):
                w, pos, p, s, i, label, synset = (
                    w.attrib.get("form"),
                    w.attrib.get("pos"),
                    w.attrib.get("polarity", 0.0),
                    w.attrib.get("subjectivity", 0.0),
                    w.attrib.get("intensity", 1.0),
                    w.attrib.get("label"),
                    w.attrib.get(self._synset) # wordnet_id, cornetto_id, ...
                )
                psi = (float(p), float(s), float(i))
                if w:
                    words.setdefault(w, {}).setdefault(pos, []).append(psi)
                if w and label:
                    labels[w] = label
                if synset:
                    synsets.setdefault(synset, []).append(psi)
        self._language = xml.attrib.get("language", self._language)
        # Average scores of all word senses per part-of-speech tag.
        for w in words:
            words[w] = dict((pos, [avg(each) for each in zip(*psi)]) for pos, psi in words[w].items())
        # Average scores of all part-of-speech tags.
        for w, pos in list(words.items()):
            words[w][None] = [avg(each) for each in zip(*pos.values())]
        # Average scores of all synonyms per synset.
        for id, psi in synsets.items():
            synsets[id] = [avg(each) for each in zip(*psi)]
        dict.update(self, words)
        dict.update(self.labeler, labels)
        dict.update(self._synsets, synsets)


    def __call__(self, s, negation=True, **kwargs):

        """ Returns a (polarity, subjectivity)-tuple for the given sentence,
            with polarity between -1.0 and 1.0 and subjectivity between 0.0 and 1.0.
            The sentence can be a string, Synset, Text, Sentence, Chunk, Word, Document, Vector.
            An optional weight parameter can be given,
            as a function that takes a list of words and returns a weight.
        """
        def avg(assessments, weighted=lambda w: 1):
            s, n = 0, 0
            for words, score in assessments:
                w = weighted(words)
                s += w * score
                n += w
            return s / float(n or 1)
        # A pattern.en.wordnet.Synset.
        # Sentiment(synsets("horrible", "JJ")[0]) => (-0.6, 1.0)
        if hasattr(s, "gloss"):
            a = [(s.synonyms[0],) + self.synset(s.id, pos=s.pos) + (None,)]
        # A synset id.
        # Sentiment("a-00193480") => horrible => (-0.6, 1.0)   (English WordNet)
        # Sentiment("c_267") => verschrikkelijk => (-0.9, 1.0) (Dutch Cornetto)
        elif isinstance(s, basestring) and RE_SYNSET.match(s):
            a = [(s.synonyms[0],) + self.synset(s.id, pos=s.pos) + (None,)]
        # A string of words.
        # Sentiment("a horrible movie") => (-0.6, 1.0)
        elif isinstance(s, basestring):
            a = self.assessments(((w.lower(), None) for w in " ".join(self.tokenizer(s)).split()), negation)
        # A pattern.en.Text.
        elif hasattr(s, "sentences"):
            a = self.assessments(((w.lemma or w.string.lower(), w.pos[:2]) for w in chain(*s)), negation)
        # A pattern.en.Sentence or pattern.en.Chunk.
        elif hasattr(s, "lemmata"):
            a = self.assessments(((w.lemma or w.string.lower(), w.pos[:2]) for w in s.words), negation)
        # A pattern.en.Word.
        elif hasattr(s, "lemma"):
            a = self.assessments(((s.lemma or s.string.lower(), s.pos[:2]),), negation)
        # A pattern.vector.Document.
        # Average score = weighted average using feature weights.
        # Bag-of words is unordered: inject None between each two words
        # to stop assessments() from scanning for preceding negation & modifiers.
        elif hasattr(s, "terms"):
            a = self.assessments(chain(*(((w, None), (None, None)) for w in s)), negation)
            kwargs.setdefault("weight", lambda w: s.terms[w[0]])
        # A dict of (word, weight)-items.
        elif isinstance(s, dict):
            a = self.assessments(chain(*(((w, None), (None, None)) for w in s)), negation)
            kwargs.setdefault("weight", lambda w: s[w[0]])
        # A list of words.
        elif isinstance(s, list):
            a = self.assessments(((w, None) for w in s), negation)
        else:
            a = []
        weight = kwargs.get("weight", lambda w: 1) # [(w, p) for w, p, s, x in a]
        return Score(polarity = avg( [(w, p) for w, p, s, x in a], weight ),
                 subjectivity = avg([(w, s) for w, p, s, x in a], weight),
                  assessments = a)

    def assessments(self, words=[], negation=True):
        """ Returns a list of (chunk, polarity, subjectivity, label)-tuples for the given list of words:
            where chunk is a list of successive words: a known word optionally
            preceded by a modifier ("very good") or a negation ("not good").
        """
        a = []
        m = None # Preceding modifier (i.e., adverb or adjective).
        n = None # Preceding negation (e.g., "not beautiful").
        for w, pos in words:
            # Only assess known words, preferably by part-of-speech tag.
            # Including unknown words (polarity 0.0 and subjectivity 0.0) lowers the average.
            if w is None:
                continue
            if w in self and pos in self[w]:
                p, s, i = self[w][pos]
                # Known word not preceded by a modifier ("good").
                if m is None:
                    a.append(dict(w=[w], p=p, s=s, i=i, n=1, x=self.labeler.get(w)))
                # Known word preceded by a modifier ("really good").
                if m is not None:
                    a[-1]["w"].append(w)
                    a[-1]["p"] = max(-1.0, min(p * a[-1]["i"], +1.0))
                    a[-1]["s"] = max(-1.0, min(s * a[-1]["i"], +1.0))
                    a[-1]["i"] = i
                    a[-1]["x"] = self.labeler.get(w)
                # Known word preceded by a negation ("not really good").
                if n is not None:
                    a[-1]["w"].insert(0, n)
                    a[-1]["i"] = 1.0 / a[-1]["i"]
                    a[-1]["n"] = -1
                # Known word may be a negation.
                # Known word may be modifying the next word (i.e., it is a known adverb).
                m = None
                n = None
                if pos and pos in self.modifiers or any(map(self[w].__contains__, self.modifiers)):
                    m = (w, pos)
                if negation and w in self.negations:
                    n = w
            else:
                # Unknown word may be a negation ("not good").
                if negation and w in self.negations:
                    n = w
                # Unknown word. Retain negation across small words ("not a good").
                elif n and len(w.strip("'")) > 1:
                    n = None
                # Unknown word may be a negation preceded by a modifier ("really not good").
                if n is not None and m is not None and (pos in self.modifiers or self.modifier(m[0])):
                    a[-1]["w"].append(n)
                    a[-1]["n"] = -1
                    n = None
                # Unknown word. Retain modifier across small words ("really is a good").
                elif m and len(w) > 2:
                    m = None
                # Exclamation marks boost previous word.
                if w == "!" and len(a) > 0:
                    a[-1]["w"].append("!")
                    a[-1]["p"] = max(-1.0, min(a[-1]["p"] * 1.25, +1.0))
                # Exclamation marks in parentheses indicate sarcasm.
                if w == "(!)":
                    a.append(dict(w=[w], p=0.0, s=1.0, i=1.0, n=1, x=IRONY))
                # EMOTICONS: {("grin", +1.0): set((":-D", ":D"))}
                if w.isalpha() is False and len(w) <= 5 and w not in PUNCTUATION: # speedup
                    for (type, p), e in EMOTICONS.items():
                        if w in imap(lambda e: e.lower(), e):
                            a.append(dict(w=[w], p=p, s=1.0, i=1.0, n=1, x=MOOD))
                            break
        for i in range(len(a)):
            w = a[i]["w"]
            p = a[i]["p"]
            s = a[i]["s"]
            n = a[i]["n"]
            x = a[i]["x"]
            # "not good" = slightly bad, "not bad" = slightly good.
            a[i] = (w, p * -0.5 if n < 0 else p, s, x)
        return a

    def annotate(self, word, pos=None, polarity=0.0, subjectivity=0.0, intensity=1.0, label=None):
        """ Annotates the given word with polarity, subjectivity and intensity scores,
            and optionally a semantic label (e.g., MOOD for emoticons, IRONY for "(!)").
        """
        w = self.setdefault(word, {})
        w[pos] = w[None] = (polarity, subjectivity, intensity)
        if label:
            self.labeler[word] = label



