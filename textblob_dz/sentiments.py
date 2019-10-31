# -*- coding: utf-8 -*-
"""Algerian dialect sentiment analysis implementations.
"""
from __future__ import absolute_import
from textblob.base import BaseSentimentAnalyzer, CONTINUOUS
from textblob_dz.dz import sentiment as pattern_sentiment_dz


class PatternAnalyzer(BaseSentimentAnalyzer):

    '''Sentiment analyzer that uses the same implementation as the
    pattern library. Returns results as a tuple of the form:
    ``(polarity, subjectivity)``
    '''

    kind = CONTINUOUS

    def analyze(self, text, dialect="dz"):
        """Return the sentiment as a tuple of the form:
        ``(polarity, subjectivity)``
        """
        return pattern_sentiment_dz(text)

