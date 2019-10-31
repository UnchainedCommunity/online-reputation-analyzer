# online-reputation-analyzer

Algerian Dialect support for `TextBlob`.

Features
--------

* Sentiment analysis (``PatternAnalyzer``)

Usage
-----

.. code-block:: python

    >>> from textblob import Blobber
    >>> from textblob_dz import PatternAnalyzer
    >>> tb = Blobber(analyzer=PatternAnalyzer())
    >>> blob1 = tb(u"نحب تطبيق يسير، هايل")
    >>> blob1.sentiment
    (0.725, 0.7749999999999999)
    >>> blob2 = tb(u"ماهوش مليح")
    >>> blob2.sentiment
    (-0.35, 0.7)

TODO
----

- Algerian Dialect with French Alphabet
- Stop words
- Dialect Morpholigical aspect
- Context influence
- Tagging
- Extending XML File


