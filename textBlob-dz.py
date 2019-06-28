from textblob import Blobber
from textblob_dz import PatternAnalyzer

tb = Blobber(analyzer=PatternAnalyzer())
neg = u"ماهوش مليح"
pos = u"هايلة بزاف"

blob1 = tb(pos)
print(blob1.sentiment)

blob2 = tb(neg)
print(blob2.sentiment)


