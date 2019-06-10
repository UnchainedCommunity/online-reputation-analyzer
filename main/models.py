from django.db import models


class Keyword(models.Model):
    text = models.CharField(max_length=50)
    # language (default) # choices
    # reference = models.

class Topic(models.Model):
    name = models.CharField(max_length=50)
    keywords = models.ManyToManyField(Keyword)
    is_active = models.BooleanField(default=True)
    # is_public
