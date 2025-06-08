from django.db import models
from django.contrib.auth.models import User

class Measurement(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    pulse = models.FloatField(null=True, blank=True)
    breathing = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField()