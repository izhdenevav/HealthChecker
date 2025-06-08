from django.db import models
from django.contrib.auth.models import User

class Measurement(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    pulse = models.FloatField()
    breathing = models.FloatField()
    head_position = models.CharField(max_length=100)
    created_at = models.DateTimeField()