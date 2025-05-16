from django.db import models
from django.contrib.auth.models import User

class HealthMeasurement(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    date = models.DateTimeField(verbose_name="Дата и время измерения")
    hr = models.PositiveSmallIntegerField(verbose_name="Пульс (уд/мин)")
    br = models.PositiveSmallIntegerField(verbose_name="Дыхание (дых/мин)")

    class Meta:
        ordering = ['-date']
        verbose_name = 'Измерение'
        verbose_name_plural = 'Измерения'

    def __str__(self):
        return f"{self.user.username} - {self.date.strftime('%d.%m.%Y %H:%M')}"