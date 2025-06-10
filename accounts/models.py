from django.db import models
from django.contrib.auth.models import User

# Модель для хранения измерений пульса и дыхания пользователя
class HealthMeasurement(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE) # Связь с пользователем
    date = models.DateTimeField(verbose_name="Дата и время измерения")
    hr = models.PositiveSmallIntegerField(verbose_name="Пульс (уд/мин)")
    br = models.PositiveSmallIntegerField(verbose_name="Дыхание (дых/мин)")

    class Meta:
        ordering = ['-date'] # Последние измерения первыми
        verbose_name = 'Измерение'
        verbose_name_plural = 'Измерения'

    def __str__(self):
        return f"{self.user.username} - {self.date.strftime('%d.%m.%Y %H:%M')}"
