# Generated by Django 5.1.7 on 2025-06-06 08:33

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("videodelivery", "0001_initial"),
    ]

    operations = [
        migrations.DeleteModel(
            name="BreathingMeasurement",
        ),
        migrations.DeleteModel(
            name="PulseMeasurement",
        ),
    ]
