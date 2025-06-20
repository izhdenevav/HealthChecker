# Generated by Django 5.1.7 on 2025-06-06 08:30

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0001_initial"),
    ]

    operations = [
        migrations.AlterModelOptions(
            name="healthmeasurement",
            options={},
        ),
        migrations.RemoveField(
            model_name="healthmeasurement",
            name="br",
        ),
        migrations.RemoveField(
            model_name="healthmeasurement",
            name="date",
        ),
        migrations.RemoveField(
            model_name="healthmeasurement",
            name="hr",
        ),
        migrations.AddField(
            model_name="healthmeasurement",
            name="breathing_rate",
            field=models.FloatField(default=0),
        ),
        migrations.AddField(
            model_name="healthmeasurement",
            name="heart_rate",
            field=models.FloatField(default=0),
        ),
        migrations.AddField(
            model_name="healthmeasurement",
            name="timestamp",
            field=models.DateTimeField(auto_now_add=True, null=True),
        ),
    ]
