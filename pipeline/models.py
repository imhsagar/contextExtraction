# pipeline/models.py
from django.db import models

class ProjectTask(models.Model):
    task_id = models.IntegerField()
    task_name = models.TextField(null=True, blank=True)
    duration_days = models.IntegerField(null=True, blank=True)
    start_date = models.DateField(null=True, blank=True)
    finish_date = models.DateField(null=True, blank=True)

    def __str__(self):
        return f"{self.task_id}: {self.task_name}"

class RegulatoryRule(models.Model):
    rule_id = models.CharField(max_length=255, unique=True)
    rule_summary = models.TextField()
    measurement_basis = models.TextField()

    def __str__(self):
        return self.rule_id