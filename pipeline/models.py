from django.db import models

# Create your models here.
class RegulatoryRule(models.Model):
    rule_id = models.CharField(max_length=50, unique=True, help_text="Unique identifier (e.g., Q1)")
    rule_summary = models.TextField(help_text="Concise summary of the rule")
    measurement_basis = models.CharField(max_length=500, help_text="Key measurement principle")

    source_file = models.CharField(max_length=255, default="URA-Circular.pdf")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.rule_id

class ProjectTask(models.Model):
    task_id = models.IntegerField(help_text="Unique ID from the ID column")
    task_name = models.CharField(max_length=255, help_text="Name of the task")
    duration_days = models.IntegerField(help_text="Duration in days")
    start_date = models.DateField(null=True, blank=True)
    finish_date = models.DateField(null=True, blank=True)

    source_file = models.CharField(max_length=255, default="Project-Schedule.pdf")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.task_id} - {self.task_name}"