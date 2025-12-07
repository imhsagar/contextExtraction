from ninja import Schema
from typing import Optional
from datetime import date

class RuleSchema(Schema):
    rule_id: str
    rule_summary: str
    measurement_basis: str

class TaskSchema(Schema):
    task_id: int
    task_name: str
    duration_days: int
    start_date: Optional[date] = None
    finish_date: Optional[date] = None