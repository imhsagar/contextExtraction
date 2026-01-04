from ninja import Schema
from typing import Optional, Union
from datetime import date

class RuleSchema(Schema):
    rule_id: str
    rule_summary: str
    measurement_basis: str

class TaskSchema(Schema):
    # Allow int or str for ID
    task_id: Union[str, int, float]

    # Make task_name Optional to prevent crashes on empty rows
    task_name: Optional[str] = None

    duration_days: Optional[int] = None
    start_date: Optional[date] = None
    finish_date: Optional[date] = None