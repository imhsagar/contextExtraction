from prefect import task
from core.utils.chunk_utils import (
    aggregate_tasks_by_building,
    create_semantic_chunks
)

@task(name="Transform Schedule Data")
def transform_schedule_data(tasks):
    """
    Takes extracted TaskSchema list and outputs:
    - row_chunks
    - summary_chunks
    using Option C
    """
    summaries = aggregate_tasks_by_building(tasks)
    row_chunks, summary_chunks = create_semantic_chunks(tasks, summaries)
    return row_chunks, summary_chunks
