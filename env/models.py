from pydantic import BaseModel
from typing import Optional

class TriageAction(BaseModel):
    action_type: str
    patient_id: int

class TriageObservation(BaseModel):
    # Core fields
    er_queue_size: int
    icu_available: int
    ward_available: int
    avg_patient_health: float
    critical_count: int
    warning_signal: Optional[str] = None
    # Extended fields (Change 1: agent can now see the next patient)
    next_patient_severity: int = 0       # Severity of patient at front of queue (0 = empty)
    next_patient_health: float = 100.0   # HP of patient at front of queue
    deaths: int = 0                      # Deaths so far this episode
    steps_remaining: int = 20            # Steps left (budget awareness)
    most_urgent_health: float = 100.0    # Lowest HP in queue (urgency signal)