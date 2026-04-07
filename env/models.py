from pydantic import BaseModel
from typing import Optional

class TriageAction(BaseModel):
    action_type: str
    patient_id: int

class TriageObservation(BaseModel):
    er_queue_size: int
    icu_available: int
    ward_available: int
    avg_patient_health: float
    critical_count: int
    warning_signal: Optional[str] = None
    next_patient_severity: int = 0       # Severity of patient at front of queue (0 = empty)
    next_patient_health: float = 100.0   # HP of patient at front of queue
    deaths: int = 0                      # Deaths so far this episode
    steps_remaining: int = 20            # Steps left (budget awareness)
    most_urgent_health: float = 100.0