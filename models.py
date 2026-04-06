# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Data models for the Hospital Mgmt Env Environment.
"""

from typing import Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field

class HospitalMgmtAction(Action):
    """Action for the ER Sentinel triage environment."""
    action_type: int = Field(..., description="0: wait, 1: admit_icu, 2: admit_ward, 3: discharge")

class HospitalMgmtObservation(Observation):
    """Observation from the ER Sentinel environment."""
    er_queue_size: int = Field(default=0, description="Number of patients in queue")
    icu_available: int = Field(default=0, description="Number of available ICU beds")
    ward_available: int = Field(default=0, description="Number of available Ward beds")
    avg_patient_health: float = Field(default=100.0, description="Average health of patients in queue")
    critical_count: int = Field(default=0, description="Number of critical patients (severity <= 2)")
    warning_signal: Optional[str] = Field(default=None, description="Optional system warning")
    next_patient_severity: int = Field(default=0, description="Severity of patient at front of queue")
    next_patient_health: float = Field(default=100.0, description="Health of patient at front of queue")
    deaths: int = Field(default=0, description="Total deaths so far")
    steps_remaining: int = Field(default=20, description="Steps left in the episode")
    most_urgent_health: float = Field(default=100.0, description="Lowest HP in the queue")
