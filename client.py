
from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import HospitalMgmtAction, HospitalMgmtObservation
except ImportError:
    from models import HospitalMgmtAction, HospitalMgmtObservation

class HospitalMgmtEnv(
    EnvClient[HospitalMgmtAction, HospitalMgmtObservation, State]
):
    """
    Client for the Hospital Mgmt Env Environment.
    """

    def _step_payload(self, action: HospitalMgmtAction) -> Dict:
        return {
            "action_type": action.action_type,
        }

    def _parse_result(self, payload: Dict) -> StepResult[HospitalMgmtObservation]:
        obs_data = payload.get("observation", {})
        observation = HospitalMgmtObservation(
            er_queue_size=obs_data.get("er_queue_size", 0),
            icu_available=obs_data.get("icu_available", 0),
            ward_available=obs_data.get("ward_available", 0),
            avg_patient_health=obs_data.get("avg_patient_health", 100.0),
            critical_count=obs_data.get("critical_count", 0),
            warning_signal=obs_data.get("warning_signal"),
            next_patient_severity=obs_data.get("next_patient_severity", 0),
            next_patient_health=obs_data.get("next_patient_health", 100.0),
            deaths=obs_data.get("deaths", 0),
            steps_remaining=obs_data.get("steps_remaining", 20),
            most_urgent_health=obs_data.get("most_urgent_health", 100.0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
