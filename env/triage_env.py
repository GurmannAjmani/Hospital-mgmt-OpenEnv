import numpy as np
from .models import TriageObservation
from .hospital_logic import PatientManager

SPARSE_REWARDS = {
    "easy":   {"perfect": 0.95, "no_deaths": 0.75, "death_penalty": 0.14, "collapse": 0.05},
    "medium": {"perfect": 0.95, "no_deaths": 0.65, "death_penalty": 0.18, "collapse": 0.05},
    "hard":   {"perfect": 0.95, "no_deaths": 0.45, "death_penalty": 0.14, "collapse": 0.05},
}

class HospitalTriageEnv:
    """ER Triage Environment.  action_space: 0=wait, 1=icu, 2=ward, 3=discharge"""

    STATE_DIM = 9
    ACTION_DIM = 4

    def __init__(self):
        self.logic = PatientManager()
        self.steps = 0
        self.max_steps = 20
        self.task_id = "easy"

    def reset(self, task_id="easy"):
        self.steps = 0
        self.task_id = task_id
        self.logic.initialize(task_id)
        if task_id == "hard":
            self.max_steps = 16
        elif task_id == "medium":
            self.max_steps = 20
        else:
            self.max_steps = 30
        return self._get_obs()

    def _get_obs(self):
        stats = self.logic.get_stats()
        stats["steps_remaining"] = self.max_steps - self.steps
        return TriageObservation(**stats)

    def obs_to_vector(self, obs: TriageObservation) -> list:
        return [
            obs.er_queue_size,
            obs.icu_available,
            obs.ward_available,
            obs.avg_patient_health / 100.0,
            obs.critical_count,
            obs.next_patient_severity / 5.0,
            obs.next_patient_health / 100.0,
            obs.deaths,
            obs.steps_remaining / self.max_steps,
        ]

    def _compute_sparse_reward(self):
        deaths = self.logic.deaths
        
        if self.logic.is_collapsed():
            return 0.05
            
        penalty_per_death = 0.4 / self.logic.collapse_threshold
        
        return max(0.05, 0.45 - (deaths * penalty_per_death))

    def step(self, action: int):
        self.steps += 1
        self.logic.update_health()
        reward = self.logic.apply_action(action)
        done = self.steps >= self.max_steps or self.logic.is_collapsed()
        if done:
            reward += self._compute_sparse_reward()

        return self._get_obs(), reward, done