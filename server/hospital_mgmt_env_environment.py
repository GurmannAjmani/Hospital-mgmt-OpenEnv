# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import HospitalMgmtAction, HospitalMgmtObservation
except ImportError:
    from models import HospitalMgmtAction, HospitalMgmtObservation

from .hospital_logic import PatientManager

SPARSE_REWARDS = {
    "easy":   {"perfect": 0.45, "no_deaths": 0.35, "death_penalty": 0.07, "collapse": 0.05},
    "medium": {"perfect": 0.45, "no_deaths": 0.30, "death_penalty": 0.09, "collapse": 0.05},
    "hard":   {"perfect": 0.45, "no_deaths": 0.20, "death_penalty": 0.07, "collapse": 0.05},
}


class HospitalMgmtEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.logic = PatientManager()
        self.max_steps = 20
        self.task_id = "hard"

    def reset(self, config: dict = None) -> HospitalMgmtObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        if config and "difficulty" in config:
            self.task_id = config["difficulty"]
        
        self.logic.initialize(self.task_id)
        
        if self.task_id == "hard":
            self.max_steps = 16
        elif self.task_id == "medium":
            self.max_steps = 20
        else:
            self.max_steps = 30
            
        return self._get_obs()

    def _get_obs(self, done=False, reward=0.0):
        stats = self.logic.get_stats()
        stats["steps_remaining"] = self.max_steps - self._state.step_count
        return HospitalMgmtObservation(
            done=done,
            reward=reward,
            **stats
        )

    def _compute_sparse_reward(self):
        table = SPARSE_REWARDS.get(self.task_id, SPARSE_REWARDS["hard"])
        deaths = self.logic.deaths
        treated = len([p for p in self.logic.all_patients if p.status == "treated"])
        total = len(self.logic.all_patients)

        if deaths == 0 and treated == total:
            return table["perfect"]
        elif deaths == 0:
            return table["no_deaths"]
        elif deaths < self.logic.collapse_threshold:
            return max(0.05, table["no_deaths"] - table["death_penalty"] * deaths)
        else:
            return table["collapse"]

    def step(self, action: HospitalMgmtAction) -> HospitalMgmtObservation:
        self._state.step_count += 1
        self.logic.update_health()
        reward = self.logic.apply_action(action.action_type)

        done = self._state.step_count >= self.max_steps or self.logic.is_collapsed()
        if done:
            reward += self._compute_sparse_reward()

        return self._get_obs(done=done, reward=reward)

    @property
    def state(self) -> State:
        return self._state
