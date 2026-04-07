"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]
"""

import os
import re
import textwrap
from typing import List, Optional
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

from models import HospitalMgmtAction, HospitalMgmtObservation
from server.hospital_mgmt_env_environment import HospitalMgmtEnvironment
from training.dqn_agent import DQNAgent

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
TASK_NAME = os.getenv("HOSPITAL_MGMT_TASK", "hard")
BENCHMARK = os.getenv("HOSPITAL_MGMT_BENCHMARK", "hospital_mgmt_env")
MAX_STEPS = 30
TEMPERATURE = 0.1
MAX_TOKENS = 10
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI Triage Agent in a busy hospital ER.
    Your goal is to maximize total reward by keeping patients alive and managing beds proactively.
    You will see the current hospital state, queue size, available beds (ICU and Ward), and the next patient's severity (1 is critical, 5 is low) and health.
    Each turn you must choose ONE action from the following options:
    0 - wait (do nothing)
    1 - admit_icu (admit patient to ICU)
    2 - admit_ward (admit patient to Ward)
    3 - discharge (discharge patient)

    Reply with exactly one integer: 0, 1, 2, or 3. Do not include any explanations or extra characters.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, obs: HospitalMgmtObservation, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Queue Size: {obs.er_queue_size}
        ICU Available: {obs.icu_available}
        Ward Available: {obs.ward_available}
        Critical Count: {obs.critical_count}
        Next Patient Severity: {obs.next_patient_severity}
        Next Patient Health: {obs.next_patient_health:.1f}
        Deaths: {obs.deaths}
        Steps Remaining: {obs.steps_remaining}
        Last reward: {last_reward:.2f}
        
        Previous steps:
        {history_block}
        
        Send your next action (0, 1, 2, or 3):
        """
    ).strip()


def get_model_action(client: OpenAI, step: int, obs: HospitalMgmtObservation, last_reward: float, history: List[str]) -> int:
    user_prompt = build_user_prompt(step, obs, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        match = re.search(r'[0-3]', text)
        return int(match.group()) if match else 0
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return 0


def obs_to_vector(obs: HospitalMgmtObservation, max_steps: int) -> List[float]:
    """Vectorize observation for DQN."""
    return [
        float(obs.er_queue_size),
        float(obs.icu_available),
        float(obs.ward_available),
        obs.avg_patient_health / 100.0,
        float(obs.critical_count),
        obs.next_patient_severity / 5.0,
        obs.next_patient_health / 100.0,
        float(obs.deaths),
        obs.steps_remaining / float(max_steps),
    ]


def run_llm_inference(client: OpenAI, task_name: str) -> None:
    """Run baseline LLM inference."""
    env = HospitalMgmtEnvironment()
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        obs = env.reset(config={"difficulty": task_name})
        last_reward = 0.0
        max_steps_local = env.max_steps

        for step in range(1, max_steps_local + 1):
            if obs.done:
                break

            action_idx = get_model_action(client, step, obs, last_reward, history)
            obs = env.step(HospitalMgmtAction(action_type=action_idx))

            reward = obs.reward or 0.0
            done = obs.done
            
            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=str(action_idx), reward=reward, done=done, error=None)
            history.append(f"Step {step}: Action={action_idx} -> reward {reward:+.2f}")

            if done:
                break

        max_reward = 12.5 if task_name in ["hard", "medium"] else 10.5
        score = min(max(sum(rewards) / max_reward, 0.01), 0.99)
        success = score >= SUCCESS_SCORE_THRESHOLD
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    except Exception as e:
        print(f"[DEBUG] LLM Runtime error: {e}", flush=True)


def train_dqn(task_name: str, epochs: int = 150) -> DQNAgent:
    """Train DQN agent."""
    env = HospitalMgmtEnvironment()
    # State dim is 9, Action dim is 4
    agent = DQNAgent(state_dim=9, action_dim=4)
    
    for _ in range(epochs):
        obs = env.reset(config={"difficulty": task_name})
        state = obs_to_vector(obs, env.max_steps)
        done = False
        
        while not done:
            action = agent.select_action(state)
            obs = env.step(HospitalMgmtAction(action_type=action))
            next_state = obs_to_vector(obs, env.max_steps)
            reward = obs.reward or 0.0
            done = obs.done
            
            agent.memory.append((state, action, reward, next_state, float(done)))
            agent.train_step()
            
            state = next_state
    
    return agent


def run_dqn_inference(agent: DQNAgent, task_name: str) -> None:
    """Run inference using trained DQN agent."""
    env = HospitalMgmtEnvironment()
    rewards: List[float] = []
    steps_taken = 0
    
    log_start(task=task_name, env=BENCHMARK, model="DQN-Agent")
    
    try:
        obs = env.reset(config={"difficulty": task_name})
        max_steps_local = env.max_steps

        for step in range(1, max_steps_local + 1):
            if obs.done:
                break

            state = obs_to_vector(obs, max_steps_local)
            epsilon_save = agent.epsilon
            agent.epsilon = 0.0
            action_idx = agent.select_action(state)
            agent.epsilon = epsilon_save
            
            obs = env.step(HospitalMgmtAction(action_type=action_idx))

            reward = obs.reward or 0.0
            done = obs.done
            
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=str(action_idx), reward=reward, done=done, error=None)

            if done:
                break

        max_reward = 12.5 if task_name in ["hard", "medium"] else 10.5
        score = min(max(sum(rewards) / max_reward, 0.01), 0.99)
        success = score >= SUCCESS_SCORE_THRESHOLD
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    except Exception as e:
        print(f"[DEBUG] DQN Runtime error: {e}", flush=True)


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    run_llm_inference(client, TASK_NAME)
    tasks = ["easy", "medium", "hard"]
    for task in tasks:
        agent = train_dqn(task, epochs=500)
        run_dqn_inference(agent, task)



if __name__ == "__main__":
    main()
