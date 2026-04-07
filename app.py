import os
import sys
import numpy as np
import pandas as pd
import streamlit as st


from env.triage_env import HospitalTriageEnv
from training.dqn_agent import DQNAgent

st.set_page_config(page_title="Hospital ER tracker", layout="wide")

st.title("Hospital Triage Management System ")
success_placeholder = st.empty()
with st.sidebar:
    st.header("Training Settings")
    epochs = st.number_input("Number of Epochs (Recommended: 500)", min_value=10, max_value=50000, value=500, step=50, help="Default and recommended is 500")
    difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"], index=0)
    update_freq = 1
    
    st.markdown("---")
    train_btn = st.button("Start Training", use_container_width=True, type="primary")
    reset_btn = st.button("Reset Environment", use_container_width=True)
    if reset_btn:
        st.rerun()

st.subheader("Training Progress")
progress_text = st.empty()
progress_bar = st.empty()
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(" Reward vs Epoch")
    chart_placeholder = st.empty()
    chart_placeholder.line_chart(pd.DataFrame(columns=["Reward", "100-Ep Moving Avg"]), color=["#d3d3d3", "#ff0000"])

with col2:
    st.subheader("Action Logs (epoch wise)")
    log_container = st.container(height=600)
    if "train_btn" not in locals() or not train_btn:
        log_container.info("Click 'Start Training' in the sidebar to begin monitoring.")


if train_btn:
    env = HospitalTriageEnv()
    agent = DQNAgent(
        state_dim=HospitalTriageEnv.STATE_DIM,
        action_dim=HospitalTriageEnv.ACTION_DIM,
    )
    
    all_rewards = []
    progress_bar.progress(0.0)
    
    for ep in range(1, epochs + 1):
        obs = env.reset(task_id=difficulty)
        state = env.obs_to_vector(obs)
        episode_reward = 0.0
        done = False
        
        log_this_epoch = (ep == epochs or ep % update_freq == 0 or ep == 1)
        if log_this_epoch:
            logs = [f"--- EPISODE {ep} BEGIN ---"]
            
        step = 0
        while not done:
            action = agent.select_action(state)
            
            if log_this_epoch:
                actions = ["Wait/None", "Admit to ICU", "Admit to Ward", "Discharge"]
                act_str = actions[action] if action < len(actions) else f"Action {action}"
                q = env.logic.get_stats()["er_queue_size"]
                logs.append(f"Step {step:>2} | Queue: {q} | Action: {act_str}")
                
            obs, reward, done = env.step(action)
            next_state = env.obs_to_vector(obs)
            
            agent.memory.append((state, action, reward, next_state, float(done)))
            agent.train_step()
            
            state = next_state
            episode_reward += reward
            step += 1
            
        all_rewards.append(episode_reward)
        
        if log_this_epoch:
            deaths = env.logic.get_stats()["deaths"]
            log_str = "\n".join(logs)
            
            with log_container.expander(f"Episode {ep} | {deaths} Deaths | Reward: {episode_reward:.2f}", expanded=False):
                st.code(log_str, language="text")
            
            df = pd.DataFrame({"Reward": all_rewards})
            df["100-Ep Moving Avg"] = df["Reward"].rolling(min(100, max(1, len(all_rewards))), min_periods=1).mean()
            chart_placeholder.line_chart(df, color=["#d3d3d3", "#ff0000"])
            
            pct = ep / epochs
            progress_bar.progress(pct)
            progress_text.markdown(f"**Processed:** `{ep} / {epochs}` epochs")
            
    progress_bar.progress(1.0)
    success_placeholder.success(f"Training completed successfully! Final moving average reward: {np.mean(all_rewards[-100:]):.2f}")
