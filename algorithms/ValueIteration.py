import time
from matplotlib import pyplot as plt
import streamlit as st
import gymnasium as gym
from ansi2html import Ansi2HTMLConverter
import numpy as np

url = "https://gymnasium.farama.org/environments/toy_text/taxi/"
st.sidebar.header("Official Documentation") 
st.sidebar.write(f"For more information, please refer to Gymnasium's official Taxi environment [documentation]({url}).")

valueIter_url = "https://en.wikipedia.org/wiki/Value_iteration"
st.write(f"For more information, check out the wikipedia page for [Value Iteration]({valueIter_url}).")

st.header("Formula")
st.latex(r'''
V_{k+1}(s) \;=\; \max_{a} \; \sum_{s'} P\bigl(s' \mid s,a\bigr)\,\bigl[\,R(s,a,s') \;+\; \gamma\,V_{k}(s')\bigr]
''')

st.header("Hyperparameters")
st.write("Feel free to adjust the hyperparameters above to see how they affect the training process.")

discount_rate = st.slider(
    "Discount Rate (Gamma)",
    min_value=0.01,
    max_value=1.0,
    value=0.8,
    step=0.01,
    help="How much future rewards are discounted."
)

num_episodes = st.slider(
    "Number of Episodes",
    min_value=10,
    max_value=2000,
    value=100,
    step=10,
    help="How many episodes to train over."
)

max_steps = st.slider(
    "Max Steps per Episode",
    min_value=50,
    max_value=1000,
    value=100,
    step=10,
    help="Maximum steps allowed in one episode."
)

log_placeholder1 = st.empty()
# Initialize ANSI to HTML converter
conv = Ansi2HTMLConverter()

def render_env_as_text(env):
    """ Capture ANSI-rendered Taxi-v3 output, convert ANSI codes to HTML, and display it in Streamlit using st.markdown() """
    ansi_text = env.render()

    # Convert ANSI text to HTML
    html_output = conv.convert(ansi_text)
    # Display the grid with HTML rendering
    taxi_display.markdown(f"<div style='font-family:monospace; white-space: pre;'>{html_output}</div>", unsafe_allow_html=True)

def train_value_iteration():
    # Create the environment in ANSI mode
    env = gym.make('Taxi-v3', render_mode='ansi').unwrapped
    state_size = env.observation_space.n
    action_size = env.action_space.n
    theta = 1e-5
    
    # Initialize States Table (Value table) - there are 500 states in total
    V = np.zeros((state_size))

    # List to track delta - convergence
    delta_plot = []
    
    # Training loop
    for episode in range(num_episodes):
        delta = 0
        # For each state s in the state space
        for s in range(state_size):
            old_value = V[s]
            
            # Compute the value of each action
            #   Q(s, a) = ∑  P(s'|s,a)  [ r(s,a,s') + γ V(s') ]
            #   V_k+1(s) = max_a Q(s, a)
            action_values = []
            # Loop over all actions to find the maximum
            for a in range(action_size):
                transitions = env.P[s][a]  # list of (prob, next_state, reward, done)
                q_sa = 0
                # Loop over all possible transitions
                for (prob, next_s, reward, done) in transitions:
                    q_sa += prob * (reward + discount_rate * V[next_s])
                action_values.append(q_sa)

            # Best possible value from taking the best action - the max in the equation
            best_action_value = max(action_values)
            # Update the Value function
            V[s] = best_action_value

            # Track the biggest difference this iteration
            delta = max(delta, abs(old_value - V[s]))
            log_placeholder1.text(f"🚀 Training Episode {episode+1}/{num_episodes}, Step {s+1}/{max_steps}")
        
        delta_plot.append(delta)   
         
        # Check for convergence
        if delta < theta:
            break
                
    # Training finished
    log_placeholder1.text("")
    st.success(f"✅ Training completed over {num_episodes} episodes!")
    st.write("📊 Value for each state after training:")
    
    # Set up plot of cumulative rewards per episode
    fig, ax = plt.subplots()
    ax.plot(delta_plot, label="Delta")
    ax.set_title("Delta per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Delta")
    ax.legend()
    
    return V, env, fig

if st.button("🚀 Train Agent"):
    V, env, fig = train_value_iteration()
    st.session_state["V"] = V
    st.session_state["fig_ValueIter"] = fig

if "V" in st.session_state and st.session_state["V"] is not None:
    st.dataframe(st.session_state["V"], use_container_width=True)
    
if "fig_ValueIter" in st.session_state and st.session_state["fig_ValueIter"] is not None:
    st.pyplot(st.session_state["fig_ValueIter"])

#-------------------------- Simulation --------------------------#

st.divider()

def run_trained_agent(value_function):
    env = gym.make('Taxi-v3', render_mode='ansi').unwrapped
    state, info = env.reset()
    action_mask = info.get("action_mask", None)
    done = False
    rewards = 0

    for step in range(50):
        # Policy Extraction:
        # Compute Q(s,a) for each possible action, then pick the best.
        # If there's an action_mask, only compute Q for those valid actions:
        if action_mask is not None:
            valid_actions = [a for a, valid in enumerate(action_mask) if valid == 1]
        else:
            valid_actions = range(env.action_space.n)

        best_q = float('-inf')
        best_action = None
        for a in valid_actions:
            # Retrieve transitions: list of (prob, next_state, reward, done)
            transitions = env.P[state][a]
            q_sa = 0
            for (prob, next_s, r, _) in transitions:
                q_sa += prob * (r + discount_rate * value_function[next_s])
            if q_sa > best_q:
                best_q = q_sa
                best_action = a

        # Execute best_action
        action = best_action
        new_state, reward, done, truncated, info = env.step(action)
        new_state = new_state if isinstance(new_state, int) else new_state[0]
        action_mask = info.get("action_mask", None)
        rewards += reward

        log_placeholder2.markdown(
            f"🚀 **Step {step+1}:** Action {action}, Reward {reward}, Total Score: {rewards}",
            unsafe_allow_html=True
        )
        render_env_as_text(env)

        state = new_state
        if done or truncated:
            break

        time.sleep(0.5)

    if done:
        st.success(f"🎉 Trained agent finished the episode with total reward of {rewards}!")
    else:
        st.warning("⚠️ Trained agent reached the max steps limit.")

# Streamlit Placeholders
log_placeholder2 = st.empty()
taxi_display = st.empty()

if st.button("🏁 Run a simulation with the trained agent"):
    if "V" in st.session_state:
        V = st.session_state["V"]
        run_trained_agent(V)
    else:
        st.warning("Please train the agent first!")
        
st.page_link("Home.py", label="Back to Home", icon="🏠")