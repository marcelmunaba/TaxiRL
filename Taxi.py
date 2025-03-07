import numpy as np
import gymnasium as gym
import random
import streamlit as st
import time
import re
from ansi2html import Ansi2HTMLConverter

# Streamlit App Title
st.title("🚖 Gymnasium's Taxi Problem")

# Hyperparameters
learning_rate = 0.9
discount_rate = 0.8
epsilon = 1.0
decay_rate = 0.005
num_episodes = 1000
max_steps = 99  # Per episode

url = "https://gymnasium.farama.org/environments/toy_text/taxi/"
st.write("Gymnasium's official Taxi environment [documentation](%s)" % url)
st.write(""" 
Choose which algorithm you want to use, and also use the sliders to adjust the learning parameters.""")
QLearning = st.checkbox("Q-Learning", value=True)
Random = st.checkbox("Random", value=False)

# Initialize ANSI to HTML converter
conv = Ansi2HTMLConverter()

def render_env_as_text(env):
    """ Capture ANSI-rendered Taxi-v3 output, convert ANSI codes to HTML, and display it in Streamlit using st.markdown() """
    ansi_text = env.render()

    # Convert ANSI text to HTML
    html_output = conv.convert(ansi_text)
    # Display the grid with HTML rendering
    taxi_display.markdown(f"<div style='font-family:monospace; white-space: pre;'>{html_output}</div>", unsafe_allow_html=True)

def train_q_learning():
    """ Trains the Q-learning agent and updates the display dynamically in Streamlit. """
    global epsilon  # Keep track of epsilon decay across runs

    env = gym.make('Taxi-v3', render_mode='ansi')  # Use text-based ANSI rendering
    state_size = env.observation_space.n
    action_size = env.action_space.n

    # Initialize Q-table
    qtable = np.zeros((state_size, action_size))

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False

        for step in range(max_steps):
            # Exploration-exploitation tradeoff
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(qtable[state, :])  # Exploit

            new_state, reward, done, truncated, _ = env.step(action)
            new_state = new_state if isinstance(new_state, int) else new_state[0]

            # Q-learning Update Rule
            qtable[state, action] = qtable[state, action] + learning_rate * (
                reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action]
            )
            # Update Streamlit logs (Single Line)
            log_placeholder1.markdown(f"🎬 **Episode {episode+1}/{num_episodes}**", unsafe_allow_html=True)

            state = new_state
            if done or truncated:
                break

        # Decay epsilon
        epsilon = np.exp(-decay_rate * episode)
    # Once training is finished, display the final Q-table exactly once
    st.success(f"✅ Training completed over {num_episodes} episodes!")
    st.write("📊 Q-Table after training:")
    st.dataframe(qtable, use_container_width=True)
    st.write("""
    **How to read the Q-Table**:

    There are **500 discrete states** since there are 25 taxi positions, 5 possible locations of the passenger (including the case when the passenger is in the taxi), and 4 destination locations. Destinations on the map are represented with the first letter of the color.

    **Passenger locations**:
    - 0: Red
    - 1: Green
    - 2: Yellow
    - 3: Blue
    - 4: In taxi

    **Destinations**:
    - 0: Red
    - 1: Green
    - 2: Yellow
    - 3: Blue
    """)
    return qtable, env


def run_trained_agent(qtable):
    """ Runs the trained Q-learning agent in Streamlit and updates display dynamically. """
    st.subheader("🏁 Running Trained Agent...")
    
    # Reset the environment to get a fresh starting state
    env = gym.make('Taxi-v3', render_mode='ansi')  # Use text-based ANSI rendering
    state, _ = env.reset()
    done = False
    rewards = 0

    for step in range(50):
        action = np.argmax(qtable[state, :])
        new_state, reward, done, truncated, _ = env.step(action)
        rewards += reward

        # Display live updates of the agent's actions and score
        log_placeholder2.markdown(f"🚀 **Step {step+1}:** Action {action}, Reward {reward}, Total Score: {rewards}", unsafe_allow_html=True)

        # Render the grid with the highlighted taxi
        render_env_as_text(env)

        state = new_state
        if done or truncated:
            break
        
        time.sleep(0.5)  # Small delay to make logs readable

    st.success("🎉 Trained agent finished the episode!")

def RandomSample():
    st.warning("For future implementation")
    # # create Taxi environment
    # env = gym.make('Taxi-v3')

    # # create a new instance of taxi, and get the initial state
    # state = env.reset()

    # num_steps = 99
    # for s in range(num_steps+1):
    #     print(f"step: {s} out of {num_steps}")

    #     # sample a random action from the list of available actions
    #     action = env.action_space.sample()

    #     # perform this action on the environment
    #     env.step(action)

    #     # print the new state
    #     render_env_as_text(env)

    # # end this instance of the taxi environment
    # env.close()

# Streamlit Placeholders for Live Updates
log_placeholder1 = st.empty()

# Streamlit Button to Train & Run the Agent
if st.button("🚀 Train Agent"):
    if QLearning and Random:
        st.warning("Please select only one algorithm to train the agent!")
    elif QLearning:
        qtable, env = train_q_learning()
        st.session_state["qtable"] = qtable
    elif Random:
        RandomSample()
    else:
        st.warning("Please select an algorithm to train the agent!")
        
log_placeholder2 = st.empty()
taxi_display = st.empty()
    
if st.button("🏁 Run Trained Agent"):
    # Retrieve the qtable from session state
    if "qtable" in st.session_state:
        qtable = st.session_state["qtable"]
        run_trained_agent(qtable)
    else:
        st.warning("Please train the agent first!")