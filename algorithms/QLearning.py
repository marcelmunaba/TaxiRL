import numpy as np
import gymnasium as gym
import random
import streamlit as st
import time
from ansi2html import Ansi2HTMLConverter
from matplotlib import pyplot as plt

url = "https://gymnasium.farama.org/environments/toy_text/taxi/"
st.sidebar.header("Official Documentation") 
st.sidebar.write(f"For more information, please refer to Gymnasium's official Taxi environment [documentation]({url}).")

qlearn_url = "https://en.wikipedia.org/wiki/Q-learning"
st.write(f"For more information, check out the wikipedia page for [Q-Learning]({qlearn_url}).")
st.header("Update formula")
st.latex(r"""
Q(s, a) \leftarrow Q(s, a) + \alpha 
\Bigl( 
    r + \gamma \max_{a^\prime} Q\bigl(s^\prime, a^\prime\bigr) 
    \;-\; Q(s, a) 
\Bigr)
""")

st.header("Hyperparameters")

st.write("Feel free to adjust the hyperparameters above to see how they affect the training process.")

learning_rate = st.slider(
        "Learning Rate (Alpha)",
        min_value=0.01,
        max_value=1.0,
        value=0.9,
        step=0.01,
        help="Step size for the Q-value updates.",
    )

discount_rate = st.slider(
    "Discount Rate (Gamma)",
    min_value=0.01,
    max_value=1.0,
    value=0.8,
    step=0.01,
    help="How much future rewards are discounted."
)

epsilon = st.slider(
    "Exploration Rate (Epsilon)",
    min_value=0.0,
    max_value=1.0,
    value=1.0,
    step=0.01,
    help="Probability of choosing a random action."
)

decay_rate = st.slider(
    "Decay Rate for Epsilon",
    min_value=0.0,
    max_value=0.05,
    value=0.005,
    step=0.001,
    help="Exponential decay applied to epsilon each episode."
)

num_episodes = st.slider(
    "Number of Episodes",
    min_value=50,
    max_value=2000,
    value=1000,
    step=50,
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

def train_q_learning():
    """ Trains the Q-learning agent and updates the display dynamically in Streamlit. """
    global epsilon  # Keep track of epsilon decay across runs

    # Create the environment in ANSI mode
    env = gym.make('Taxi-v3', render_mode='ansi')
    state_size = env.observation_space.n
    action_size = env.action_space.n

    # Initialize Q-table
    qtable = np.zeros((state_size, action_size))
    
    # List to track total reward per episode
    episode_rewards = []

    # Training loop
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False

        # Track the total reward this episode
        total_reward = 0

        for step in range(max_steps):
            # Exploration vs. Exploitation - here we use epsilon-greedy with exponentially decaying epsilon
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(qtable[state, :])  # Exploit

            new_state, reward, done, truncated, _ = env.step(action)
            new_state = new_state if isinstance(new_state, int) else new_state[0]

            # Q-learning Update
            qtable[state, action] += learning_rate * (
                reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action]
            )

            state = new_state
            total_reward += reward
            
            # Update training status in Streamlit
            log_placeholder1.text(f"🚀 Training Episode {episode+1}/{num_episodes}, Step {step+1}/{max_steps}")
            if done or truncated:
                break
        
        # Store the total reward for current episode
        episode_rewards.append(total_reward)
        
        # Decay epsilon after each episode
        if epsilon > 0:
            epsilon = np.exp(-decay_rate * episode)

    # Training finished
    log_placeholder1.text("")
    st.success(f"✅ Training completed over {num_episodes} episodes!")
        
    # Set up plot of cumulative rewards per episode
    fig, ax = plt.subplots()
    ax.plot(episode_rewards, label="Q-Learning")
    ax.set_title("Cumulative Reward per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    
    return qtable, env, fig

if st.button("🚀 Train Agent"):
    qtable, env, fig = train_q_learning()
    st.session_state["qtable"] = qtable
    st.session_state["fig"] = fig
    
if "qtable" in st.session_state and st.session_state["qtable"] is not None:
    st.write("📊 Q-Table after training:")
    qtable = st.session_state["qtable"]
    st.dataframe(qtable, use_container_width=True)
    with st.expander("Reading the Q-Table"):
        st.write("""
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
        
if "fig" in st.session_state and st.session_state["fig"] is not None:
    st.pyplot(st.session_state["fig"])

#-------------------------- Simulation --------------------------#

st.divider()

def render_env_as_text(env):
    """ Capture ANSI-rendered Taxi-v3 output, convert ANSI codes to HTML, and display it in Streamlit using st.markdown() """
    ansi_text = env.render()

    # Convert ANSI text to HTML
    html_output = conv.convert(ansi_text)
    # Display the grid with HTML rendering
    taxi_display.markdown(f"<div style='font-family:monospace; white-space: pre;'>{html_output}</div>", unsafe_allow_html=True)

def run_trained_agent(qtable):
    # Reset the environment to get a fresh starting state
    env = gym.make('Taxi-v3', render_mode='ansi')
    state, info = env.reset()
    action_mask = info.get("action_mask", None)
    done = False
    rewards = 0

    for step in range(50):
        # If there's an action_mask, pick from valid actions
        if action_mask is not None:
            valid_actions = [a for a, valid in enumerate(action_mask) if valid == 1]
            # Exploit Q-table among valid actions
            best_action = None
            best_q = float('-inf')
            for a in valid_actions:
                if qtable[state, a] > best_q:
                    best_q = qtable[state, a]
                    best_action = a
            action = best_action
        else:
            # Fallback if action_mask not present
            action = np.argmax(qtable[state, :])

        new_state, reward, done, truncated, info = env.step(action)
        new_state = new_state if isinstance(new_state, int) else new_state[0]
        action_mask = info.get("action_mask", None)
        rewards += reward

        # Display live updates of the agent's actions and score
        log_placeholder2.markdown(
            f"🚀 **Step {step+1}:** Action {action}, Reward {reward}, Total Score: {rewards}",
            unsafe_allow_html=True
        )

        # Render the grid with the updated state
        render_env_as_text(env)

        state = new_state
        if done or truncated:
            break
        
        time.sleep(0.5)  # Slow down for readability
    if done:
        st.success(f"🎉 Trained agent finished the episode with total reward of {rewards}!")
    else:
        st.warning("⚠️ Trained agent reached the maximum number of steps. Try playing around with the number of episodes/steps per episode.")

# Streamlit Placeholders
log_placeholder2 = st.empty()
taxi_display = st.empty()

if st.button("🏁 Run a simulation with the trained agent"):
    if "qtable" in st.session_state and st.session_state["qtable"] is not None:
        qtable = st.session_state["qtable"]
        run_trained_agent(qtable)
    else:
        st.warning("Please train the agent first!")
        
st.page_link("Home.py", label="Back to Home", icon="🏠")