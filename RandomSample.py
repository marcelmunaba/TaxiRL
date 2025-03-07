import time
import streamlit as st
import gymnasium as gym
from ansi2html import Ansi2HTMLConverter

url = "https://gymnasium.farama.org/environments/toy_text/taxi/"
st.sidebar.header("Official Documentation") 
st.sidebar.write(f"For more information, please refer to Gymnasium's official Taxi environment [documentation]({url}).")

st.header("Random Sample")
st.write("In this section, we will run a simulation with a random agent that samples actions from the environment's action space.")
st.write("Due to the random sampling, the agent could be stuck in an episode for indefinitely long periods.")
log_placeholder = st.empty()
taxi_display = st.empty()
conv = Ansi2HTMLConverter()

max_steps = st.slider(
    "Max Steps per Episode",
    min_value=10,
    max_value=1000,
    value=100,
    step=10,
    help="Maximum steps allowed in one episode."
)

def render_env_as_text(env):
    """ Capture ANSI-rendered Taxi-v3 output, convert ANSI codes to HTML, and display it in Streamlit using st.markdown() """
    ansi_text = env.render()

    # Convert ANSI text to HTML
    html_output = conv.convert(ansi_text)
    # Display the grid with HTML rendering
    taxi_display.markdown(f"<div style='font-family:monospace; white-space: pre;'>{html_output}</div>", unsafe_allow_html=True)

def run_agent():
    # create Taxi environment
    env = gym.make('Taxi-v3', render_mode='ansi')

    # create a new instance of taxi, and get the initial state
    state = env.reset()
    rewards = 0
    num_steps = 99
    for s in range(max_steps+1):
        print(f"step: {s} out of {num_steps}")

        # sample a random action from the list of available actions
        action = env.action_space.sample()

        # perform this action on the environment
        ew_state, reward, done, truncated, _ = env.step(action)
        
        log_placeholder.markdown(
            f"🚀 **Step {s+1}:** Action {action}, Reward {reward}, Total Score: {rewards}",
            unsafe_allow_html=True
        )
        rewards += reward
        # print the new state
        render_env_as_text(env)
        time.sleep(0.1)
    # end this instance of the taxi environment
    env.close()
    if done:
        st.success(f"🎉 Agent finished the episode with total reward of {rewards}!")
    else:
        st.warning("⚠️ Trained agent reached the maximum number of steps and does not reach the goal in the given maximum steps.")
    
if st.button("🏁 Run simulation"):
    run_agent()
    
st.page_link("Home.py", label="Back to Home", icon="🏠")
    