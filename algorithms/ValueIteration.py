import time
import streamlit as st
import gymnasium as gym
from ansi2html import Ansi2HTMLConverter

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
    value=99,
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

# def train_value_iteration():
#     # Create the environment in ANSI mode
#     env = gym.make('Taxi-v3', render_mode='ansi')
#     state_size = env.observation_space.n

#     # Initialize States Table - there are 500 states in total
#     states = np.zeros((state_size))

#     # Training loop
#     for episode in range(num_episodes):
#         state, _ = env.reset()
#         done = False

#     # Training finished
#     log_placeholder1.text("")
#     st.success(f"✅ Training completed over {num_episodes} episodes!")
#     st.write("📊 Value for each state after training:")
#     st.dataframe(states, use_container_width=True)
#     return states, env

if st.button("🚀 Train Agent"):
    states, env = train_value_iteration()
    st.session_state["states"] = states

st.divider()

# def run_trained_agent(qtable):
#     # Reset the environment to get a fresh starting state
#     env = gym.make('Taxi-v3', render_mode='ansi')
#     state, info = env.reset()
#     action_mask = info.get("action_mask", None)
#     done = False
#     rewards = 0

#     for step in range(50):
#         # If there's an action_mask, pick from valid actions
#         if action_mask is not None:
#             valid_actions = [a for a, valid in enumerate(action_mask) if valid == 1]
#             # Exploit Q-table among valid actions
#             best_action = None
#             best_q = float('-inf')
#             for a in valid_actions:
#                 if qtable[state, a] > best_q:
#                     best_q = qtable[state, a]
#                     best_action = a
#             action = best_action
#         else:
#             # Fallback if action_mask not present
#             action = np.argmax(qtable[state, :])

#         new_state, reward, done, truncated, info = env.step(action)
#         new_state = new_state if isinstance(new_state, int) else new_state[0]
#         action_mask = info.get("action_mask", None)
#         rewards += reward

#         # Display live updates of the agent's actions and score
#         log_placeholder2.markdown(
#             f"🚀 **Step {step+1}:** Action {action}, Reward {reward}, Total Score: {rewards}",
#             unsafe_allow_html=True
#         )

#         # Render the grid with the updated state
#         render_env_as_text(env)

#         state = new_state
#         if done or truncated:
#             break
        
#         time.sleep(0.5)  # Slow down for readability
#     if done:
#         st.success(f"🎉 Trained agent finished the episode with total reward of {rewards}!")
#     else:
#         st.warning("⚠️ Trained agent reached the maximum number of steps. Try playing around with the number of episodes/steps per episode.")

# Streamlit Placeholders
log_placeholder2 = st.empty()
taxi_display = st.empty()

if st.button("🏁 Run a simulation with the trained agent"):
    if "qtable" in st.session_state:
        qtable = st.session_state["qtable"]
        #run_trained_agent(qtable)
    else:
        st.warning("Please train the agent first!")
        
st.page_link("Home.py", label="Back to Home", icon="🏠")