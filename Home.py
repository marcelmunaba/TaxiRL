import numpy as np
import gymnasium as gym
import random
import streamlit as st
import time
import re
from ansi2html import Ansi2HTMLConverter

url = "https://gymnasium.farama.org/environments/toy_text/taxi/"

st.sidebar.header("Official Documentation") 
st.sidebar.write(f"For more information, please refer to Gymnasium's official Taxi environment [documentation]({url}).")
# Streamlit App Title
st.title("Welcome to TaxiRL 🚖")

st.write("""
         In this project, we explore different methods to solve the Taxi Problem using Reinforcement Learning and how they compare between them.
        """)

st.write("""     
    The main goal is to pick up a passenger at one location (Blue) and drop them off at another (Magenta). The taxi will receive a reward for successfully delivering the passenger, and a penalty for executing illegal actions. The episode ends when the passenger is dropped off at the destination.
""")
st.subheader("**Map**")
st.code("""
    +---------+
    |R: | : :G|
    | : | : : |
    | : : : : |
    | | : | : |
    |Y| : |B: |
    +---------+
    """)
st.subheader("**Rewards**")
st.write("""
    -1 per step unless other reward is triggered.

    +20 delivering passenger.

    -10 executing “pickup” and “drop-off” actions illegally.

    An action that results a noop, like moving into a wall, will incur the time step penalty.
    """)
st.divider()
st.subheader("Select an algorithm to start")
st.page_link("algorithms/RandomSample.py", label="Random Sample", icon="🎲")
st.page_link("algorithms/QLearning.py", label="Q-Learning", icon="❇️")
st.page_link("algorithms/ValueIteration.py", label="Value Iteration", icon="🔁")