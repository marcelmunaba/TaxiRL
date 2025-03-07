import streamlit as st

pg = st.navigation([st.Page("Home.py"), st.Page("QLearning.py"), st.Page("RandomSample.py")])
pg.run()
