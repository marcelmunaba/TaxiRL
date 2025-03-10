import streamlit as st

pg = st.navigation(
    [
        st.Page("Home.py"), 
        st.Page("algorithms/RandomSample.py"), 
        st.Page("algorithms/QLearning.py"), 
        st.Page("algorithms/ValueIteration.py")
        ]
    )
pg.run()
