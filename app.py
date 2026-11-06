"""Entry point for the multi-page Streamlit app.

Defines the page set (Home, Guided Demo, Simulator) and hands off to
st.navigation; each page's own logic lives under views/. Does not
contain any simulation logic itself.
"""

import streamlit as st

st.set_page_config(page_title="CPS Attack Simulation", layout="wide")

pages = [
    st.Page("views/home.py", title="Home", icon=":material/home:", default=True),
    st.Page("views/demo.py", title="Guided Demo", icon=":material/play_circle:"),
    st.Page("views/simulator.py", title="Simulator", icon=":material/tune:"),
]

nav = st.navigation(pages)
nav.run()
