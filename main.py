# BIPO App
# Authors: Setra Rakotovao, Antonio Loison, Lila Sainero, Géraud Faye

import streamlit as st
import pandas as pd
import altair as alt
from altair import datum
from vega_datasets import data
from streamlit_vega_lite import vega_lite_component, altair_component
import time

st.write("# BIPO: Biggest Polluters")

st.write("""
**Task:** Visualization of agricultural-linked pollution in the world

**Description:**
This tool shows the pollution caused by countries due to agriculture, with respect to their production. 
""")


# Load Gapminder data
# @st.cache decorator skip reloading the code when the apps rerun.
@st.cache
def load_data():
    df = pd.read_csv("data/agriculture_data.csv", delimiter=";")
    return df

df = load_data()

# Use st.write() to render any objects on the web app
st.write(df)
