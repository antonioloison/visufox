# BIPO App
# Authors: Setra Rakotovao, Antonio Loison, Lila Sainero, Géraud Faye

import json
from collections import defaultdict

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TOP_NUMBER_OF_COUNTRIES = 20

st.set_page_config(layout="wide")

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
    agriculture_data = pd.read_csv("data/agriculture_data.csv", delimiter=";")
    with open("assets/countries.json") as f:
        countries = json.load(f)
    regions = defaultdict(list)
    for code, country in countries.items():
        regions[country["region"]].append(code)
        regions["All world"].append(code)
    return agriculture_data, countries, regions


agriculture_data, countries, regions = load_data()

region_option = st.selectbox(
    "Select your region...",
    ("All world",
     "East Asia & Pacific",
     "Europe & Central Asia",
     "Latin America & Caribbean",
     "Middle East & North Africa",
     "North America",
     "South Asia",
     "Sub-Saharan Africa "))

st.write("You selected:", region_option)

# Display region bar chart
region_df = agriculture_data[agriculture_data["Country Code"].isin(regions[region_option])]
region_df = region_df[region_df["Year"] == "2015-2020"].sort_values("average_value_Cereal production (metric tons)",
                                                                    ascending=False)


fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Bar(
        x=region_df["Country Name"][:TOP_NUMBER_OF_COUNTRIES],
        y=region_df["average_value_Cereal production (metric tons)"][:TOP_NUMBER_OF_COUNTRIES],
        name="Average Cereal Production",
        marker=dict(color="green")
    ), secondary_y=False)

fig.add_trace(
    go.Scatter(
        x=region_df["Country Name"][:TOP_NUMBER_OF_COUNTRIES],
        y=region_df["average_value_Fertilizer consumption (kilograms per hectare of arable land)"][
          :TOP_NUMBER_OF_COUNTRIES],
        name="Average Fertilizer Consumption",
        marker=dict(color="brown")
    ), secondary_y=True)
fig.update_layout(
    title_text=f"Top {TOP_NUMBER_OF_COUNTRIES} Biggest Cereal Producers in {region_option}"
)

# Set x-axis title
fig.update_xaxes(title_text="Country names")

# Set y-axes titles
fig.update_yaxes(title_text="Average Cereal Production (metric tons)", secondary_y=False)
fig.update_yaxes(title_text="Average Fertilizer Consumption \n(kilograms per hectare of arable land)", secondary_y=True)

# fig.update_layout(xaxis=list(range = c(0,10)))
st.plotly_chart(fig, use_container_width=True)
