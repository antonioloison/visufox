# BIPO App
# Authors: Setra Rakotovao, Antonio Loison, Lila Sainero, Géraud Faye

import json
from collections import defaultdict

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
from vega_datasets import data

TOP_NUMBER_OF_COUNTRIES = 20

st.set_page_config(layout="wide")

st.write("# BIPO: Biggest Agricultural Polluters")

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
if region_option == "All world":
    region_df = agriculture_data[agriculture_data["Country Name"] == "World"]
else:
    region_df = agriculture_data[agriculture_data["Country Name"] == region_option]

region_df = region_df[region_df["Year"] != "2015-2020"]
region_countries_df = agriculture_data[agriculture_data["Country Code"].isin(regions[region_option])]

region_countries_df = region_countries_df[region_countries_df["Year"] == "2015-2020"].sort_values("average_value_Cereal production (metric tons)",
                                                                    ascending=False)


fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Bar(
        x=region_countries_df["Country Name"][:TOP_NUMBER_OF_COUNTRIES],
        y=region_countries_df["average_value_Cereal production (metric tons)"][:TOP_NUMBER_OF_COUNTRIES],
        name="Average Cereal Production",
        marker=dict(color="green")
    ), secondary_y=False)

fig.add_trace(
    go.Scatter(
        x=region_countries_df["Country Name"][:TOP_NUMBER_OF_COUNTRIES],
        y=region_countries_df["average_value_Fertilizer consumption (kilograms per hectare of arable land)"][
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

col1, col2 = st.columns(2)

with col1:
    st.header("Emissions Details")

    # TO DO

with col2:
    st.header("Evolution over time")

    country_option = st.selectbox(
        "Select your region...",
        tuple(region_countries_df["Country Name"]))

    country_df = agriculture_data[agriculture_data["Country Name"] == country_option]
    country_df = country_df[country_df["Year"] != "2015-2020"]
    comparison_df = pd.concat([region_df, country_df])

    time_evolution = alt.Chart(comparison_df).mark_line(point=True).encode(
        alt.X('average_value_Cereal production (metric tons)'),
        alt.Y('average_value_Agricultural methane emissions (thousand metric tons of CO2 equivalent)'),
        order='Year',
        color="Country Name"
    )

    col2.altair_chart(time_evolution)
