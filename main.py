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
        regions["World"].append(code)
    return agriculture_data, countries, regions


agriculture_data, countries, regions = load_data()

region_option = st.selectbox(
    "Select your region...",
    ("World",
     "East Asia & Pacific",
     "Europe & Central Asia",
     "Latin America & Caribbean",
     "Middle East & North Africa",
     "North America",
     "South Asia",
     "Sub-Saharan Africa "))

st.write("You selected:", region_option)

# Display region bar chart
region_df = agriculture_data[agriculture_data["Country Name"] == region_option]

region_df = region_df[region_df["Year"] != "2015-2020"]
region_countries_df = agriculture_data[agriculture_data["Country Code"].isin(regions[region_option])]

region_countries_df = region_countries_df[region_countries_df["Year"] == "2015-2020"].sort_values(
    "average_value_Cereal production (metric tons)",
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

country_option = st.selectbox(
    "Select your country...",
    tuple(region_countries_df["Country Name"]))

region_df = agriculture_data[agriculture_data["Country Name"] == region_option]
region_df = region_df[~region_df["Year"].isin(["1969", "2015-2020"])]

country_df = agriculture_data[agriculture_data["Country Name"] == country_option]
country_df = country_df[~country_df["Year"].isin(["1969", "2015-2020"])]

col1, col2 = st.columns(2)

with col1:
    st.header("Emissions Details")

    col_pollutions = [
        {
            "name": "Methane emissions",
            "key": "average_value_Agricultural methane emissions (thousand metric tons of CO2 equivalent)",
            "to_normalize": True
        },
        {
            "name": "Nitrous oxide emissions",
            "key": "average_value_Agricultural nitrous oxide emissions (thousand metric tons of CO2 equivalent)",
            "to_normalize": True
        },
        {
            "name": "Fertilizer consumption",
            "key": "average_value_Fertilizer consumption (kilograms per hectare of arable land)",
            "to_normalize": False
        },
        {
            "name": "Annual freshwater withdrawals",
            "key": "average_value_Annual freshwater withdrawals, agriculture (% of total freshwater withdrawal)",
            "to_normalize": False
        }
    ]
    categories, region_values, country_values, region_real_values, country_real_values = [], [], [], [], []

    logscale = st.checkbox('logscale')
    for col_info in col_pollutions:
        colname = col_info["key"]
        all_data = agriculture_data[["Country Name", "Population", colname]].copy()
        if col_info["to_normalize"]:
            all_data[colname] /= all_data["Population"]
        all_data = all_data.groupby("Country Name")[colname].mean()
        all_data = all_data[all_data != 0]
        max_value, min_value = all_data.max(), all_data.min()
        if logscale:
            max_value, min_value = np.log(max_value), np.log(min_value)
        categories.append(col_info["name"])
        region_value = region_df[colname].copy()
        if col_info["to_normalize"]:
            region_value /= region_df["Population"]
        region_mean_value = np.log(region_value.mean()) if logscale else region_value.mean()
        region_real_values.append(region_mean_value)
        region_value = max((region_mean_value - min_value) / (max_value - min_value), 0)
        region_values.append(region_value)
        country_value = country_df[colname].copy()
        if col_info["to_normalize"]:
            country_value /= country_df["Population"]
        country_mean_value = np.log(country_value.mean()) if logscale else country_value.mean()
        country_real_values.append(country_mean_value)
        country_value = max((country_mean_value - min_value) / (max_value - min_value), 0)
        country_values.append(country_value)

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=region_values,
        theta=categories,
        fill='toself',
        name=region_option
    ))
    fig.add_trace(go.Scatterpolar(
        r=country_values,
        theta=categories,
        fill='toself',
        name=country_option
    ))

    fig.update_layout(
    title='0 = min, 1 = max',
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    showlegend=True
    )

    st.plotly_chart(fig)

with col2:
    st.header("Evolution over time")


    def get_label(date, df):
        """ Gets the label of the first and last point in the chart """
        years = df[df[["average_value_Agricultural methane emissions (thousand metric tons of CO2 equivalent)",
                       "average_value_Cereal production (metric tons)",
                       "Population"]].notnull().all(axis=1)]["Year"]
        if date in [min(years), max(years)]:
            return date
        else:
            return ""


    region_df["label"] = region_df["Year"].apply(lambda x: get_label(x, region_df))
    country_df["label"] = country_df["Year"].apply(lambda x: get_label(x, country_df))

    comparison_df = pd.concat([region_df, country_df])

    comparison_df["polution_per_pop"] = \
        comparison_df["average_value_Agricultural methane emissions (thousand metric tons of CO2 equivalent)"] / \
        comparison_df["Population"]
    comparison_df["cereal_prod_per_pop"] = comparison_df["average_value_Cereal production (metric tons)"] / \
                                           comparison_df["Population"]
    comparison_df = comparison_df[["polution_per_pop",
                                   "cereal_prod_per_pop",
                                   "Population",
                                   "Year",
                                   "Country Name",
                                   "label"]]

    time_evolution = alt.Chart(comparison_df).mark_line(point=True).encode(
        alt.X('cereal_prod_per_pop', axis=alt.Axis(title="Cereal Production Per Person")),
        alt.Y('polution_per_pop', axis=alt.Axis(title="Methane Emissions Per Person")),
        order='Year',
        color="Country Name"
    ).properties(
        width=600,
        height=400
    )

    text = time_evolution.mark_text(
        align='left',
        baseline='middle',
        dx=7,
        dy=7,
        fontSize=16
    ).encode(
        text='label'
    )

    col2.altair_chart(time_evolution + text)
