# BIPO App
# Authors: Setra Rakotovao, Antonio Loison, Lila Sainero, Géraud Faye

import json
from collections import defaultdict
from tkinter import TOP
from turtle import color, fillcolor

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
from vega_datasets import data
from streamlit_plotly_events import plotly_events
from plotly.graph_objs import *


TOP_NUMBER_OF_COUNTRIES = 20
with open('metric_to_display.txt', 'r') as key:
    key_metric_to_display = key.read()
    key_metric_to_display = int(key_metric_to_display)%4

    
col_pollutions = [
        {
            "name": "Methane emissions",
            "key": "average_value_Agricultural methane emissions (thousand metric tons of CO2 equivalent)",
            "metric": "eqCO2 kg",
            "to_normalize": True,
            "logscale": True,
            "factor": 1e6
        },
        {
            "name": "Nitrous oxide emissions",
            "key": "average_value_Agricultural nitrous oxide emissions (thousand metric tons of CO2 equivalent)",
            "metric": "eqCO2 kg",
            "to_normalize": True,
            "logscale": True,
            "factor": 1e6
        },
        {
            "name": "Fertilizer consumption",
            "key": "average_value_Fertilizer consumption (kilograms per hectare of arable land)",
            "metric": "kilograms per hectare of arable land",
            "to_normalize": False,
            "logscale": True,
        },
        {
            "name": "Annual freshwater withdrawals caused by agriculture",
            "key": "average_value_Annual freshwater withdrawals, agriculture (% of total freshwater withdrawal)",
            "metric": "% of total freshwater withdrawal",
            "to_normalize": False,
            "logscale": False,
        },
        {
            "name": "Total fertilizer consumption",
            "key": "Total fertilizer consumption",
            "metric": "kilograms",
            "to_normalize": False,
            "logscale": False,
        }
    ]

st.set_page_config(
     page_title="BAPO: Biggest Agricultural Polluters",
     page_icon="🚜",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'About': " This application shows the characteristics of the biggest agricultural polluters in the different regions of the world. \n You can check how individual countries compare to other countries of the region or to average figures of their region. \n The purpose of this app is to clearly visualize the difference between countries in terms of agricultural pollution and have an idea of the reasons of their pollution \n *Enjoy!*\n\n The data comes from World Bank Data and can be downloaded at https://github.com/ZeningQu/World-Bank-Data-by-Indicators/blob/master/agriculture-and-rural-development/agriculture-and-rural-development.csv"
     }
 )

bigcol1, bigcol2 = st.columns((1,2))

with bigcol1:

    st.write("# BAPO: Biggest Agricultural Polluters")

    st.write("""
    **Task:** Visualization of agricultural-linked pollution in the world

    **Description:** This application shows the characteristics of the biggest agricultural polluters in the different regions of 
    the world. You can check how individual countries compare to other countries of the region or to average 
    figures of their region. The purpose of this app is to clearly visualize the difference between countries in 
    terms of agricultural pollution and have an idea of the reasons of their pollution. Enjoy!
    
    The data comes from World Bank Data and can be downloaded at this [link](https://github.com/ZeningQu/World-Bank-Data-by-Indicators/blob/master/agriculture-and-rural-development/agriculture-and-rural-development.csv).
    
    """)


    # Load Gapminder data
    # @st.cache decorator skip reloading the code when the apps rerun.
    @st.cache
    def load_data():
        with open("country_to_display.txt", "w") as f:
            f.write("World")
        with open("metric_to_display.txt", "w") as f:
            f.write("2")
        agriculture_data = pd.read_csv("data/agriculture_data.csv", delimiter=";")
        with open("assets/countries.json") as f:
            countries = json.load(f)
        regions = defaultdict(list)
        for code, country in countries.items():
            regions[country["region"]].append(code)
            regions["World"].append(code)
        agriculture_data['Total fertilizer consumption'] = agriculture_data["average_value_Fertilizer consumption (kilograms per hectare of arable land)"] * \
                    agriculture_data["average_value_Arable land (hectares)"] / 1e3
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

    year_min, year_max = st.slider('Year', min_value=1960, max_value=2020, value=(2010, 2020))


with bigcol2:
# Display region bar chart

    if region_option == "North America":
        TOP_NUMBER_OF_COUNTRIES = 3
    elif region_option == "South Asia":
        TOP_NUMBER_OF_COUNTRIES = 8

    st.header(f"Top {TOP_NUMBER_OF_COUNTRIES} Biggest Cereal Producers in {region_option}")

    region_df = agriculture_data[agriculture_data["Country Name"] == region_option]


    region_df = region_df[region_df["Year"] != "2015-2020"]
    region_countries_df = agriculture_data[agriculture_data["Country Code"].isin(regions[region_option])]

    region_countries_df = region_countries_df[region_countries_df["Year"] == "2015-2020"].sort_values(
        "average_value_Cereal production (metric tons)",
        ascending=False)

    region_countries_df['Total fertilizer consumption'] = region_countries_df["average_value_Fertilizer consumption (kilograms per hectare of arable land)"] * \
        region_countries_df["average_value_Arable land (hectares)"] / 1e3

    fig = make_subplots(specs=[[{"secondary_y": False}]])

    if key_metric_to_display == 1:
        scale = 1000
    elif key_metric_to_display == 2:
        scale = 10
        key_metric_to_display = 4
    elif key_metric_to_display == 3:
        scale = 1000000
    else:
        scale = 1000

    exposant = int(np.log10(np.max(region_countries_df["average_value_Cereal production (metric tons)"][:TOP_NUMBER_OF_COUNTRIES])))

    cereal_max = np.max(region_countries_df["average_value_Cereal production (metric tons)"][:TOP_NUMBER_OF_COUNTRIES])//10**exposant
    if cereal_max < 2:
        exposant = exposant - 1
        cereal_max = 8
    fig.add_trace(
        go.Bar(
            x=region_countries_df["Country Name"][:TOP_NUMBER_OF_COUNTRIES],
            y=region_countries_df["average_value_Cereal production (metric tons)"][:TOP_NUMBER_OF_COUNTRIES],
            base=0,
            name="Average Cereal Production (tons)",
            marker=dict(color="green"), 
            hovertemplate="%{x} <br>"+"Cereal Production: %{y:.0e} tons",
        ), secondary_y=False)

    y_scaled = list(region_countries_df[col_pollutions[key_metric_to_display]["key"]][
                :TOP_NUMBER_OF_COUNTRIES].values)

    try:
        
        fig.add_trace(
            go.Bar(
                x=region_countries_df["Country Name"][:TOP_NUMBER_OF_COUNTRIES],
                y = -scale*region_countries_df[col_pollutions[key_metric_to_display]["key"]][
                :TOP_NUMBER_OF_COUNTRIES],
                name=col_pollutions[key_metric_to_display]["name"],
                marker=dict(color="maroon"), 
                hovertemplate=["%{x} <br>"+"{}: <br> {:.0e} {}".format(col_pollutions[key_metric_to_display]["name"], y_scaled[i], col_pollutions[key_metric_to_display]["metric"]) for i in range(TOP_NUMBER_OF_COUNTRIES)],
            ), secondary_y=False)
    except:
        fig.add_trace(
            go.Bar(
                x=region_countries_df["Country Name"][:TOP_NUMBER_OF_COUNTRIES],
                y=-scale*region_countries_df['Total fertilizer consumption'][
                :TOP_NUMBER_OF_COUNTRIES],
                name="Average Fertilizer Consumption (tons)",
                marker=dict(color="maroon"), 
                hovertemplate=["%{x} <br>"+"Fertilizer consumption: {:.0e} tons".format(y_scaled[i]) for i in range(TOP_NUMBER_OF_COUNTRIES)],
            ), secondary_y=False)

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title_font_color="#391d04",
        font_color="#391d04", 
        barmode='relative',
        margin=go.layout.Margin(
        l=20, #left margin
        r=20, #right margin
        b=80, #bottom margin
        t=0  #top margin
        ), 
        legend=dict(yanchor="top", y=1.1, xanchor="left", x=-0.1), 
        yaxis = dict(
            tickmode = 'array',
            tickvals = [cereal_max//2*i*10**exposant for i in range(-2, 3)],
            ticktext = [f'{-2*i*10**exposant/scale:.0e}' for i in range(-2, 1)] + [f'{abs(2*i*10**exposant):.0e}' for i in range(1, 4)], 

        )
    )


    selected_points_country = plotly_events(fig, click_event=True, hover_event=False)
    try:
        country_to_display = selected_points_country[0]["x"]
        with open("country_to_display.txt", "w") as f:
            f.write(country_to_display)
    except:
        with open("country_to_display.txt", "r") as f:
            country_to_display = f.read()

    st.write("You can select a country on the bar chart above by clicking on the corresponding bar.")
    if country_to_display == "World":
        st.write("There is currently no selected country.")
    else:
        st.write(f"The currently selected country is **{country_to_display}**.")


col1, col2 = st.columns((2,2))

with col1:

    region_df = agriculture_data[agriculture_data["Country Name"] == region_option].copy()
    region_df = region_df[~region_df["Year"].isin(["1969", "2015-2020"])]
    region_df = region_df[(region_df.Year <= str(year_max))&(region_df.Year >= str(year_min))]
    region_df['Total fertilizer consumption'] = region_df["average_value_Fertilizer consumption (kilograms per hectare of arable land)"] * \
        region_df["average_value_Arable land (hectares)"] / 1e3

    country_df = agriculture_data[agriculture_data["Country Name"] == country_to_display].copy()
    country_df = country_df[~country_df["Year"].isin(["1969", "2015-2020"])]
    country_df = country_df[(country_df.Year <= str(year_max))&(country_df.Year >= str(year_min))]
    country_df['Total fertilizer consumption'] = country_df["average_value_Fertilizer consumption (kilograms per hectare of arable land)"] * \
        country_df["average_value_Arable land (hectares)"] / 1e3

    st.header("Emissions Details")

    
    categories, region_values, country_values, region_real_values, country_real_values, metrics, normalized = [], [], [], [], [], [], []

    for col_info in col_pollutions[:-1]:
        colname = col_info["key"]
        factor = col_info.get("factor", 1)
        logscale = col_info["logscale"]

        all_data = agriculture_data[["Country Name", "Population", colname]].copy()
        
        if col_info["to_normalize"]:
            all_data[colname] /= all_data["Population"]
        all_data[colname] = all_data[colname] * factor
        all_data = all_data.groupby("Country Name")[colname].mean()
        all_data = all_data[all_data != 0]
        max_value, min_value = all_data.max(), all_data.min()

        if logscale:
            max_value, min_value = np.log(max_value), np.log(min_value)
        
        categories.append(col_info["name"])
        region_value = region_df[colname].copy()
        
        if col_info["to_normalize"]:
            region_value /= region_df["Population"]
        
        region_value = region_value * factor
        region_mean_value = np.nanmean(np.log(region_value)) if logscale else np.nanmean(region_value)
        region_real_values.append(np.exp(region_mean_value) if logscale else region_mean_value)
        region_value = max((region_mean_value - min_value) / (max_value - min_value), 0)
        region_values.append(region_value)
        country_value = country_df[colname].copy()
        
        if col_info["to_normalize"]:
            country_value /= country_df["Population"]
        
        country_value = country_value * factor
        country_mean_value = np.log(country_value.mean()) if logscale else np.nanmean(country_value)
        country_real_values.append(np.exp(country_mean_value) if logscale else country_mean_value)
        country_value = max((country_mean_value - min_value) / (max_value - min_value), 0)
        country_values.append(country_value)

        metrics.append(col_info["metric"])
        normalized.append(col_info["to_normalize"])
    
    region_values.append(region_values[0])
    country_values.append(country_values[0])
    region_real_values.append(region_real_values[0])
    country_real_values.append(country_real_values[0])
    metrics.append(metrics[0])
    normalized.append(normalized[0])
    
    categories.append(categories[0])
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=region_values,
        theta=categories,
        fill='tonext',
        name=region_option, 
        line_color='#319177', 
        hovertemplate=[f'{categ} {"per person" if norm else ""}: <br> {real_value:.2f} ({metric})' \
            for categ, norm, real_value, metric in zip(categories, normalized, region_real_values, metrics)]
    ))
    if country_to_display != "World":
        fig.add_trace(go.Scatterpolar(
            r=country_values,
            theta=categories,
            name=country_to_display, 
            line_color='#E2774E', 
            hovertemplate=[f'{categ} {"per person" if norm else ""}: <br> {real_value:.2f} ({metric})' \
                for categ, norm, real_value, metric in zip(categories, normalized, country_real_values, metrics)]
            
        ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        polar=dict(
            radialaxis=dict(
            visible=True,
            range=[0, 1]
            )),
        showlegend=True,
            legend=dict(yanchor="top", y=1.2, xanchor="left", x=0.8), 
         margin=go.layout.Margin(
        l=100, #left margin
        r=100, #right margin
        b=100, #bottom margin
        t=100  #top margin
        )
    )


    selected_points = plotly_events(fig, click_event=True, hover_event=False)
    try:
        key_metric_to_display = selected_points[0]["pointNumber"]
        metric_to_display = categories[key_metric_to_display]
        with open('metric_to_display.txt', 'w') as output:
            output.write(str(key_metric_to_display))
    except:
        with open('metric_to_display.txt', 'r') as f:
            key_metric_to_display = int(f.read())%4
        metric_to_display = col_pollutions[key_metric_to_display]["name"]

    st.write("You can select a pollution metric to display on the other charts by **double**-clicking on the corresponding metric above.")
    st.write(f"The currently selected metric is **{metric_to_display}**")

with col2:
    st.header("Evolution over time")


    def get_label(date, df):
        """ Gets the label of the first and last point in the chart """
        years = df[df[[col_pollutions[key_metric_to_display]["key"],
                    "average_value_Cereal production (metric tons)",
                    "Population"]].notnull().all(axis=1)]["Year"]
        try:            
            if date in [min(years), max(years)]:
                return date
            else:
                return ""
        except:
            return ""

    region_df["label"] = region_df["Year"].apply(lambda x: get_label(x, region_df))
    country_df["label"] = country_df["Year"].apply(lambda x: get_label(x, country_df))


    comparison_df = pd.concat([region_df, country_df])

    if col_pollutions[key_metric_to_display]["to_normalize"]:
        comparison_df["pollution"] = \
            comparison_df[col_pollutions[key_metric_to_display]["key"]] / \
            comparison_df["Population"]
        title = col_pollutions[key_metric_to_display]["name"] + " per person"
    else:
        comparison_df["pollution"] = \
            comparison_df[col_pollutions[key_metric_to_display]["key"]]
        title = col_pollutions[key_metric_to_display]["name"]
    comparison_df["cereal_prod_per_pop"] = comparison_df["average_value_Cereal production (metric tons)"] / \
                                        comparison_df["Population"]
    comparison_df = comparison_df[["pollution",
                                "cereal_prod_per_pop",
                                "Population",
                                "Year",
                                "Country Name",
                                "label"]]

    time_evolution = alt.Chart(comparison_df).mark_line(point=True).encode(
        alt.X('cereal_prod_per_pop', axis=alt.Axis(title="Cereal Production Per Person")),
        alt.Y('pollution', axis=alt.Axis(title=title)),
        order='Year',
        color=alt.Color("Country Name", scale=alt.Scale(scheme='dark2'), sort=[
        "East Asia & Pacific",
        "Europe & Central Asia",
        "Latin America & Caribbean",
        "Middle East & North Africa",
        "North America",
        "South Asia",
        "Sub-Saharan Africa ", "World"]), 
        tooltip=["Year", "Country Name", "cereal_prod_per_pop", "pollution"]
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

    col2.altair_chart((time_evolution + text).interactive(), use_container_width=True)
