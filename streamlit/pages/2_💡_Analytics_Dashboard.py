import pandas as pd
import os
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import reverse_geocoder as rg
import pycountry
import streamlit as st
from streamlit_folium import st_folium
import datetime
import branca

import matplotlib.pyplot as plt
import plotly.express as px  # interactive charts
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Layout
st.set_page_config(layout="wide")
margin = 0
padding = 2

# Layout
st.markdown(f"""
    <style>
        .block-container{{
            padding-top: 0rem;
            padding-bottom : 0rem;
            padding-left: {padding}rem;
            padding-right: {padding}rem;
            margin: {margin}rem;
        }}

        .css-1oe5cao{{
            padding-top: 2rem;
        }}
    </style>""",
    unsafe_allow_html=True,
)

# Get the base path of the Streamlit app
base_path = os.path.abspath(__file__)

#parent directory to get to the map folder
parent_path = os.path.dirname(os.path.dirname(base_path))

# Specify the relative path to the Shapefile within the subfolder
file_path = parent_path + "/map/map.shp"

# read map file
#gdf = gpd.read_file(
#    "/Users/clarabesnard/Desktop/Desktop - MacBook Air de Clara (2)/DSBA 2/QB/QB-AI-for-Good/streamlit/map/map.shp")

gdf = gpd.read_file(file_path)

# Add datetime
gdf['datetime'] =  pd.to_datetime(gdf['date'], format= "%Y%m%d")


# Title and Side Bar for filters
st.title("Analytics Dashboard")
with st.sidebar:
    st.header('Enter your filters:')
    plumes = st.selectbox('Display', ('All','Only Plumes'))
    period = st.date_input( "Period of Interest", (datetime.date(2023, 1, 1),datetime.date(2023, 12, 31) ))
    sectors = st.multiselect('Sectors', sorted(list(gdf['sector'].unique())))
    companies = st.multiselect('Companies', sorted(list(gdf['company'].unique())))
    countries = st.multiselect('Countries', sorted(list(gdf['country'].unique())))

#Apply filters
gdf_filtered = gdf.copy()

# Filter on the display
if plumes=='Only Plumes':
    gdf_filtered = gdf_filtered[gdf_filtered['plume']=='yes']

    # Filter on the sectors
    if sectors !=[]:
        gdf_filtered = gdf_filtered[gdf_filtered['sector'].isin(sectors)]

    # Filter on the companies
    if companies !=[]:
        gdf_filtered = gdf_filtered[gdf_filtered['company'].isin(companies)]

    # Filter on the countries
    if countries !=[]:
        gdf_filtered = gdf_filtered[gdf_filtered['country'].isin(countries)]

    # Filter on date
    if len(period)<2:
        gdf_filtered = gdf_filtered[(gdf_filtered["datetime"] == pd.Timestamp(period[0]))]
    else:
        gdf_filtered = gdf_filtered[(gdf_filtered["datetime"] >= pd.Timestamp(period[0])) & (gdf_filtered["datetime"] <= pd.Timestamp(period[1]))]
    gdf_filtered["datetime"] = gdf_filtered["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

else:
    # Filter on the sectors
    if sectors !=[]:
        gdf_filtered = gdf_filtered[gdf_filtered['sector'].isin(sectors)]

    # Filter on the companies
    if companies !=[]:
        gdf_filtered = gdf_filtered[gdf_filtered['company'].isin(companies)]

    # Filter on the countries
    if countries !=[]:
        gdf_filtered = gdf_filtered[gdf_filtered['country'].isin(countries)]

    # Filter on date
    if len(period)<2:
        gdf_filtered = gdf_filtered[(gdf_filtered["datetime"] == pd.Timestamp(period[0]))]
    else:
        gdf_filtered = gdf_filtered[(gdf_filtered["datetime"] >= pd.Timestamp(period[0])) & (gdf_filtered["datetime"] <= pd.Timestamp(period[1]))]
    gdf_filtered["datetime"] = gdf_filtered["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")


# Check if filter gives results
if gdf_filtered.shape[0]<1:
    st.header('No Result found for this query')
else:
    #Filter on the columns to be displayed
    gdf_filtered = gdf_filtered.rename(columns={'Concentrat':'Concentration Uncertainty (ppm m)',
                                                 'Max Plume':'Max Plume Concentration (ppm m)',
                                                 'Emission': 'Estimated Emission rate (CH4 tonnes/hour)',
                                                 'Duration':'Estimated Duration (hours)',
                                                 'Total' : 'Total Emissions (kt CH4)' ,
                                                 'CO2eq': 'Total Emissions (kt CO2eq)',
                                                 'Credit' : 'Carbon Credit cost ($)' })
    display_columns = ['id_coord',
                        'plume',
                        'city',
                        'country',
                        'company',
                        'sector',
                        'geometry',
                        'Concentration Uncertainty (ppm m)',
                        'Max Plume Concentration (ppm m)',
                        'datetime',
                        'Estimated Emission rate (CH4 tonnes/hour)',
                        'Estimated Duration (hours)',
                        'Total Emissions (kt CH4)',
                        'Total Emissions (kt CO2eq)',
                        'Carbon Credit cost ($)']

    # Filter on display columns
    gdf_filtered = gdf_filtered[display_columns]


####ANALYTICS DASHBOARD

# # Time tabs
# col1, col2, col3, col4 = st.columns(4)

# view = "year"


# with col1:
#   if st.button('Yearly'):
#     view = "year"
# with col2:
#   if st.button('Quarterly'):
#     view = "quarter"
# with col3:
#   if st.button('Monthly'):
#     view = "month"
# with col4:
#   if st.button('Weekly'):
#     view = "week"

### Trend


####
# Plot the concentration of methane (Concentrat) or emission levels (Emission) over time (date or datetime) to visualize trends, seasonal variations, or any significant changes.
###
#print(gdf_filtered.dtypes)
# Deal with NAs
temp = gdf_filtered
temp['Total Emissions (kt CH4)'] = temp['Total Emissions (kt CH4)'].fillna(0)
#sorted_df = temp.sort_values('datetime')
grouped_df = temp.groupby('datetime')[
    'Total Emissions (kt CH4)'].sum().reset_index()

#print(gdf_filtered.columns)
# Create a line chart showing the evolution of methane concentration over time
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=grouped_df['datetime'], y=grouped_df['Total Emissions (kt CH4)'], mode='lines'))
fig.update_layout(title='Evolution of Total Emissions (kt CH4)',
                  xaxis_title='Date',
                  yaxis_title='Emissions (kt CH4)')

# Display the chart
#fig.show()

st.plotly_chart(fig, use_container_width=True)



# by country and sector
col1, col2 = st.columns(2)

## by country
grouped_df = temp.groupby('country')[
    'Total Emissions (kt CH4)'].sum().reset_index()

grouped_df = grouped_df.sort_values('Total Emissions (kt CH4)', ascending=False)

fig = go.Figure(
    [go.Bar(x=grouped_df['country'], y=grouped_df['Total Emissions (kt CH4)'])])

fig.update_layout(title='Total Emissions (kt CH4) per Country',
                  xaxis_title='Country',
                  yaxis_title='Emissions (kt CH4)')

fig.update_xaxes(tickangle=45)


with col1:
  st.plotly_chart(fig, use_container_width=True)


## by sector
grouped_df = temp.groupby('sector')[
    'Total Emissions (kt CH4)'].sum().reset_index()

grouped_df = grouped_df.sort_values('Total Emissions (kt CH4)', ascending=False)

fig = go.Figure(
    [go.Bar(x=grouped_df['sector'], y=grouped_df['Total Emissions (kt CH4)'])])

fig.update_layout(title='Total Emissions (kt CH4) per Sector',
                  xaxis_title='Sector',
                  yaxis_title='Emissions (kt CH4)')


with col2:
  st.plotly_chart(fig, use_container_width=True)



# Company View
col1, col2 = st.columns(2)

# Display top 10 companies polluting
grouped_df = temp.groupby('company')[
    'Total Emissions (kt CH4)'].sum().reset_index()
grouped_df = grouped_df.sort_values(
    'Total Emissions (kt CH4)', ascending=True)

grouped_df = grouped_df.tail(10)

fig = go.Figure(
    [go.Bar(x=grouped_df['Total Emissions (kt CH4)'], y=grouped_df['company'], orientation='h')])

fig.update_layout(title='Top 10 Companies in producing emissions (kt CH4)',
                  yaxis_title='Companies',
                  xaxis_title='Emissions (kt CH4)')

with col1:
  st.plotly_chart(fig, use_container_width=True)


# break down of companies per sector
grouped_df = temp.groupby(['country','sector'])[
    'Total Emissions (kt CH4)'].sum().reset_index()

grouped_df = grouped_df.sort_values(
    'Total Emissions (kt CH4)', ascending=False)

# Create a list of unique sectors and companies for the x-axis and y-axis labels
sectors = grouped_df['sector'].unique()
countries = grouped_df['country'].unique()

# Create an empty dictionary to store the emissions data for each company within each sector
emissions_data = {}
for sector in sectors:
    emissions_data[sector] = grouped_df[grouped_df['sector']
                                        == sector]['Total Emissions (kt CH4)'].tolist()

fig = go.Figure()
for i, sector in enumerate(sectors):
    fig.add_trace(go.Bar(
        x=countries,
        y=emissions_data[sector],
        name=sector))

fig.update_layout(title='Break down of polluting sectors per country',
                  xaxis_title='Countries',
                  yaxis_title='Emissions (kt CH4)',
                  barmode='stack')

fig.update_xaxes(tickangle=45)

with col2:
  st.plotly_chart(fig, use_container_width=True)
