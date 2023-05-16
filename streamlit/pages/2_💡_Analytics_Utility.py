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
import streamlit.components.v1 as components


import matplotlib.pyplot as plt
import plotly.express as px  # interactive charts
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Layout
st.set_page_config(layout="wide")
margin = 0
padding = 2

graph_color = '#053E57'
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

        [data-testid=stDecoration] {{
            background-image: linear-gradient(90deg, #053E57, #FFFFFF);
        }}

        [data-testid=stSidebarNav] .css-wjbhl0 {{
            padding-top: 2rem;
        }}

        [data-testid=stSidebar] {{
            background-color: #053E57;
            color:#FFFFFF;
        }}

        [data-testid=stSidebar] .block-container {{
            margin-top: 0rem;
        }}

        [data-testid=stMarkdownContainer] h2{{
            color:#FFFFFF;
        }}

        [data-testid=stSidebar] [data-testid=stMarkdownContainer] {{
            color:#FFFFFF;
        }}

        [data-testid=stSidebar] [data-testid=stImage] {{
            text-align: center;
            padding-top: 2rem;
            display: block;
            margin-left: auto;
            margin-right: auto;
        }}

        [data-testid=stSidebarNav] a span {{
            color:#FFFFFF;
        }}

        [data-testid=stMarkdownContainer] h1{{
            color:#053E57;
        }}

        [data-testid=metric-container] {{
            color:#053E57;
        }}

        [data-testid=stMarkdownContainer] {{
            color:#053E57;
        }}

        [data-baseweb=tab-highlight] {{
            background-color:#053E57;
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

#st.sidebar.image("/Users/clarabesnard/Desktop/Desktop - MacBook Air de Clara (2)/DSBA 2/QB/QB-AI-for-Good/streamlit/logo.png",
#                 use_column_width=True)
# Title and Side Bar for filters
st.title("CrystalPeak Oil & Gas Methane Emission Dashboard")
with st.sidebar:
    st.header('Enter your filters:')
    plumes = st.selectbox('Display', ('All','Only Plumes'))
    period = st.date_input( "Period of Interest", (datetime.date(2023, 1, 1),datetime.date(2023, 12, 31) ))
    sites = st.multiselect('Site', sorted(
        list(gdf[gdf['company'] == 'CrystalPeak Oil & Gas']['Site'].unique())))
    countries = st.multiselect('Countries', sorted(list(
        gdf[gdf['company'] == 'CrystalPeak Oil & Gas']['country'].unique())))

#Apply filters
gdf_filtered = gdf.copy()

# Filter on the display
if plumes=='Only Plumes':
    gdf_filtered = gdf_filtered[gdf_filtered['plume']=='yes']

    # Filter on the companies
    if sites !=[]:
        gdf_filtered = gdf_filtered[gdf_filtered['Site'].isin(sites)]

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


    # Filter on the companies
    if sites !=[]:
        gdf_filtered = gdf_filtered[gdf_filtered['Site'].isin(sites)]

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
                        'Carbon Credit cost ($)',
                        'Site',
                        'Status'
                        ]

    # Filter on display columns
    gdf_filtered = gdf_filtered[display_columns]

    st.sidebar.image("/Users/clarabesnard/Desktop/Desktop - MacBook Air de Clara (2)/DSBA 2/QB/QB-AI-for-Good/streamlit/pages/upload/logo.png",
                     width=150)

####ANALYTICS DASHBOARD
    sector_gdf = gdf_filtered[gdf_filtered['sector']
                              == 'Landfills']
    avg_emission = sector_gdf['Total Emissions (kt CH4)'].mean()
    sum_emission = sector_gdf['Total Emissions (kt CH4)'].sum()
    gdf_filtered = gdf_filtered[gdf_filtered['company'] == 'CrystalPeak Oil & Gas']

    ## Display summary data
    col1, col2, col3, col4 = st.columns(4)
    # total sites
    total_site = gdf_filtered['Site'].nunique()
    col1.metric("Total Sites", total_site)

    st.write(
        """
    <style>
    [data-testid="stMetricDelta"] svg {
        display: none;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # open sites
    yes_site = gdf_filtered.loc[gdf_filtered['plume'] =='yes']['Site'].nunique()
    no_site = gdf_filtered.loc[gdf_filtered['plume'] == 'no']['Site'].nunique()
    col2.metric("Sites with leaks", str(yes_site), str(
        round(yes_site/total_site*100, 1))+'% of Sites', delta_color="inverse")  # intitulé à revoir
    total_emission = gdf_filtered['Total Emissions (kt CH4)'].sum()
    col3.metric("Total emission over period",
                str(round(total_emission,1)) + ' kt CH4', str(round(total_emission/sum_emission*100,1))+'% of Sector', delta_color='off')
    avg_days = gdf_filtered['Estimated Duration (hours)'].mean()
    col4.metric("Average leak duration", str(round(avg_days,1)) + 'h')
### Trend


####
# Plot the concentration of methane (Concentrat) or emission levels (Emission) over time (date or datetime) to visualize trends, seasonal variations, or any significant changes.
###


col1, col2 = st.columns(2)
#print(gdf_filtered.dtypes)
# Deal with NAs
tab1, tab2, tab3, tab4 = st.tabs(["Day", "Week", "Month", "Year"])

temp = gdf_filtered
temp['datetime'] = pd.to_datetime(temp['datetime'])
temp['year'] = temp['datetime'].dt.year
temp['month'] = temp['datetime'].dt.strftime('%m-%Y')
temp['week'] = temp['datetime'].dt.strftime('%W-%Y')
temp['week'] = 'W' + temp['week'].astype(str)
temp['day'] = temp['datetime'].dt.strftime('%d-%m-%Y')
temp['Total Emissions (kt CH4)'] = temp['Total Emissions (kt CH4)'].fillna(0)
#sorted_df = temp.sort_values('datetime')


######### YEAR #########
grouped_df = temp.groupby('year')[
    'Total Emissions (kt CH4)'].sum().reset_index()
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=grouped_df['year'], y=grouped_df['Total Emissions (kt CH4)'], mode='markers+lines', marker_color=graph_color))
fig1.update_layout(title='Evolution of Total Emissions (kt CH4)',
                  xaxis_title='Year',
                  yaxis_title='Emissions (kt CH4)')
fig1.update_xaxes(dtick= 1)


grouped_df = temp[temp['plume'] == 'yes'].groupby('year')['Site'].nunique()
fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=grouped_df.index, y=grouped_df.values, marker_color=graph_color))
fig2.update_layout(title='Evolution of Total Leaks',
                  xaxis_title='Year',
                  yaxis_title='#Leaks')
fig2.update_xaxes(dtick= 1)

with tab4:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)


######### MONTH #########
temp.sort_values('datetime', ascending=True)
grouped_df = temp.groupby('month')[
    'Total Emissions (kt CH4)'].sum().reset_index()
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=grouped_df['month'], y=grouped_df['Total Emissions (kt CH4)'], mode='markers+lines', marker_color=graph_color))
fig1.update_layout(title='Evolution of Total Emissions (kt CH4)',
                  xaxis_title='Month',
                  yaxis_title='Emissions (kt CH4)')
fig1.update_xaxes(dtick=1)

grouped_df = temp[temp['plume'] == 'yes'].groupby('month')['Site'].nunique()
fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=grouped_df.index, y=grouped_df.values, marker_color=graph_color))
fig2.update_layout(title='Evolution of Total Leaks',
                   xaxis_title='Month',
                   yaxis_title='#Leaks')
fig2.update_xaxes(dtick=1)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)


grouped_df = temp.groupby('week')[
    'Total Emissions (kt CH4)'].sum().reset_index()
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=grouped_df['week'], y=grouped_df['Total Emissions (kt CH4)'], mode='markers+lines', marker_color=graph_color))
fig1.update_layout(title='Evolution of Total Emissions (kt CH4)',
                  xaxis_title='Week',
                  yaxis_title='Emissions (kt CH4)')

grouped_df = temp[temp['plume'] == 'yes'].groupby('week')['Site'].nunique()
fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=grouped_df.index, y=grouped_df.values, marker_color=graph_color))
fig2.update_layout(title='Evolution of Total Leaks',
                   xaxis_title='Week',
                   yaxis_title='#Leaks')
fig2.update_xaxes(dtick=1)
fig2.update_xaxes(tickangle=45)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)


grouped_df = temp.groupby('datetime')[
    'Total Emissions (kt CH4)'].sum().reset_index()
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=grouped_df['datetime'], y=grouped_df['Total Emissions (kt CH4)'], mode='markers+lines', marker_color=graph_color))
fig1.update_layout(title='Evolution of Total Emissions (kt CH4)',
                  xaxis_title='Date',
                  yaxis_title='Emissions (kt CH4)')

grouped_df = temp[temp['plume'] == 'yes'].groupby(
    'datetime')['Site'].nunique()
grouped_df.sort_index()
fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=grouped_df.index, y=grouped_df.values, marker_color=graph_color))
fig2.update_layout(title='Evolution of Total Leaks',
                   xaxis_title='Day',
                   yaxis_title='#Leaks')


with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)


st.markdown("""---""")


# by country and site
col1, col2 = st.columns(2)

## by country
grouped_df = temp.groupby('country')[
    'Total Emissions (kt CH4)'].sum().reset_index()

grouped_df = grouped_df.sort_values('Total Emissions (kt CH4)', ascending=False)

fig = go.Figure(
    [go.Bar(x=grouped_df['country'], y=grouped_df['Total Emissions (kt CH4)'], marker_color=graph_color)])

fig.update_layout(title='Total Emissions (kt CH4) per Country',
                  xaxis_title='Country',
                  yaxis_title='Emissions (kt CH4)')

fig.update_xaxes(tickangle=45)


with col1:
  st.plotly_chart(fig, use_container_width=True)





## by site


temp['Site'] = temp['Site'].astype(str)
grouped_df = temp.groupby('Site')[
    'Total Emissions (kt CH4)'].sum().reset_index()

grouped_df = grouped_df.sort_values('Total Emissions (kt CH4)', ascending=False)

fig = go.Figure(
    [go.Bar(x=grouped_df['Site'], y=grouped_df['Total Emissions (kt CH4)'], marker_color=graph_color)])

fig.update_layout(title='Total Emissions (kt CH4) per Site',
                  xaxis_title='Site',
                  yaxis_title='Emissions (kt CH4)')
fig.update_xaxes(tickangle=45)
fig.update_layout(xaxis={'type': 'category'})

fig.add_shape(
    type='line',
    x0=0, x1=len(grouped_df['Site']),
    y0=avg_emission, y1=avg_emission,
    line=dict(color='#f06292', width=2, dash='dash')
)
fig.add_annotation(
    x=len(grouped_df['Site'])-2,
    y=avg_emission + 0.05 * max(grouped_df['Total Emissions (kt CH4)']),
    text='Sector average',
    showarrow=False,
    font=dict(color='#f06292')
)



with col2:
  st.plotly_chart(fig, use_container_width=True)





