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

# read map file
gdf = gpd.read_file("map/map.shp") 

# Add datetime
gdf['datetime'] =  pd.to_datetime(gdf['date'], format= "%Y%m%d")


# Title and Side Bar for filters
st.title("Monitor methane plumes")
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

#Cache data
@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

# Download data as csv
csv = convert_df(pd.DataFrame(gdf_filtered))
with st.sidebar:
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='monitoring_export.csv',
        mime='text/csv',
    )

# Write dataframe to map
if gdf_filtered.shape[0]<1:
    st.header('No Result found for this query')
else:
    #Filter on the columns to be displayed
    gdf_filtered = gdf_filtered.rename(columns={'Concentrat':'Concentration Uncertainty (ppm m) ',
                                                 'Max Plume':'Max Plume Concentration (ppm m) ',
                                                 'Emission': 'Estimated Emission rate (CH4 tonnes/hour) ',
                                                 'Duration':'Estimated Duration (hours) ',
                                                 'Total' : 'Total Emissions (kt CH4) ' ,
                                                 'CO2eq': 'Total Emissions (kt CO2eq)' })
    display_columns = ['id_coord',
                        'plume',
                        'city', 
                        'country',
                        'company',
                        'sector',
                        'geometry',
                        'Concentration Uncertainty (ppm m) ',
                        'Max Plume Concentration (ppm m) ',
                        'datetime',
                        'Estimated Emission rate (CH4 tonnes/hour) ',
                        'Estimated Duration (hours) ',
                        'Total Emissions (kt CH4) ',
                        'Total Emissions (kt CO2eq)']
    gdf_map = gdf_filtered[display_columns]
    
    map = gdf_map.explore("plume", location=(29.63, 80),tiles = "CartoDB positron", cmap = "RdYlGn_r",zoom_start=2)

    # Legend
    legend_html = '''
    {% macro html(this, kwargs) %}
    <div style="
        position: fixed; 
        bottom: 7%;
        right: 2%;
        width: 100px;
        height: 35px; 
        z-index:9998;
        font-size:80%;
        background-color: #ffffff;
        opacity: 0.7;
        ">

        <p style='margin:auto;display:flex;flex-direction: column;align-items: center;justify-content: center;'>
            <div style='margin-left:15px'><span style="color:#a30021">&#9679;</span>&emsp;plume</div>
            <div style='margin-left:15px'><span style="color:#006837">&#9679;</span>&emsp;no plume</div>
        </p>

    </div>
    {% endmacro %}
    '''
    legend = branca.element.MacroElement()
    legend._template = branca.element.Template(legend_html)

    map.get_root().add_child(legend)

    #folium_map
    folium_map = st_folium(map, width=1500, height=600)