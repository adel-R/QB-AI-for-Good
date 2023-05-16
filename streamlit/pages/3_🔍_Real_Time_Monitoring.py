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
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from PIL import Image
import util.inference as inference
import util.modeling as modeling
import zipfile
import shutil
import glob
import util.GradCam as GradCam

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

        button [data-testid=stMarkdownContainer] p{{
          color:#053E57
        }}

        .st-ei {{
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
gdf = gpd.read_file(file_path)

# Add datetime
gdf['datetime'] =  pd.to_datetime(gdf['date'], format= "%Y%m%d")


with st.sidebar:
    st.header('Enter your filters:')
    plumes = st.selectbox('Display', ('All','Only Plumes'))
    period = st.date_input( "Period of Interest", (datetime.date(2023, 1, 1),datetime.date(2023, 12, 31) ))
    status = st.multiselect('Status', ['Ongoing','Closed','To Be Verified'])
    sectors = st.multiselect('Sectors', sorted(list(gdf['sector'].unique())))
    companies = st.multiselect('Companies', sorted(list(gdf['company'].unique())))
    countries = st.multiselect('Countries', sorted(list(gdf['country'].unique())))


#Apply filters
gdf_filtered = gdf.copy()

# Filter on the display
if plumes=='Only Plumes':
    gdf_filtered = gdf_filtered[gdf_filtered['plume']=='yes']

    # Filter on the status
    if status !=[]:
        gdf_filtered = gdf_filtered[gdf_filtered['Status'].isin(status)]

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
    # Filter on the status
    if status !=[]:
        gdf_filtered = gdf_filtered[gdf_filtered['Status'].isin(status)]

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
        file_name='large_df.csv',
        mime='text/csv',
    )
    st.sidebar.image("/Users/clarabesnard/Desktop/Desktop - MacBook Air de Clara (2)/DSBA 2/QB/QB-AI-for-Good/streamlit/pages/upload/logo.png",
                      width=150)

# Write dataframe to map
if gdf_filtered.shape[0]<1:
    st.header('No Result found for this query')
else:
    #Filter on the columns to be displayed
    gdf_filtered = gdf_filtered.rename(columns={'Concentrat':'Concentration Uncertainty (ppm m)',
                                                 'Max Plume':'Max Plume Concentration (ppm m)',
                                                 'Emission': 'Estimated Emission rate (CH4 tonnes/hour)',
                                                 'Duration':'Days since',
                                                 'Total' : 'Total Emissions (kt CH4)' ,
                                                 'CO2eq': 'Emissions (kt CO2eq)',
                                                 'Site' : 'Site ID',
                                                 'Credit' : 'Carbon Credit cost ($)' })

    display_columns = ['Site ID',
                        'city',
                        'country',
                        'company',
                        'plume',
                        'Status',
                        'Days since',
                        'Contact']



    # Filter on display columns
    gdf_filtered = gdf_filtered[display_columns]

    ### Prediction from model
    # Title and Side Bar for filters
    st.title("Real Time Monitoring")

    # Boolean to resize the dataframe, stored as a session state variable
    st.checkbox("Use container width", value=False, key="use_container_width")

    # Follow-up dataframe
    display_image = st.session_state.use_container_width
    original_filename = parent_path+'/map/images/plume/20230102_methane_mixing_ratio_id_1465.tif'
    original_image = Image.open(original_filename)
    original_image = original_image.convert("RGB")
    gradcam_filename = parent_path+'/map/images/no_plume/20230305_methane_mixing_ratio_id_2384.tif'
    gradcam_image = Image.open(gradcam_filename)
    gradcam_image = gradcam_image.convert("RGB")

    if display_image:
        # columns
        col1, col2 = st.columns([6,1])

        with col1:
            st.dataframe(pd.DataFrame(gdf_filtered),height = 530 , use_container_width=True)

        with col2:
            st.write('Most recent image')
            st.image(original_image,use_column_width=True)
            st.divider()
            st.write('Heatmap')
            st.image(gradcam_image,use_column_width=True)
    else:
        st.dataframe(pd.DataFrame(gdf_filtered),height = 530 , use_container_width=True)


    # Title and Side Bar for filters
    st.header("Inspect entries")
    # Add New entry for prediction
    zip_file = st.file_uploader('Upload satelite images and their metadata to identify potential plumes:',
                                type=None, accept_multiple_files=False,
                                help='The zip file must contain no subfolders. The metadata must contain complete and accurate information.',
                                )

    # loading model
    model = inference.load_resnet34()

    # getting to correct device
    device, model = modeling.get_device(model)

    # Check if file is tif file
    def is_tiff_file(file_path):
        file_extension = os.path.splitext(file_path)[1].lower()
        return file_extension == ".tif" or file_extension == ".tiff"

    # When zip file is loaded
    if zip_file !=None:
        # Create the output folder if it doesn't exist
        output_folder = os.path.dirname(base_path)+"/upload/"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Check if the uploaded file is a zip file
        if zipfile.is_zipfile(zip_file):
            # Open the zip file
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Extract all files to the output folder
                zip_ref.extractall(output_folder)
        else:
            # Save the loaded file in the subfolder
            zip_file_path = os.path.join(output_folder, zip_file.name)
            with open(zip_file_path, "wb") as file:
                file.write(zip_file.getbuffer())



            metadata_df = metadata(output_folder)

            # # Prediction
            # prob, lbl = inference.infer(model=model,path_to_img=path_to_img,device=device)
            # print(f"Probability of plume: {prob}")
            # print(f"Predicted label: {lbl}")


        # if len(unzipped)>5:
        #     st.header('The first '+ str(min(len(unzipped),5)) +' results are displayed below')
        # else:
        #     st.header('Results are displayed below')

        # for i in range(min(len(unzipped),5)):
        #     col3,col4 = st.columns(2)

        #     with col3:
        #         st.subheader('Predictions results for scene ID :'+unzipped)
        #         st.write('Scene ID: ')
        #         st.write('Date taken: 2222222')
        #         st.write('Latitude: 45')
        #         st.write('Longitude:30')
        #         st.write('Site ID:')
        #         st.write('City:')
        #         st.write('Country:')
        #         st.write('Original Image name: FOO.tif')
        #         st.write('Heatmap Image name: FOO_heatmap.tif')
        #         if i>0:
        #             st.subheader(':warning: :red[A plume has been identified]')
        #         else:
        #             st.subheader(':heavy_check_mark: :green[No plume has been identified]')

                # Fetch an image path
            path_to_img = image_list_paths[i]

        #     with col4:
        #         gradcam_filename = parent_path+'/map/images/no_plume/20230305_methane_mixing_ratio_id_2384.tif'
        #         gradcam_image = Image.open(gradcam_filename)
        #         gradcam_image = gradcam_image.convert("RGB")
        #         st.image(gradcam_image,width=300)
        #         st.caption('Heatmap of Scene ID XXX')


        #     st.divider()

        d,col5,col6,col7,c = st.columns([3,1,1,1,3])
        with col7:
            st.download_button(
                label="Download all results",
                data=csv,
                file_name='large_df.csv',
                mime='text/csv',
            )
        with col5:
            val_button = st.button('Validate Analysis')
        with col6:
            verif_button = st.button('Request Verification')

            # Function to remove the folder and its contents
            def remove_folder(folder_path):
                if os.path.exists(folder_path):
                    shutil.rmtree(folder_path)
                    st.success(f"Folder '{folder_path}' removed successfully.")
                else:
                    st.warning(f"Folder '{folder_path}' does not exist.")

        # Check if the button is clicked
        if val_button:
            remove_folder(output_folder)
            val_button=None
