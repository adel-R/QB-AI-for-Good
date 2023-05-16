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
import io
import matplotlib.pyplot as plt

# Layout
st.set_page_config(layout="wide")
margin = 0
padding = 2

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

        [data-testid=stSidebar] .block-container {{
            margin-top: 0rem;
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

        button [data-testid=stMarkdownContainer] .css-1offfwpp{{
          color:#053E57
        }}

        [data-testid=stMarkdownContainer] h2 span{{
                    color:#053E57;
                }}

        [data-testid=stSidebar] button [data-testid=stMarkdownContainer] {{
            color:#053E57;
        }}
        

        .css-1oe5cao{{
            padding-top: 2rem;
        }}

        .stCheckbox{{
            opacity:0;
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
    sites = st.multiselect('Sites', sorted(list(gdf['Site'].unique())))
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

    # Filter on the sites
    if sites !=[]:
        gdf_filtered = gdf_filtered[gdf_filtered['Site'].isin(sites)]

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

    # Filter on the sites
    if sites !=[]:
        gdf_filtered = gdf_filtered[gdf_filtered['Site'].isin(sites)]

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
    
    st.sidebar.image(parent_path +"/logo/logo.png",
                     width=150)

    ### Prediction from model 
    # Title and Side Bar for filters
    st.title("Sites Monitoring")

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
    st.header("Inspect Satellite Images")
    # Add New entry for prediction
    zip_file = st.file_uploader('Upload satelite images and their metadata to identify potential plumes:', 
                                type=None, accept_multiple_files=False, 
                                help='The zip file must contain no subfolders. The metadata must contain complete and accurate information.',
                                )

    # loading model
    model = inference.load_resnet34()

    # getting to correct device
    device, model = modeling.get_device(model)
    
    # Function to remove the folder and its contents
    def remove_folder(folder_path):
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)


    #Cache data
    @st.cache_data
    # Check if metadata is valid (and concatenate multiple files if any)
    def metadata(folder_path, required_columns=['date','id_coord','lat','lon','coord_x','coord_y','image_name']):    
        # Get a list of all CSV files in the folder
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        results = []
        # Check if at least one CSV file exists in the folder
        if csv_files:
            # Iterate over each CSV file
            for csv_file in csv_files:
                # Read the CSV file
                df = pd.read_csv(csv_file)
                
                # Check if the required columns are present in the CSV file
                if all(column in df.columns for column in required_columns):
                    results.append(df.copy())
            
            # Concatenate valid metadata files
            if len(results)>1:
                results_df = pd.concat(results, ignore_index=True)
                return results_df
            elif len(results)==1:
                return results[0]
            else:
                return False  
        else:
            return False



    #Cache data
    @st.cache_data
    def display_metadata(metadata_img,prob,lbl,lbl_h,img_raw, heat_map):
        zz,col4,col5,zw = st.columns([1,4,3,1])
        
        with col4:
            st.write('Predictions results for site ID: '+ str(metadata_img['id_coord'][0]))
            date_obj = datetime.datetime.strptime(str(metadata_img['date'][0]), '%Y%m%d')
            formatted_date = date_obj.strftime('%d/%m/%Y')
            st.write('Date taken: '+ formatted_date)
            st.write('Latitude: '+str(metadata_img['lat'][0]))
            st.write('Longitude: '+str(metadata_img['lon'][0]))
            # st.write('City: '+str(metadata_img['city'][0]))
            # st.write('Country: '+str(metadata_img['country'][0]))
            st.write('File name: '+str(metadata_img['image_name'][0]))
            st.write('Confidence: '+str(round(prob*100,2))+' %')
            if lbl>0:
                st.subheader(':warning: :red[A plume has been identified]')
            else:
                st.subheader(':heavy_check_mark: :green[No plume has been identified]')


        with col5:
            # gradcam_filename = parent_path+'/map/images/no_plume/20230305_methane_mixing_ratio_id_2384.tif'
            # gradcam_image = Image.open(gradcam_filename)
            # gradcam_image = gradcam_image.convert("RGB")
            # st.image(gradcam_image,width=300)
            fig = GradCam.visualize_heatmap(img_raw, heat_map, lbl_h)

            # Save the figure to a BytesIO object
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)

            # Display the saved image using st.image
            st.image(buffer, width=200)

        
        st.divider()
    
    #Cache data
    @st.cache_data
    def display_no_metadata(path_to_img,prob,lbl,img_raw, heat_map, lbl_h):
        zz,col4,col5,zw = st.columns([1,4,3,1])
        
        with col4:
            st.write('Predictions results for image: '+ str(os.path.basename(path_to_img)))
            st.write('Confidence: '+str(round(prob*100,2))+' %')
            if lbl>0:
                st.subheader(':warning: :red[A plume has been identified]')
            else:
                st.subheader(':heavy_check_mark: :green[No plume has been identified]')


        with col5:
            # gradcam_filename = parent_path+'/map/images/no_plume/20230305_methane_mixing_ratio_id_2384.tif'
            # gradcam_image = Image.open(gradcam_filename)
            # gradcam_image = gradcam_image.convert("RGB")
            # st.image(gradcam_image,width=300)
            fig = GradCam.visualize_heatmap(img_raw, heat_map, lbl_h)
            
            # Save the figure to a BytesIO object
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)

            # Display the saved image using st.image
            st.image(buffer, width=200)

        st.divider()

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
        
        # Fetch image list 
        image_list_paths = glob.glob(output_folder + "/*.tiff") + glob.glob(output_folder + "/*.tif")

        if len(image_list_paths)>0:
            if len(image_list_paths)>5:
                st.header('The first '+ str(min(len(image_list_paths),5)) +' results are displayed below')
            else:
                st.header('Results are displayed below')

            metadata_df = metadata(output_folder)

            # # Fetch country and city
            # if isinstance(metadata_df, bool)==False:
            #     # Use reverse_geocoder to get the city and country information
            #     coords = list(zip(metadata_df["lat"], metadata_df["lon"]))
            #     coord_results = rg.search(coords)

            #     # Extract city and country and add to the dataframe
            #     metadata_df["city"] = [r["name"] for r in coord_results]
            #     metadata_df["country_code"] = [r["cc"] for r in coord_results]

            #     # Convert country codes to country names
            #     def get_country_name(country_code):
            #         try:
            #             return pycountry.countries.get(alpha_2=country_code).name
            #         except AttributeError:
            #             return None
                    
            #     metadata_df["country"] = metadata_df["country_code"].apply(get_country_name)

            result_pred_csv = []
            for i in range(len(image_list_paths)):

                # Fetch an image path
                path_to_img = image_list_paths[i]

                # Prediction 
                prob, lbl = inference.infer(model=model,path_to_img=path_to_img,device=device)

                # GradCam
                heat_map, img_raw, lbl_h = GradCam.get_heatmap(model=model, path_to_img=path_to_img, device=device, against_label=None)

                # metadata valid format
                if isinstance(metadata_df, bool)==False:
                    # Fetch metadata
                    metadata_img = metadata_df[metadata_df['image_name']==os.path.basename(path_to_img)[:os.path.basename(path_to_img).find('.')]].fillna('-').copy()
                    metadata_img.reset_index(drop=True, inplace=True)
                    if metadata_img.shape[0]>0:
                        metadata_img['plume predicted'] = lbl
                        metadata_img['probability'] = prob
                        result_pred_csv.append(metadata_img.copy())
                        if i<5:
                            display_metadata(metadata_img,prob,lbl,lbl_h,img_raw, heat_map)
                    else:
                        pred_df = pd.DataFrame([[os.path.basename(path_to_img),lbl,prob]], columns=['image_name','plume predicted','probability'])
                        result_pred_csv.append(pred_df.copy())
                        if i<5:
                            display_no_metadata(path_to_img,prob,lbl,img_raw, heat_map, lbl_h)
                else:
                    pred_df = pd.DataFrame([[os.path.basename(path_to_img),lbl,prob]], columns=['image_name','plume predicted','probability'])
                    result_pred_csv.append(pred_df.copy())
                    if i <5:
                        display_no_metadata(path_to_img,prob,lbl,img_raw, heat_map, lbl_h)

            d,col5,col6,col7,c = st.columns([3,1,1,1,3])
            final_predictions = convert_df(pd.concat(result_pred_csv))
            with col7:
                st.download_button(
                    label="Download all results",
                    data=final_predictions,
                    file_name='predictions.csv',
                    mime='text/csv',
                )
            with col5:
                val_button = st.button('Validate Analysis')
            with col6:
                verif_button = st.button('Request Verification')

            # Check if the button is clicked
            if val_button:
                remove_folder(output_folder)
                val_button=None
                verif_button=None
                zip_file=None
            
            # Check if the button is clicked
            if verif_button:
                remove_folder(output_folder)
                val_button=None
                verif_button=None
                zip_file=None

        else:
            st.warning('Invalid image file format. The file must be images with .tif or .tiff extensions')
    else:
        remove_folder(os.path.dirname(base_path)+"/upload/")
        val_button=None
        verif_button=None
        zip_file=None

            