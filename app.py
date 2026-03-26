import streamlit as st
import geopandas as gpd
import pydeck as pdk
import os
import openai
from dotenv import load_dotenv
import json
import pandas as pd
import ee
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Mosquito Data Visualization", layout="wide")

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("MODEL")
base_url = os.getenv("OPENAI_BASE_URL")

st.title("Global Mosquito Data Explorer")

# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    if base_url:
        return openai.OpenAI(api_key=api_key, base_url=base_url)
    else:
        return openai.OpenAI(api_key=api_key)

client = get_openai_client()

# Load GeoPackage
@st.cache_data
def load_data():
    gdf = gpd.read_file("globe_mosquito.gpkg")
    return gdf

gdf = load_data()

# Initialize Earth Engine
@st.cache_resource
def init_earth_engine():
    """Initialize Earth Engine with error handling"""
    # Check if user has set a project in environment variable
    project = os.environ.get('EARTHENGINE_PROJECT', None)

    try:
        if project:
            ee.Initialize(project=project)
            return True, None
        else:
            # Try without project
            ee.Initialize()
            return True, None
    except Exception as e:
        error_msg = str(e)
        if "no project found" in error_msg.lower():
            return False, "no_project"
        else:
            return False, error_msg

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["Data Explorer", "Interactive Map", "AI Insights"])

with tab1:
    st.write("### Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(gdf))
    with col2:
        st.metric("Countries", gdf['CountryName'].nunique())
    with col3:
        st.metric("Unique Species", gdf['Species'].nunique())
    with col4:
        st.metric("Organizations", gdf['OrganizationName'].nunique())

    st.write("### Dataset Preview")
    # Display dataframe without geometry column to avoid Streamlit warnings
    display_df = gdf.head(20).drop(columns=['geometry'])
    st.dataframe(display_df)

with tab2:
    # Check Earth Engine initialization
    ee_ready, ee_error = init_earth_engine()

    if not ee_ready:
        st.error("**Earth Engine not initialized**")

        if ee_error == "no_project":
            st.warning("**Google Cloud Project Required**")
            st.markdown("""
            Earth Engine now requires a Google Cloud project. Follow these steps:

            **Option 1: Set up a Cloud Project (Recommended)**
            1. Go to: https://console.cloud.google.com/
            2. Create a new project (or use existing one)
            3. Enable Earth Engine API for your project
            4. Set the project in terminal:
            ```bash
            source venv/bin/activate
            export EARTHENGINE_PROJECT='your-project-id'
            streamlit run app.py
            ```

            **Option 2: Set default project**
            ```bash
            source venv/bin/activate
            earthengine set_project your-project-id
            ```

            **Find your project ID at:** https://console.cloud.google.com/home/dashboard
            """)
        else:
            st.info(f"Error: {ee_error}\n\nRun authentication:\n```bash\nsource venv/bin/activate\nearthengine authenticate\n```")
        st.stop()

    st.title("Mosquito Habitat & Environmental Correlation Map")
    st.caption("Visualizing how environmental factors correlate with mosquito populations using Google Earth Engine")

    # ============================================================================
    # EARTH ENGINE LAYER FUNCTIONS
    # ============================================================================

    @st.cache_resource
    def get_ee_layers():
        """Get Earth Engine environmental layers"""
        return {
            'elevation': ee.Image('USGS/SRTMGL1_003').select('elevation'),
            'temperature': ee.ImageCollection('MODIS/061/MOD11A1') \
                .filterDate('2023-01-01', '2024-01-01') \
                .select('LST_Day_1km') \
                .mean(),
            'precipitation': ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
                .filterDate('2023-01-01', '2024-01-01') \
                .select('precipitation') \
                .sum(),
            'ndvi': ee.ImageCollection('MODIS/061/MOD13A2') \
                .filterDate('2023-01-01', '2024-01-01') \
                .select('NDVI') \
                .mean(),
            'landcover': ee.ImageCollection('MODIS/061/MCD12Q1') \
                .filterDate('2023-01-01', '2024-01-01') \
                .first() \
                .select('LC_Type1')
        }

    @st.cache_resource
    def get_vis_params():
        """Get visualization parameters for Earth Engine layers"""
        return {
            'elevation': {
                'min': 0,
                'max': 3000,
                'palette': ['blue', 'green', 'yellow', 'orange', 'red']
            },
            'temperature': {
                'min': 13000,
                'max': 16500,
                'palette': ['blue', 'cyan', 'yellow', 'orange', 'red']
            },
            'precipitation': {
                'min': 0,
                'max': 2000,
                'palette': ['white', 'lightblue', 'blue', 'darkblue', 'purple']
            },
            'ndvi': {
                'min': 0,
                'max': 9000,
                'palette': ['brown', 'yellow', 'lightgreen', 'green', 'darkgreen']
            },
            'landcover': {
                'min': 1,
                'max': 17,
                'palette': [
                    '05450a', '086a10', '54a708', '78d203', '009900',
                    'c6b044', 'dcd159', 'dade48', 'fbff13', 'b6ff05',
                    '27ff87', 'c24f44', 'a5a5a5', 'ff6d4c', '69fff8',
                    'f9ffa4', '1c0dff'
                ]
            }
        }

    def get_env_data_for_point(lat, lon):
        """Extract environmental data from Earth Engine for a point"""
        point = ee.Geometry.Point([lon, lat])

        # Elevation
        elevation = ee.Image('USGS/SRTMGL1_003').select('elevation')
        elev = elevation.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=30
        ).getInfo().get('elevation')

        # Temperature
        temp = ee.ImageCollection('MODIS/061/MOD11A1') \
            .filterDate('2023-01-01', '2024-01-01') \
            .select('LST_Day_1km') \
            .mean()
        temp_val = temp.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=1000
        ).getInfo().get('LST_Day_1km')
        temp_celsius = (temp_val * 0.02 - 273.15) if temp_val else None

        # Precipitation
        precip = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
            .filterDate('2023-01-01', '2024-01-01') \
            .select('precipitation') \
            .sum()
        precip_val = precip.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=5000
        ).getInfo().get('precipitation')

        # NDVI
        ndvi = ee.ImageCollection('MODIS/061/MOD13A2') \
            .filterDate('2023-01-01', '2024-01-01') \
            .select('NDVI') \
            .mean()
        ndvi_val = ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=1000
        ).getInfo().get('NDVI')
        ndvi_normalized = (ndvi_val * 0.0001) if ndvi_val else None

        return {
            'elevation': elev,
            'temperature': temp_celsius,
            'precipitation': precip_val,
            'ndvi': ndvi_normalized
        }

    # ============================================================================
    # LOAD DATA
    # ============================================================================

    @st.cache_data
    def load_mosquito_data():
        """Load mosquito trapping data from GeoPackage"""
        gdf = gpd.read_file("globe_mosquito.gpkg")
        gdf = gdf.to_crs(epsg=4326)
        gdf["lon"] = gdf.geometry.x
        gdf["lat"] = gdf.geometry.y
        return gdf

    @st.cache_data
    def load_landcover_data():
        """Load land cover observation data from GeoPackage"""
        gdf = gpd.read_file("globe_land_cover.gpkg")
        gdf = gdf.to_crs(epsg=4326)
        gdf["lon"] = gdf.geometry.x
        gdf["lat"] = gdf.geometry.y
        return gdf

    mosquito_gdf = load_mosquito_data()
    landcover_gdf = load_landcover_data()

    # ============================================================================
    # SIDEBAR CONTROLS
    # ============================================================================

    st.sidebar.header(" Map Controls")

    # Data layers
    st.sidebar.subheader("Data Points")
    show_mosquito = st.sidebar.checkbox(" Mosquito Locations", value=True)
    show_landcover = st.sidebar.checkbox(" Land Cover Sites", value=True)

    # Environmental layer selection
    st.sidebar.subheader("Environmental Layer")
    env_layer = st.sidebar.radio(
        "Select Earth Engine Layer",
        ["None", "Elevation", "Temperature", "Precipitation", "NDVI (Vegetation)", "Land Cover"],
        index=1
    )

    if env_layer != "None":
        opacity = st.sidebar.slider("Layer Opacity", 0.0, 1.0, 0.6, 0.1)

    # Map performance settings
    st.sidebar.subheader("Map Settings")
    max_markers = st.sidebar.slider(
        "Max markers to display",
        min_value=100,
        max_value=5000,
        value=500,
        step=100,
        help="Limit number of markers for faster loading"
    )

    # Sample size for correlation display
    st.sidebar.subheader("Correlation Analysis")
    sample_size = st.sidebar.slider(
        "Number of sites to analyze",
        min_value=10,
        max_value=min(200, len(mosquito_gdf)),
        value=50,
        step=10,
        help="Shows environmental data in popup for sampled mosquito sites"
    )

    show_correlation = st.sidebar.checkbox(
        "Show environmental data in popups",
        value=False,
        help="Extracts Earth Engine data for each point (SLOW - only use with small sample sizes)"
    )

    # ============================================================================
    # STATS
    # ============================================================================

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(" Mosquito Sites", f"{len(mosquito_gdf):,}")
    with col2:
        st.metric(" Land Cover Sites", f"{len(landcover_gdf):,}")
    with col3:
        if env_layer != "None":
            st.metric(" EE Layer", env_layer)

    # ============================================================================
    # CREATE MAP
    # ============================================================================

    st.write("###  Interactive Correlation Map")

    # Show loading info
    if show_mosquito or show_landcover:
        total_mosquito = len(mosquito_gdf) if show_mosquito else 0
        total_landcover = len(landcover_gdf) if show_landcover else 0
        displaying = min(max_markers, total_mosquito + total_landcover)
        st.info(f"Displaying up to {displaying:,} markers (Total available: {total_mosquito:,} mosquito + {total_landcover:,} land cover)")

    # Calculate center
    all_lats = []
    all_lons = []
    if show_mosquito:
        all_lats.extend(mosquito_gdf["lat"].tolist())
        all_lons.extend(mosquito_gdf["lon"].tolist())
    if show_landcover:
        all_lats.extend(landcover_gdf["lat"].tolist())
        all_lons.extend(landcover_gdf["lon"].tolist())

    center_lat = sum(all_lats) / len(all_lats) if all_lats else 0
    center_lon = sum(all_lons) / len(all_lons) if all_lons else 0

    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=3,
        tiles='OpenStreetMap'
    )

    # Add Earth Engine layer
    if env_layer != "None":
        try:
            layer_mapping = {
                "Elevation": "elevation",
                "Temperature": "temperature",
                "Precipitation": "precipitation",
                "NDVI (Vegetation)": "ndvi",
                "Land Cover": "landcover"
            }

            layer_key = layer_mapping[env_layer]
            ee_layers = get_ee_layers()
            vis_params = get_vis_params()

            ee_image = ee_layers[layer_key]
            vis_param = vis_params[layer_key]

            # Get map tiles from Earth Engine
            map_id_dict = ee_image.getMapId(vis_param)

            folium.TileLayer(
                tiles=map_id_dict['tile_fetcher'].url_format,
                attr='Google Earth Engine',
                name=env_layer,
                overlay=True,
                control=True,
                opacity=opacity
            ).add_to(m)

        except Exception as e:
            st.error(f"Error loading Earth Engine layer: {e}")

    # Add data points (with performance limits)
    if show_landcover:
        # Limit markers based on user setting
        landcover_limit = max_markers // 2  # Split between mosquito and land cover
        landcover_sample = landcover_gdf.head(landcover_limit)

        for idx, row in landcover_sample.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=4,
                color='green',
                fill=True,
                fillColor='green',
                fillOpacity=0.6,
                popup=folium.Popup(
                    f"""
                    <b>Land Cover Site</b><br>
                    <b>Country:</b> {row.get('CountryName', 'N/A')}<br>
                    <b>Date:</b> {row.get('MeasuredDate', 'N/A')}<br>
                    <b>Description:</b> {row.get('MucDescription', 'N/A')}<br>
                    <b>Coordinates:</b> {row['lat']:.4f}, {row['lon']:.4f}
                    """,
                    max_width=300
                )
            ).add_to(m)

    if show_mosquito:
        # Sample mosquito data based on settings
        if show_correlation:
            mosquito_sample = mosquito_gdf.head(sample_size)
        else:
            mosquito_limit = max_markers // 2  # Split between mosquito and land cover
            mosquito_sample = mosquito_gdf.head(mosquito_limit)

        for idx, row in mosquito_sample.iterrows():
            lat, lon = row['lat'], row['lon']

            # Base popup info
            popup_html = f"""
            <b>Mosquito Trap Site</b><br>
            <b>Country:</b> {row.get('CountryName', 'N/A')}<br>
            <b>Date:</b> {row.get('MeasuredDate', 'N/A')}<br>
            <b>Water Source:</b> {row.get('WaterSourceType', 'N/A')}<br>
            <b>Larvae Count:</b> {row.get('LarvaeCount', 'N/A')}<br>
            <b>Coordinates:</b> {lat:.4f}, {lon:.4f}
            """

            # Add environmental data if correlation mode is on
            if show_correlation:
                try:
                    env_data = get_env_data_for_point(lat, lon)
                    popup_html += f"""
                    <br><hr><b> Environmental Data (2023):</b><br>
                    <b>Elevation:</b> {env_data['elevation']:.0f if env_data['elevation'] else 'N/A'} m<br>
                    <b>Temperature:</b> {env_data['temperature']:.1f if env_data['temperature'] else 'N/A'} C<br>
                    <b>Precipitation:</b> {env_data['precipitation']:.0f if env_data['precipitation'] else 'N/A'} mm/year<br>
                    <b>NDVI:</b> {env_data['ndvi']:.3f if env_data['ndvi'] else 'N/A'}
                    """
                except Exception as e:
                    popup_html += f"<br><i>Error fetching env data</i>"

            # Determine marker color based on larvae count
            larvae_count = row.get('LarvaeCount', 0)

            # Handle None or non-numeric values
            try:
                larvae_count = float(larvae_count) if larvae_count is not None else 0
            except (ValueError, TypeError):
                larvae_count = 0

            if larvae_count == 0:
                color = 'orange'
            elif larvae_count < 10:
                color = 'yellow'
            elif larvae_count < 50:
                color = 'red'
            else:
                color = 'darkred'

            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                popup=folium.Popup(popup_html, max_width=350)
            ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Display map
    with st.spinner("Rendering map... This may take a moment with many markers."):
        st_folium(m, width=1400, height=700, key="main_map")

    # Legend
    st.write("---")
    col1, col2 = st.columns(2)

    with col1:
        st.write("** Map Legend**")
        st.write(" Green = Land Cover Sites")
        st.write(" Yellow/Orange = Low mosquito larvae")
        st.write(" Red = Moderate mosquito larvae")
        st.write(" Dark Red = High mosquito larvae")

    with col2:
        st.write("**Environmental Layers**")
        if env_layer == "Elevation":
            st.write("Blue (low) to Red (high elevation)")
        elif env_layer == "Temperature":
            st.write("Blue (cold) to Red (hot)")
        elif env_layer == "Precipitation":
            st.write("White (dry) to Purple (wet)")
        elif env_layer == "NDVI (Vegetation)":
            st.write("Brown (no vegetation) to Green (dense vegetation)")
        elif env_layer == "Land Cover":
            st.write("Different colors = Different land cover types")

    # Info box
    st.info("""
    **How to use:**
    1. Toggle mosquito/land cover sites on/off in the sidebar
    2. Select an environmental layer to see how it correlates with mosquito habitats
    3. Enable 'Show environmental data in popups' to see exact values for each site
    4. Click on map markers to see detailed information
    5. Adjust opacity to better see the correlation between layers
    """)

############################################################################
with tab3:
    st.write("### AI-Powered Dataset Insights")
    
    if not api_key:
        st.warning("OpenAI API key not configured. Set OPENAI_API_KEY in your .env file to use AI insights.")
        st.info("You can still explore the data using the Data Explorer and Interactive Map tabs.")
        
        # Show basic stats as alternative
        st.write("### Basic Dataset Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top 5 Countries:**")
            for country, count in gdf['CountryName'].value_counts().head(5).items():
                st.write(f"- {country}: {count:,} observations")
            
            st.write("\n**Top Species:**")
            for species, count in gdf['Species'].value_counts().head(5).items():
                st.write(f"- {species}: {count:,} observations")
        
        with col2:
            st.write("**Water Source Types:**")
            for source, count in gdf['WaterSourceType'].value_counts().head(5).items():
                st.write(f"- {source}: {count:,} occurrences")
            
            st.write("\n**Geographic Range:**")
            st.write(f"- Latitude: {gdf['MeasurementLatitude'].min():.2f} to {gdf['MeasurementLatitude'].max():.2f}")
            st.write(f"- Longitude: {gdf['MeasurementLongitude'].min():.2f} to {gdf['MeasurementLongitude'].max():.2f}")
    else:
        st.write("Get intelligent analysis of your mosquito habitat data using OpenAI")
        
        # Prepare dataset summary for OpenAI
        def prepare_dataset_summary(gdf, sample_size=5000):
            """Prepare a concise summary of the dataset for OpenAI"""
            # Create a copy for sampling and convert timestamps
            sample_df = gdf.head(sample_size).drop(columns=['geometry']).copy()
            # Convert any datetime columns to strings
            for col in sample_df.columns:
                if sample_df[col].dtype == 'datetime64[ns]' or sample_df[col].dtype == 'datetime64[ms]':
                    sample_df[col] = sample_df[col].astype(str)
            
            summary = {
                "total_records": len(gdf),
                "columns": gdf.columns.tolist(),
                "geographic_coverage": {
                    "countries": gdf['CountryName'].value_counts().head(10).to_dict(),
                    "latitude_range": [float(gdf['MeasurementLatitude'].min()), float(gdf['MeasurementLatitude'].max())],
                    "longitude_range": [float(gdf['MeasurementLongitude'].min()), float(gdf['MeasurementLongitude'].max())]
                },
                "species_info": {
                    "unique_species": int(gdf['Species'].nunique()),
                    "top_species": gdf['Species'].value_counts().head(10).to_dict()
                },
                "temporal_info": {
                    "date_range": [str(gdf['MeasuredAt'].min()), str(gdf['MeasuredAt'].max())]
                },
                "water_sources": gdf['WaterSourceType'].value_counts().head(10).to_dict(),
                "sample_records": sample_df.to_dict('records')[:5]
            }
            return summary
    
        # Query type selection
        query_type = st.radio(
            "Select analysis type:",
            ["General Overview", "Geographic Patterns", "Species Distribution", "Temporal Trends", "Custom Query"]
        )
        
        # Predefined prompts based on query type
        prompt_templates = {
            "General Overview": "Analyze this mosquito habitat dataset and provide key insights including: 1) Overall patterns in the data, 2) Geographic distribution summary, 3) Most common species and their characteristics, 4) Water source preferences, 5) Any notable trends or anomalies. Be specific and data-driven.",
            "Geographic Patterns": "Analyze the geographic distribution of mosquito habitats in this dataset. Focus on: 1) Which countries have the most observations, 2) Regional patterns, 3) Correlation between elevation and mosquito presence, 4) Geographic clustering patterns.",
            "Species Distribution": "Analyze the species distribution in this mosquito dataset. Include: 1) Most prevalent species, 2) Species diversity by region, 3) Habitat preferences by species, 4) Any rare or concerning species.",
            "Temporal Trends": "Analyze temporal patterns in this mosquito habitat data. Focus on: 1) Seasonal patterns, 2) Trends over time, 3) Peak activity periods, 4) Changes in observations over the years.",
            "Custom Query": ""
        }
        
        if query_type == "Custom Query":
            user_prompt = st.text_area("Enter your custom question about the dataset:", 
                                    height=100,
                                    placeholder="e.g., What is the relationship between water source type and mosquito larvae count?")
        else:
            user_prompt = prompt_templates[query_type]
            st.info(f"Query: {user_prompt}")
        
        if st.button("Get AI Insights", type="primary"):
            if not api_key:
                st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
            elif query_type == "Custom Query" and not user_prompt:
                st.warning("Please enter a custom question.")
            else:
                with st.spinner("Analyzing dataset with AI..."):
                    try:
                        # Prepare data summary
                        data_summary = prepare_dataset_summary(gdf)
                        
                        # Create the messages for OpenAI
                        messages = [
                            {
                                "role": "system",
                                "content": "You are an expert data analyst specializing in mosquito ecology, epidemiology, and environmental science. Analyze the provided dataset and give detailed, actionable insights."
                            },
                            {
                                "role": "user",
                                "content": f"""Here is a mosquito habitat dataset summary:

                                {json.dumps(data_summary, indent=2)}

                                Question/Task: {user_prompt}

                                Please provide a comprehensive analysis with specific data points, trends, and actionable insights."""
                            }
                        ]
                        
                        # Call OpenAI API
                        response = client.chat.completions.create(
                            model=model,
                            messages=messages
                        )
                        
                        # Display the response
                        st.success("Analysis Complete!")
                        st.markdown("### AI Analysis Results")
                        st.markdown(response.choices[0].message.content)
                        
                        # Display token usage
                        with st.expander("API Usage Details"):
                            st.write(f"**Model:** {model}")
                            st.write(f"**Tokens Used:** {response.usage.total_tokens}")
                            st.write(f"- Prompt: {response.usage.prompt_tokens}")
                            st.write(f"- Completion: {response.usage.completion_tokens}")
                        
                    except Exception as e:
                        st.error(f"Error calling OpenAI API: {str(e)}")
                        st.write("Please check your API key and base URL configuration.")
        
        # Additional options
        st.write("---")
        st.write("### Tips for Better Insights")
        st.markdown("""
        - **General Overview**: Get a broad understanding of your dataset
        - **Geographic Patterns**: Understand spatial distribution and regional differences
        - **Species Distribution**: Learn about mosquito species diversity and prevalence
        - **Temporal Trends**: Discover seasonal and long-term patterns
        - **Custom Query**: Ask specific questions about correlations, anomalies, or specific aspects
        """)
