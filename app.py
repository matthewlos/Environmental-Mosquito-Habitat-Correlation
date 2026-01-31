"""
Mosquito Habitat & Environmental Correlation Map Visualization
Shows how environmental factors correlate with mosquito habitats using Google Earth Engine
"""

import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import ee
import sys

# Initialize Earth Engine
@st.cache_resource
def init_earth_engine():
    """Initialize Earth Engine with error handling"""
    # Check if user has set a project in environment variable
    import os
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

st.set_page_config(page_title="Mosquito Habitat Environmental Correlation", layout="wide")

# Check Earth Engine initialization
ee_ready, ee_error = init_earth_engine()

if not ee_ready:
    st.error("⚠️ **Earth Engine not initialized**")

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

st.title("🦟 Mosquito Habitat & Environmental Correlation Map")
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

st.sidebar.header("🎛️ Map Controls")

# Data layers
st.sidebar.subheader("Data Points")
show_mosquito = st.sidebar.checkbox("🦟 Mosquito Locations", value=True)
show_landcover = st.sidebar.checkbox("🌳 Land Cover Sites", value=True)

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
    st.metric("🦟 Mosquito Sites", f"{len(mosquito_gdf):,}")
with col2:
    st.metric("🌳 Land Cover Sites", f"{len(landcover_gdf):,}")
with col3:
    if env_layer != "None":
        st.metric("📡 EE Layer", env_layer)

# ============================================================================
# CREATE MAP
# ============================================================================

st.write("### 🗺️ Interactive Correlation Map")

# Show loading info
if show_mosquito or show_landcover:
    total_mosquito = len(mosquito_gdf) if show_mosquito else 0
    total_landcover = len(landcover_gdf) if show_landcover else 0
    displaying = min(max_markers, total_mosquito + total_landcover)
    st.info(f"📍 Displaying up to {displaying:,} markers (Total available: {total_mosquito:,} mosquito + {total_landcover:,} land cover)")

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
                <b>🌳 Land Cover Site</b><br>
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
        <b>🦟 Mosquito Trap Site</b><br>
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
                <br><hr><b>🌍 Environmental Data (2023):</b><br>
                <b>Elevation:</b> {env_data['elevation']:.0f if env_data['elevation'] else 'N/A'} m<br>
                <b>Temperature:</b> {env_data['temperature']:.1f if env_data['temperature'] else 'N/A'} °C<br>
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
with st.spinner("🗺️ Rendering map... This may take a moment with many markers."):
    st_folium(m, width=1400, height=700, key="main_map")

# Legend
st.write("---")
col1, col2 = st.columns(2)

with col1:
    st.write("**🗺️ Map Legend**")
    st.write("🟢 Green = Land Cover Sites")
    st.write("🟡 Yellow/Orange = Low mosquito larvae")
    st.write("🔴 Red = Moderate mosquito larvae")
    st.write("🔴 Dark Red = High mosquito larvae")

with col2:
    st.write("**📊 Environmental Layers**")
    if env_layer == "Elevation":
        st.write("Blue (low) → Red (high elevation)")
    elif env_layer == "Temperature":
        st.write("Blue (cold) → Red (hot)")
    elif env_layer == "Precipitation":
        st.write("White (dry) → Purple (wet)")
    elif env_layer == "NDVI (Vegetation)":
        st.write("Brown (no vegetation) → Green (dense vegetation)")
    elif env_layer == "Land Cover":
        st.write("Different colors = Different land cover types")

# Info box
st.info("""
**💡 How to use:**
1. Toggle mosquito/land cover sites on/off in the sidebar
2. Select an environmental layer to see how it correlates with mosquito habitats
3. Enable 'Show environmental data in popups' to see exact values for each site
4. Click on map markers to see detailed information
5. Adjust opacity to better see the correlation between layers
""")
