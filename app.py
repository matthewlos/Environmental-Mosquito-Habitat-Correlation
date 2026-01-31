import streamlit as st
import geopandas as gpd
import pydeck as pdk

st.set_page_config(page_title="Mosquito & Land Cover Visualization", layout="wide")

st.title("🦟 Mosquito Habitat & Land Cover Explorer")

# Load data
@st.cache_data
def load_mosquito_data():
    gdf = gpd.read_file("globe_mosquito.gpkg")
    gdf = gdf.to_crs(epsg=4326)
    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y
    return gdf

@st.cache_data
def load_landcover_data():
    gdf = gpd.read_file("globe_land_cover.gpkg")
    gdf = gdf.to_crs(epsg=4326)
    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y
    return gdf

mosquito_gdf = load_mosquito_data()
landcover_gdf = load_landcover_data()

# Sidebar controls
st.sidebar.header("Map Controls")
show_mosquito = st.sidebar.checkbox("Show Mosquito Locations", value=True)
show_landcover = st.sidebar.checkbox("Show Land Cover Sites", value=True)
point_size = st.sidebar.slider("Point size", 10, 200, 50)

# Display stats
col1, col2 = st.columns(2)
with col1:
    st.metric("Mosquito Trapping Locations", len(mosquito_gdf))
with col2:
    st.metric("Land Cover Observation Sites", len(landcover_gdf))

# Build layers
layers = []

if show_landcover:
    landcover_layer = pdk.Layer(
        "ScatterplotLayer",
        data=landcover_gdf,
        get_position="[lon, lat]",
        get_radius=point_size,
        get_fill_color="[34, 139, 34, 140]",  # Forest green
        pickable=True,
    )
    layers.append(landcover_layer)

if show_mosquito:
    mosquito_layer = pdk.Layer(
        "ScatterplotLayer",
        data=mosquito_gdf,
        get_position="[lon, lat]",
        get_radius=point_size,
        get_fill_color="[200, 30, 0, 160]",  # Red
        pickable=True,
    )
    layers.append(mosquito_layer)

all_lats = []
all_lons = []
if show_mosquito:
    all_lats.extend(mosquito_gdf["lat"].tolist())
    all_lons.extend(mosquito_gdf["lon"].tolist())
if show_landcover:
    all_lats.extend(landcover_gdf["lat"].tolist())
    all_lons.extend(landcover_gdf["lon"].tolist())

view_state = pdk.ViewState(
    latitude=sum(all_lats) / len(all_lats) if all_lats else 0,
    longitude=sum(all_lons) / len(all_lons) if all_lons else 0,
    zoom=2,
)

deck = pdk.Deck(
    layers=layers,
    initial_view_state=view_state,
    tooltip={"text": "Location: {lat}, {lon}\nCountry: {CountryName}"},
)

st.write("### Interactive Map")
st.caption("Red = Mosquito Trapping Locations | Green = Land Cover Sites")
st.pydeck_chart(deck)

# Data preview
st.write("### Dataset Previews")
col1, col2 = st.columns(2)

with col1:
    st.write("**Mosquito Data**")
    st.dataframe(mosquito_gdf[["CountryName", "MeasuredDate", "WaterSourceType", "LarvaeCount", "lat", "lon"]].head())

with col2:
    st.write("**Land Cover Data**")
    st.dataframe(landcover_gdf[["CountryName", "MeasuredDate", "MucDescription", "lat", "lon"]].head())
