import streamlit as st
import geopandas as gpd
import pydeck as pdk

st.set_page_config(page_title="Mosquito Data Visualization", layout="wide")

st.title("🌍 Global Mosquito Data Explorer")

# Load GeoPackage
@st.cache_data
def load_data():
    gdf = gpd.read_file("globe_mosquito.gpkg")
    return gdf

gdf = load_data()

st.write("### Dataset Preview")
st.dataframe(gdf.head())

# Ensure latitude / longitude exist
gdf = gdf.to_crs(epsg=4326)

# Try to infer point geometry
gdf["lon"] = gdf.geometry.x
gdf["lat"] = gdf.geometry.y

# Sidebar controls
st.sidebar.header("Map Controls")
point_size = st.sidebar.slider("Point size", 10, 200, 50)

# PyDeck layer
layer = pdk.Layer(
    "ScatterplotLayer",
    data=gdf,
    get_position="[lon, lat]",
    get_radius=point_size,
    get_fill_color="[200, 30, 0, 160]",
    pickable=True,
)

view_state = pdk.ViewState(
    latitude=gdf["lat"].mean(),
    longitude=gdf["lon"].mean(),
    zoom=2,
)

deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"text": "Lat: {lat}\nLon: {lon}"},
)

st.write("### Interactive Map")
st.pydeck_chart(deck)
