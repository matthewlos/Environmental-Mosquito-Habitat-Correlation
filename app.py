import streamlit as st
import geopandas as gpd
import pydeck as pdk
import os
import openai
from dotenv import load_dotenv
import json
import pandas as pd

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
    # Ensure latitude / longitude exist
    gdf = gdf.to_crs(epsg=4326)

    # Try to infer point geometry
    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y
    presence_cols = ["MosquitoAdults", "MosquitoEggs", "LarvaeCount"]
    used = [c for c in presence_cols if c in gdf.columns]

    if used:
        checks = {}
        for c in used:
            s = gdf[c]
            if pd.api.types.is_bool_dtype(s):
                checks[c] = s.fillna(False).astype(bool)
            else:
                # coerce to numeric (strings -> NaN -> treated as 0), then > 0
                checks[c] = pd.to_numeric(s, errors="coerce").fillna(0) > 0

        present_df = pd.DataFrame(checks, index=gdf.index)
        mask = present_df.any(axis=1)
        gdf_filtered = gdf[mask].copy()
        st.sidebar.write(f"Filtering map to records where any of {used} indicate presence ({len(gdf_filtered)} records)")
    else:
        gdf_filtered = gdf
        st.sidebar.warning(f"No presence columns found (tried: {presence_cols}). Showing all points.")

    # Sidebar controls
    st.sidebar.header("Map Controls")
    point_size = st.sidebar.slider("Point size", 10, 200, 50)

    # PyDeck layer (use filtered data)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=gdf_filtered,
        get_position="[lon, lat]",
        get_radius=point_size,
        get_fill_color="[200, 30, 0, 160]",
        pickable=True,
    )

    # Safe view state
    if len(gdf_filtered) > 0:
        center_lat = gdf_filtered["lat"].mean()
        center_lon = gdf_filtered["lon"].mean()
        zoom = 4 if len(gdf_filtered) > 50 else 2
    else:
        center_lat = gdf["lat"].mean()
        center_lon = gdf["lon"].mean()
        zoom = 1
    # If we previously checked presence columns and found none matching records,
    # a warning was already emitted above; nothing more to do here.

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=zoom,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "Lat: {lat}\nLon: {lon}"},
    )

    st.write("### Interactive Map")
    st.pydeck_chart(deck)

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
            st.write(f"- Latitude: {gdf['MeasurementLatitude'].min():.2f}° to {gdf['MeasurementLatitude'].max():.2f}°")
            st.write(f"- Longitude: {gdf['MeasurementLongitude'].min():.2f}° to {gdf['MeasurementLongitude'].max():.2f}°")
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
        
        if st.button("🚀 Get AI Insights", type="primary"):
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
                        st.markdown("###AI Analysis Results")
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
        st.write("### 💡 Tips for Better Insights")
        st.markdown("""
        - **General Overview**: Get a broad understanding of your dataset
        - **Geographic Patterns**: Understand spatial distribution and regional differences
        - **Species Distribution**: Learn about mosquito species diversity and prevalence
        - **Temporal Trends**: Discover seasonal and long-term patterns
        - **Custom Query**: Ask specific questions about correlations, anomalies, or specific aspects
        """)
