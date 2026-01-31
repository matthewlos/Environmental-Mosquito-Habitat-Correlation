from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import geopandas as gpd
import ee
import json

## Earth Engine & Mosquito Habitat Correlation API

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Initialize Earth Engine
EE_INITIALIZED = False
try:
    # Try with high-volume endpoint (doesn't require project)
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    EE_INITIALIZED = True
    print("Earth Engine initialized successfully!")
except Exception as e:
    try:
        # Fall back to legacy
        ee.Initialize()
        EE_INITIALIZED = True
        print("Earth Engine initialized successfully (legacy)!")
    except Exception as e2:
        print(f"Earth Engine initialization failed: {e2}")
        print("Run 'earthengine authenticate' to set up credentials")

def get_environmental_layers() -> Dict[str, ee.Image]:
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

def get_visualization_params() -> Dict[str, Dict[str, Any]]:
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

def extract_environmental_data_for_point(lat: float, lon: float, year: str = '2023') -> Dict[str, Any]:
    """Extract comprehensive environmental data from Earth Engine for a point"""
    point = ee.Geometry.Point([lon, lat])

    # Elevation
    elevation = ee.Image('USGS/SRTMGL1_003').select('elevation')
    elev_dict = elevation.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=30
    ).getInfo()

    # Temperature (MODIS LST)
    temperature = ee.ImageCollection('MODIS/061/MOD11A1') \
        .filterDate(f'{year}-01-01', f'{year}-12-31') \
        .select('LST_Day_1km') \
        .mean()
    temp_dict = temperature.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=1000
    ).getInfo()

    # Precipitation (CHIRPS)
    precipitation = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
        .filterDate(f'{year}-01-01', f'{year}-12-31') \
        .select('precipitation') \
        .sum()
    precip_dict = precipitation.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=5000
    ).getInfo()

    # NDVI (Vegetation)
    ndvi = ee.ImageCollection('MODIS/061/MOD13A2') \
        .filterDate(f'{year}-01-01', f'{year}-12-31') \
        .select('NDVI') \
        .mean()
    ndvi_dict = ndvi.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=1000
    ).getInfo()

    # Land cover
    landcover = ee.ImageCollection('MODIS/061/MCD12Q1') \
        .filterDate(f'{year}-01-01', f'{year}-12-31') \
        .first() \
        .select('LC_Type1')
    lc_dict = landcover.reduceRegion(
        reducer=ee.Reducer.mode(),
        geometry=point,
        scale=500
    ).getInfo()

    # Convert and clean data
    temp_celsius = None
    if temp_dict.get('LST_Day_1km'):
        temp_celsius = temp_dict['LST_Day_1km'] * 0.02 - 273.15

    ndvi_value = None
    if ndvi_dict.get('NDVI'):
        ndvi_value = ndvi_dict['NDVI'] * 0.0001

    return {
        'elevation_m': elev_dict.get('elevation'),
        'temperature_celsius': temp_celsius,
        'annual_precipitation_mm': precip_dict.get('precipitation'),
        'ndvi': ndvi_value,
        'land_cover_type': lc_dict.get('LC_Type1')
    }

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Mosquito Habitat & Earth Engine API",
        description="API for analyzing mosquito habitats with Earth Engine environmental data"
    )

    class Record(BaseModel):
        lat: float
        lon: float
        properties: Dict[str, Any] = {}

    class MapRequest(BaseModel):
        points: List[Record]

    class LocationRequest(BaseModel):
        lat: float
        lon: float
        buffer_meters: Optional[int] = 1000

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "earth_engine_ready": EE_INITIALIZED
        }

    @app.get("/mosquito-data")
    async def get_mosquito_data(limit: Optional[int] = 100):
        """Load mosquito trapping data from GeoPackage"""
        try:
            gdf = gpd.read_file("globe_mosquito.gpkg")
            gdf = gdf.head(limit)
            gdf = gdf.to_crs(epsg=4326)
            gdf["lon"] = gdf.geometry.x
            gdf["lat"] = gdf.geometry.y
            return {
                "count": len(gdf),
                "data": gdf.drop(columns=["geometry"]).to_dict(orient="records")
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/landcover-data")
    async def get_landcover_data(limit: Optional[int] = 100):
        """Load land cover observation data from GeoPackage"""
        try:
            gdf = gpd.read_file("globe_land_cover.gpkg")
            gdf = gdf.head(limit)
            gdf = gdf.to_crs(epsg=4326)
            gdf["lon"] = gdf.geometry.x
            gdf["lat"] = gdf.geometry.y
            return {
                "count": len(gdf),
                "data": gdf.drop(columns=["geometry"]).to_dict(orient="records")
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/earth-engine/environment")
    async def get_environmental_data(request: LocationRequest):
        """Extract environmental data from Earth Engine for a location"""
        if not EE_INITIALIZED:
            raise HTTPException(
                status_code=503,
                detail="Earth Engine not initialized. Run 'earthengine authenticate'"
            )

        try:
            point = ee.Geometry.Point([request.lon, request.lat])
            buffer = point.buffer(request.buffer_meters)

            # Get elevation data (SRTM)
            elevation = ee.Image('USGS/SRTMGL1_003').select('elevation')
            elev_value = elevation.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=30
            ).getInfo()

            # Get temperature data (MODIS)
            temperature = ee.ImageCollection('MODIS/061/MOD11A1') \
                .filterDate('2023-01-01', '2024-01-01') \
                .select('LST_Day_1km') \
                .mean()
            temp_value = temperature.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=1000
            ).getInfo()

            # Convert Kelvin to Celsius
            temp_celsius = (temp_value.get('LST_Day_1km', 0) * 0.02 - 273.15) if temp_value.get('LST_Day_1km') else None

            # Get precipitation data (CHIRPS)
            precipitation = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
                .filterDate('2023-01-01', '2024-01-01') \
                .select('precipitation') \
                .sum()
            precip_value = precipitation.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=5000
            ).getInfo()

            # Get NDVI (vegetation index)
            ndvi = ee.ImageCollection('MODIS/061/MOD13A2') \
                .filterDate('2023-01-01', '2024-01-01') \
                .select('NDVI') \
                .mean()
            ndvi_value = ndvi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=1000
            ).getInfo()

            return {
                "location": {"lat": request.lat, "lon": request.lon},
                "buffer_meters": request.buffer_meters,
                "environmental_data": {
                    "elevation_m": elev_value.get('elevation'),
                    "temperature_celsius": temp_celsius,
                    "annual_precipitation_mm": precip_value.get('precipitation'),
                    "ndvi": ndvi_value.get('NDVI', 0) * 0.0001 if ndvi_value.get('NDVI') else None
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Earth Engine error: {str(e)}")

    @app.post("/correlate/mosquito-environment")
    async def correlate_mosquito_with_environment(limit: Optional[int] = 50):
        """Correlate mosquito data with Earth Engine environmental data"""
        if not EE_INITIALIZED:
            raise HTTPException(
                status_code=503,
                detail="Earth Engine not initialized. Run 'earthengine authenticate'"
            )

        try:
            # Load mosquito data
            gdf = gpd.read_file("globe_mosquito.gpkg").head(limit)
            gdf = gdf.to_crs(epsg=4326)

            results = []
            for idx, row in gdf.iterrows():
                lon, lat = row.geometry.x, row.geometry.y

                # Get comprehensive environmental data
                env_data = extract_environmental_data_for_point(lat, lon)

                results.append({
                    "site_id": row.get("SiteId"),
                    "country": row.get("CountryName"),
                    "lat": lat,
                    "lon": lon,
                    "larvae_count": row.get("LarvaeCount"),
                    "water_source": row.get("WaterSourceType"),
                    "measured_date": str(row.get("MeasuredDate")),
                    **env_data
                })

            return {
                "count": len(results),
                "data": results
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/ee-layers")
    async def get_ee_layer_info():
        """Get available Earth Engine layers and their visualization parameters"""
        if not EE_INITIALIZED:
            raise HTTPException(
                status_code=503,
                detail="Earth Engine not initialized"
            )

        return {
            "layers": list(get_environmental_layers().keys()),
            "visualization_params": get_visualization_params()
        }

    @app.post("/to-geojson")
    async def to_geojson(request: MapRequest):
        """Convert point data to GeoJSON format"""
        rows = []
        for rec in request.points:
            row = {"lat": rec.lat, "lon": rec.lon}
            row.update(rec.properties)
            rows.append(row)

        df = pd.DataFrame(rows)
        geojson = dataframe_to_geojson(df, lat_col="lat", lon_col="lon")
        return geojson

    def dataframe_to_geojson(df: pd.DataFrame, lat_col: str, lon_col: str) -> Dict[str, Any]:
        features = []
        for _, row in df.iterrows():
            properties = row.drop([lat_col, lon_col]).to_dict()
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row[lon_col], row[lat_col]],
                },
                "properties": properties,
            }
            features.append(feature)
        geojson = {
            "type": "FeatureCollection",
            "features": features,
        }
        return geojson

if __name__ == "__main__" and FASTAPI_AVAILABLE:
    # Run with: python api.py
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)