from typing import List, Optional, Dict, Any
import pandas as pd
import geopandas as gpd
import ee

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
    ee.Initialize()
    EE_INITIALIZED = True
except Exception as e:
    print(f"Earth Engine initialization failed: {e}")
    print("Run 'earthengine authenticate' to set up credentials")

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

                # Get environmental data for this location
                point = ee.Geometry.Point([lon, lat])

                elevation = ee.Image('USGS/SRTMGL1_003').select('elevation')
                elev_value = elevation.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=point,
                    scale=30
                ).getInfo()

                results.append({
                    "site_id": row.get("SiteId"),
                    "country": row.get("CountryName"),
                    "lat": lat,
                    "lon": lon,
                    "larvae_count": row.get("LarvaeCount"),
                    "water_source": row.get("WaterSourceType"),
                    "elevation_m": elev_value.get('elevation')
                })

            return {
                "count": len(results),
                "data": results
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

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