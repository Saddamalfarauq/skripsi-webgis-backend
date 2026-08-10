import os
import sys
import uuid
import json
import io
import zipfile
import requests
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
import ee
from google.oauth2 import service_account
import rasterio

# Menambahkan folder backend ke sys.path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if backend_path not in sys.path:
    sys.path.append(backend_path)

from inference_pipeline import FloodInferencePipeline
from database import get_db
import models

router = APIRouter()

# Inisialisasi Google Earth Engine API
print("[FASTAPI] Inisialisasi Google Earth Engine API...")
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    key_path = os.path.join(current_dir, '..', 'secrets', 'webgis-484311-b3801a93d0a6.json')
    
    if os.path.exists(key_path):
        credentials = service_account.Credentials.from_service_account_file(key_path)
        scoped_credentials = credentials.with_scopes(['https://www.googleapis.com/auth/earthengine', 'https://www.googleapis.com/auth/cloud-platform'])
        ee.Initialize(scoped_credentials, project='webgis-484311')
        print("[FASTAPI] GEE berhasil diinisialisasi dengan Service Account.")
    else:
        print(f"[FASTAPI] File secrets GEE tidak ditemukan di {key_path}. Menggunakan default credentials.")
        ee.Initialize(project='webgis-484311')
        print("[FASTAPI] GEE berhasil diinisialisasi dengan kredensial lokal.")
except Exception as e:
    print(f"[FASTAPI] Gagal inisialisasi GEE: {e}")

# Load model di memori agar cepat (singleton)
print("[FASTAPI] Loading Inference Pipeline...")
try:
    pipeline = FloodInferencePipeline()
except Exception as e:
    print(f"[FASTAPI] Gagal meload model: {e}")
    pipeline = None

# Fungsi Bantuan
def get_bbox(geojson_path):
    with open(geojson_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if data.get('type') == 'FeatureCollection':
        geom = data['features'][0]['geometry']
    else:
        geom = data.get('geometry', data)
    
    xs, ys = [], []
    def iter_coords(coords):
        if not coords: return
        if isinstance(coords[0], (int, float)):
            xs.append(coords[0])
            ys.append(coords[1])
            return
        for c in coords:
            iter_coords(c)
    iter_coords(geom['coordinates'])
    return [min(xs), min(ys), max(xs), max(ys)]

class PredictRequest(BaseModel):
    date: str  # Format YYYY-MM-DD

@router.post("/predict")
async def predict_flood(request: PredictRequest, db: Session = Depends(get_db)):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model belum siap atau gagal diload.")
        
    date_str = request.date
    task_id = str(uuid.uuid4())
    temp_dir = os.path.join(backend_path, "dataset", "staging", "gee_temp", task_id)
    output_dir = os.path.join(backend_path, "dataset", "staging", "predictions", task_id)
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Mengunduh citra dari GEE
        target_date = datetime.strptime(date_str, "%Y-%m-%d")
        to_date = target_date + timedelta(days=1)
        
        s1_from = (target_date - timedelta(days=12)).strftime("%Y-%m-%d")
        s2_from = (target_date - timedelta(days=15)).strftime("%Y-%m-%d")
        to_date_str = to_date.strftime("%Y-%m-%d")
        
        bbox = get_bbox(os.path.join(backend_path, "BatasWilayah", "73.09_Maros.geojson"))
        roi = ee.Geometry.Rectangle(bbox)
        
        # Sentinel-1
        s1_col = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filterBounds(roi) \
            .filterDate(s1_from, to_date_str) \
            .filter(ee.Filter.eq('instrumentMode', 'IW')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
            .select(['VV', 'VH'])
            
        # Sentinel-2 (Tingkatkan toleransi cloud jika rentang tanggal sempit)
        s2_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(roi) \
            .filterDate(s2_from, to_date_str) \
            .select(['B2', 'B3', 'B4', 'B8'], ['B02', 'B03', 'B04', 'B08'])
            
        # Buat dummy fallback jika salah satu koleksi citra kosong pada rentang tanggal tersebut
        dummy_s1 = ee.Image.constant([0, 0]).rename(['VV', 'VH']).clip(roi)
        dummy_s2 = ee.Image.constant([0, 0, 0, 0]).rename(['B02', 'B03', 'B04', 'B08']).clip(roi)
        
        s1_image = ee.Algorithms.If(s1_col.size().gt(0), s1_col.mosaic().select(['VV', 'VH']).clip(roi), dummy_s1)
        s2_image = ee.Algorithms.If(s2_col.size().gt(0), s2_col.mosaic().select(['B02', 'B03', 'B04', 'B08']).clip(roi), dummy_s2)
        
        s1_image = ee.Image(s1_image).unmask(0)
        s2_image = ee.Image(s2_image).unmask(0)
        
        # Get actual satellite date (using Sentinel-1 as reference)
        try:
            first_s1 = ee.Image(s1_col.sort('system:time_start', False).first())
            date_ms = first_s1.get('system:time_start').getInfo()
            actual_sat_date = datetime.fromtimestamp(date_ms/1000).strftime('%Y-%m-%d')
        except:
            actual_sat_date = date_str
            
        composite = ee.Image.cat([s1_image, s2_image])
        
        url = composite.getDownloadURL({
            'scale': 100, 
            'crs': 'EPSG:4326',
            'region': roi,
            'format': 'GEO_TIFF'
        })
        
        resp = requests.get(url)
        resp.raise_for_status()
        
        composite_path = os.path.join(temp_dir, "composite.tif")
        content = resp.content
        if content.startswith(b'PK\x03\x04'):
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                tif_name = [n for n in z.namelist() if n.endswith('.tif')][0]
                with open(composite_path, "wb") as f:
                    f.write(z.read(tif_name))
        else:
            with open(composite_path, "wb") as f:
                f.write(content)
                
        # Split band
        with rasterio.open(composite_path) as src:
            meta = src.meta.copy()
            meta.update(count=1)
            bands = ["VV", "VH", "B02", "B03", "B04", "B08"]
            for i, band_name in enumerate(bands):
                band_path = os.path.join(temp_dir, f"scene_{band_name}.tif")
                with rasterio.open(band_path, 'w', **meta) as dst:
                    dst.write(src.read(i+1), 1)
                    
        os.remove(composite_path)

        # 4. Panggil Inference Pipeline
        pipeline.predict(temp_dir, output_dir=output_dir)

        # 5. Baca hasil dari JSON
        geojson_file = os.path.join(output_dir, "flood-risk-latest.geojson")
        if not os.path.exists(geojson_file):
            raise HTTPException(status_code=500, detail="Gagal memproses prediksi. GeoJSON tidak terbentuk.")
            
        with open(geojson_file, "r") as f:
            geojson_data = json.load(f)
            
        features = geojson_data.get("features", [])
        if len(features) == 0:
            risk_level = "Sangat Rendah"
            confidence = 1.0
            flooded_count = 0
        else:
            props = features[0].get("properties", {})
            risk_level = props.get("risk_level", "Unknown")
            confidence = props.get("risk_confidence", 0.0)
            flooded_count = len(features)

        # 6. Simpan riwayat ke PostgreSQL
        history_record = models.PredictionHistory(
            location_name=f"GEE_Maros_{date_str}",
            risk_level=risk_level,
            risk_confidence=confidence,
            flooded_regions_count=flooded_count,
            geojson_path=geojson_file,
            image_path=os.path.join(output_dir, "prediction_vis.png")
        )
        db.add(history_record)
        db.commit()
        db.refresh(history_record)

        # 7. Kembalikan data ke Frontend
        return {
            "status": "success",
            "message": f"Deteksi berhasil dari GEE untuk tanggal {date_str}: {flooded_count} area banjir ditemukan.",
            "history_id": history_record.id,
            "risk_level": risk_level,
            "confidence": confidence,
            "actual_sat_date": actual_sat_date,
            "geojson": geojson_data
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
