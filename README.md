# Sentinel FloodRisk AI • FastAPI Geospatial Engine

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch_CUDA-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Google Earth Engine](https://img.shields.io/badge/GEE_Python_API-4285F4?style=for-the-badge&logo=google-earth&logoColor=white)](https://earthengine.google.com/)
[![Rasterio](https://img.shields.io/badge/Rasterio_GDAL-2E7D32?style=for-the-badge)](https://rasterio.readthedocs.io/)
[![Python](https://img.shields.io/badge/Python_3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

> **High-Throughput Deep Learning & Satellite Inference Server** for automated flood extent detection, radar backscatter analysis, and geospatial polygonization in Maros Regency.

---

## ⚡ Technical Highlights

1. **Google Earth Engine (GEE) Ingestion**:
   - Automated cloud masking on Sentinel-2 MSI using SCL / QA60 bands.
   - Sentinel-1 SAR C-Band Level-1 Ground Range Detected (GRD) calibrated to $\sigma^0$ backscatter (VV and VH polarizations).
   - Dynamic SRTM 30m Digital Elevation Model (DEM) clipping to regional bounding box.
2. **Cascaded Dual-Stage Neural Architecture**:
   - **Stage 1 (Macro Assessment)**: *EfficientNet-B4* pre-screens multi-band tiles to categorize disaster occurrence and BNPB risk tiers.
   - **Stage 2 (Instance Delineation)**: *Mask R-CNN with ResNet-101 FPN backbone* generates high-resolution segmentation masks.
3. **Adaptive Otsu Thresholding**:
   - Multi-temporal dual-polarization SAR backscatter thresholding combined with Normalized Difference Water Index (NDWI) Otsu binarization.
   - Vector polygonization using Rasterio and Shapely, with automated topology smoothing and small-speckle noise rejection.
4. **FastAPI Asynchronous Gateway**:
   - Fully asynchronous endpoints with CORS middleware (`allow_origins=["*"]`).
   - Ngrok tunnel compatibility with custom `ngrok-skip-browser-warning` bypass handling.

---

## 📡 API Specification

### `POST /api/predict`
Executes real-time satellite query and deep learning inference for a specified target date.

#### Request Body
```json
{
  "date": "2023-01-15"
}
```

#### Response Structure
```json
{
  "status": "success",
  "message": "Deteksi berhasil dari GEE: 29 area genangan banjir teridentifikasi di Maros",
  "history_id": "maros-flood-20230115",
  "risk_level": "Sangat Tinggi",
  "confidence": 0.985,
  "actual_sat_date": "2023-01-24 (Sentinel-1 SAR & Sentinel-2)",
  "geojson": {
    "type": "FeatureCollection",
    "crs": {
      "type": "name",
      "properties": { "name": "urn:ogc:def:crs:EPSG::4326" }
    },
    "features": [
      {
        "type": "Feature",
        "geometry": {
          "type": "Polygon",
          "coordinates": [[[119.5672, -5.0977], "..."]]
        },
        "properties": {
          "id": 1,
          "luas_ha": 3669.08,
          "daerah": "Bontoa",
          "tingkat_bahaya": "Sangat Tinggi",
          "sensor": "Sentinel-1 SAR & Sentinel-2 Otsu"
        }
      }
    ]
  }
}
```

---

## 🚀 Setup & Execution

### 1. Requirements
- Python `>= 3.10`
- NVIDIA GPU with CUDA support (optional, CPU fallback supported)
- Google Earth Engine Service Account JSON

### 2. Installation
```bash
cd webgis/backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Google Earth Engine Authentication
Place your Google Earth Engine service account key in `secrets/gee_service_account.json` or authenticate via CLI:
```bash
earthengine authenticate
```

### 4. Run Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Expose via Ngrok (For Vercel Production)
```bash
ngrok http 8000
```

---

## 👨‍💻 Author & Thesis Attribution

**Saddam Alfarauq**  
*Department of Informatics, Faculty of Engineering*  
**Universitas Bosowa (Makassar, Indonesia)**  
- **Email**: `takugawakun@gmail.com`  
- **GitHub**: [@Saddamalfarauq](https://github.com/Saddamalfarauq)
