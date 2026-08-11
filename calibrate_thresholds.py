"""
=============================================================================
calibrate_thresholds.py
Kalibrasi Threshold Adaptif WebGIS Deteksi Banjir - Kabupaten Maros

Menguji 12 tanggal representatif (6 banjir + 6 kemarau, 2021-2026) dengan
mengunduh komposit Sentinel-1 & Sentinel-2 dari Google Earth Engine, lalu
menghitung threshold optimal SAR VV dan NDWI menggunakan metode:
  - SAR VV  : Persentil ke-5, ke-20, ke-30
  - NDWI    : Otsu bimodal thresholding
Hasil ditulis ke calibration_report.csv.

Jalankan:
    cd d:\HASIL\webgis\backend
    python calibrate_thresholds.py
=============================================================================
"""

import os
import sys
import json
import io
import zipfile
import csv
import time
import requests
import numpy as np
from datetime import datetime, timedelta

backend_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, backend_path)

import rasterio
import ee
from google.oauth2 import service_account

TEST_DATES = [
    ("2022-12-23", "FLOOD", "Banjir Bandang & Rob Masif 14 Kecamatan Des 2022"),
    ("2023-02-19", "FLOOD", "Banjir Luapan DAS Maros 8 Kecamatan Feb 2023"),
    ("2024-12-21", "FLOOD", "Banjir Luapan Sungai Cenrana Des 2024"),
    ("2025-02-11", "FLOOD", "Banjir Trans-Sulawesi Terbesar Feb 2025"),
    ("2021-01-22", "FLOOD", "Banjir Bandang Historis Sul-Sel Jan 2021"),
    ("2021-12-05", "FLOOD", "Banjir Awal Musim Hujan Des 2021"),
    ("2022-08-15", "DRY",   "Puncak Kemarau Agustus 2022"),
    ("2023-08-20", "DRY",   "Puncak Kemarau Agustus 2023"),
    ("2024-07-10", "DRY",   "Kemarau Juli 2024"),
    ("2025-08-05", "DRY",   "Kemarau Agustus 2025"),
    ("2026-08-08", "DRY",   "Kemarau Agustus 2026"),
    ("2022-06-01", "DRY",   "Awal Kemarau Juni 2022"),
]

OUTPUT_CSV = os.path.join(backend_path, "calibration_report.csv")

def init_gee():
    key_path = os.path.join(backend_path, "secrets", "webgis-484311-b3801a93d0a6.json")
    if os.path.exists(key_path):
        credentials = service_account.Credentials.from_service_account_file(key_path)
        scoped = credentials.with_scopes([
            "https://www.googleapis.com/auth/earthengine",
            "https://www.googleapis.com/auth/cloud-platform"
        ])
        ee.Initialize(scoped, project="webgis-484311")
        print("[GEE] Initialized with Service Account")
    else:
        ee.Initialize(project="webgis-484311")
        print("[GEE] Initialized with default credentials")

def get_bbox():
    geojson_path = os.path.join(backend_path, "BatasWilayah", "73.09_Maros.geojson")
    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    geom = data["features"][0]["geometry"] if data.get("type") == "FeatureCollection" else data.get("geometry", data)
    xs, ys = [], []
    def iter_coords(c):
        if not c: return
        if isinstance(c[0], (int, float)):
            xs.append(c[0]); ys.append(c[1]); return
        for cc in c: iter_coords(cc)
    iter_coords(geom["coordinates"])
    return [min(xs), min(ys), max(xs), max(ys)]

def download_and_analyze(date_str, roi, temp_dir):
    target_date = datetime.strptime(date_str, "%Y-%m-%d")
    s1_from = (target_date - timedelta(days=12)).strftime("%Y-%m-%d")
    s2_from = (target_date - timedelta(days=15)).strftime("%Y-%m-%d")
    to_date = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")

    s1_col = (ee.ImageCollection("COPERNICUS/S1_GRD")
              .filterBounds(roi).filterDate(s1_from, to_date)
              .filter(ee.Filter.eq("instrumentMode", "IW"))
              .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
              .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
              .select(["VV", "VH"]))

    s2_col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(roi).filterDate(s2_from, to_date)
              .select(["B2", "B3", "B4", "B8"], ["B02", "B03", "B04", "B08"]))

    dummy_s1 = ee.Image.constant([0, 0]).rename(["VV", "VH"]).clip(roi)
    dummy_s2 = ee.Image.constant([0, 0, 0, 0]).rename(["B02", "B03", "B04", "B08"]).clip(roi)

    s1_img = ee.Image(ee.Algorithms.If(s1_col.size().gt(0),
               s1_col.mosaic().select(["VV", "VH"]).clip(roi), dummy_s1)).unmask(0)
    s2_img = ee.Image(ee.Algorithms.If(s2_col.size().gt(0),
               s2_col.mosaic().select(["B02", "B03", "B04", "B08"]).clip(roi), dummy_s2)).unmask(0)

    url = ee.Image.cat([s1_img, s2_img]).getDownloadURL({
        "scale": 100, "crs": "EPSG:4326", "region": roi, "format": "GEO_TIFF"
    })

    os.makedirs(temp_dir, exist_ok=True)
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()

    composite_path = os.path.join(temp_dir, "composite.tif")
    content = resp.content
    if content.startswith(b"PK\x03\x04"):
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            tif_name = [n for n in z.namelist() if n.endswith(".tif")][0]
            with open(composite_path, "wb") as f:
                f.write(z.read(tif_name))
    else:
        with open(composite_path, "wb") as f:
            f.write(content)

    with rasterio.open(composite_path) as src:
        vv_arr  = src.read(1).astype(float)
        b03_arr = src.read(3).astype(float)
        b08_arr = src.read(4).astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi_arr = np.where((b03_arr + b08_arr) > 0,
                            (b03_arr - b08_arr) / (b03_arr + b08_arr), 0.0)

    valid = (vv_arr != 0) & np.isfinite(vv_arr) & np.isfinite(ndwi_arr)
    vv_v  = vv_arr[valid]
    nd_v  = ndwi_arr[valid]

    try: os.remove(composite_path)
    except: pass

    if len(vv_v) == 0:
        return {"n_valid": 0, "error": "no valid pixels"}

    scale = "dB" if np.min(vv_v) < -5.0 else "linear"
    vv_p5  = float(np.percentile(vv_v, 5))
    vv_p20 = float(np.percentile(vv_v, 20))
    vv_p30 = float(np.percentile(vv_v, 30))

    if scale == "dB":
        sar_thresh = float(np.clip(vv_p20, -20.0, -9.0))
    else:
        sar_thresh = float(np.clip(vv_p20, 0.001, 0.12))

    ndwi_otsu = None
    if len(nd_v) > 200:
        try:
            from skimage.filters import threshold_otsu
            ndwi_otsu = float(np.clip(threshold_otsu(nd_v), 0.02, 0.30))
        except: pass

    return {
        "n_valid":    int(valid.sum()),
        "scale":      scale,
        "vv_p5":      round(vv_p5, 4),
        "vv_p20":     round(vv_p20, 4),
        "vv_p30":     round(vv_p30, 4),
        "vv_mean":    round(float(np.mean(vv_v)), 4),
        "sar_thresh": round(sar_thresh, 4),
        "ndwi_mean":  round(float(np.mean(nd_v)), 4),
        "ndwi_p80":   round(float(np.percentile(nd_v, 80)), 4),
        "ndwi_otsu":  round(ndwi_otsu, 4) if ndwi_otsu is not None else "N/A",
    }

def main():
    print("=" * 70)
    print("  KALIBRASI THRESHOLD ADAPTIF WebGIS Maros – 12 Tanggal Test")
    print("=" * 70)

    init_gee()
    roi = ee.Geometry.Rectangle(get_bbox())

    rows = []
    for i, (date_str, label, desc) in enumerate(TEST_DATES, 1):
        print(f"\n[{i:02d}/12] {date_str} | {label}")
        print(f"         {desc}")
        temp_dir = os.path.join(backend_path, "dataset", "staging", "calib_temp", date_str)
        t0 = time.time()
        try:
            stats = download_and_analyze(date_str, roi, temp_dir)
            elapsed = round(time.time() - t0, 1)
            print(f"  scale={stats.get('scale')} | VV_p20={stats.get('vv_p20')} | "
                  f"SAR_thresh={stats.get('sar_thresh')} | "
                  f"NDWI_Otsu={stats.get('ndwi_otsu')} | {elapsed}s")
            rows.append({"date": date_str, "label": label, "desc": desc,
                         "elapsed_s": elapsed, **stats})
        except Exception as e:
            print(f"  [ERROR] {e}")
            rows.append({"date": date_str, "label": label, "desc": desc, "error": str(e)})
        time.sleep(3)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    flood_r = [r for r in rows if r.get("label") == "FLOOD" and "error" not in r]
    dry_r   = [r for r in rows if r.get("label") == "DRY"   and "error" not in r]
    print("\n" + "=" * 70)
    print("  RINGKASAN")
    print("=" * 70)
    if flood_r:
        sar_f = [r["sar_thresh"] for r in flood_r if isinstance(r.get("sar_thresh"), float)]
        nd_f  = [r["ndwi_otsu"] for r in flood_r if isinstance(r.get("ndwi_otsu"), float)]
        print(f"  FLOOD - SAR thresh rata2: {np.mean(sar_f):.4f} | NDWI Otsu rata2: {np.mean(nd_f):.4f}")
    if dry_r:
        sar_d = [r["sar_thresh"] for r in dry_r if isinstance(r.get("sar_thresh"), float)]
        nd_d  = [r["ndwi_otsu"] for r in dry_r if isinstance(r.get("ndwi_otsu"), float)]
        print(f"  DRY   - SAR thresh rata2: {np.mean(sar_d):.4f} | NDWI Otsu rata2: {np.mean(nd_d):.4f}")
    print(f"\n  Laporan: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
