"""
calibrate_thresholds.py  –  Versi 68 Tanggal (Januari 2021 - Agustus 2026)
Setiap tanggal ke-1 pada setiap bulan selama 5 tahun 8 bulan.
Label musim otomatis: FLOOD (Nov-Mar), DRY (Apr-Oct) berdasarkan pola hujan Sul-Sel.

Jalankan:
    cd d:\HASIL\webgis\backend
    python calibrate_thresholds.py
"""

import os, sys, json, io, zipfile, csv, time, requests, numpy as np
from datetime import datetime, timedelta, date

backend_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, backend_path)
import rasterio, ee
from google.oauth2 import service_account

# ── Generate 68 tanggal: tiap tanggal 1 dari Jan 2021 s/d Agu 2026 ──────────
def generate_monthly_dates(start="2021-01-01", end="2026-08-01"):
    dates = []
    d = datetime.strptime(start, "%Y-%m-%d").date()
    e = datetime.strptime(end,   "%Y-%m-%d").date()
    while d <= e:
        month = d.month
        # Musim hujan Sul-Sel: Nov, Des, Jan, Feb, Mar = FLOOD_SEASON
        if month in [11, 12, 1, 2, 3]:
            season = "FLOOD_SEASON"
        else:
            season = "DRY_SEASON"
        dates.append((d.strftime("%Y-%m-%d"), season,
                      f"Bulan-{d.month:02d} Tahun-{d.year}"))
        if d.month == 12:
            d = date(d.year + 1, 1, 1)
        else:
            d = date(d.year, d.month + 1, 1)
    return dates

TEST_DATES = generate_monthly_dates()
OUTPUT_CSV = os.path.join(backend_path, "calibration_report.csv")

# ── Inisialisasi GEE ──────────────────────────────────────────────────────────
def init_gee():
    key_path = os.path.join(backend_path, "secrets", "webgis-484311-b3801a93d0a6.json")
    if os.path.exists(key_path):
        creds = service_account.Credentials.from_service_account_file(key_path)
        ee.Initialize(creds.with_scopes([
            "https://www.googleapis.com/auth/earthengine",
            "https://www.googleapis.com/auth/cloud-platform"]), project="webgis-484311")
        print("[GEE] Service Account OK")
    else:
        ee.Initialize(project="webgis-484311")
        print("[GEE] Default credentials")

# ── Baca bbox Maros ───────────────────────────────────────────────────────────
def get_bbox():
    p = os.path.join(backend_path, "BatasWilayah", "73.09_Maros.geojson")
    data = json.load(open(p, encoding="utf-8"))
    geom = data["features"][0]["geometry"] if data.get("type")=="FeatureCollection" else data.get("geometry",data)
    xs, ys = [], []
    def walk(c):
        if not c: return
        if isinstance(c[0], (int,float)): xs.append(c[0]); ys.append(c[1]); return
        [walk(cc) for cc in c]
    walk(geom["coordinates"])
    return [min(xs), min(ys), max(xs), max(ys)]

# ── Download & analisis satu tanggal ─────────────────────────────────────────
def analyze_date(date_str, roi, temp_dir):
    td   = datetime.strptime(date_str, "%Y-%m-%d")
    s1f  = (td - timedelta(days=12)).strftime("%Y-%m-%d")
    s2f  = (td - timedelta(days=15)).strftime("%Y-%m-%d")
    tod  = (td + timedelta(days=1)).strftime("%Y-%m-%d")

    s1c = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(roi).filterDate(s1f,tod)
           .filter(ee.Filter.eq("instrumentMode","IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VH"))
           .select(["VV","VH"]))
    s2c = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(roi).filterDate(s2f,tod)
           .select(["B2","B3","B4","B8"],["B02","B03","B04","B08"]))

    ds1 = ee.Image.constant([0,0]).rename(["VV","VH"]).clip(roi)
    ds2 = ee.Image.constant([0,0,0,0]).rename(["B02","B03","B04","B08"]).clip(roi)
    s1i = ee.Image(ee.Algorithms.If(s1c.size().gt(0), s1c.mosaic().select(["VV","VH"]).clip(roi), ds1)).unmask(0)
    s2i = ee.Image(ee.Algorithms.If(s2c.size().gt(0), s2c.mosaic().select(["B02","B03","B04","B08"]).clip(roi), ds2)).unmask(0)

    url = ee.Image.cat([s1i, s2i]).getDownloadURL({"scale":100,"crs":"EPSG:4326","region":roi,"format":"GEO_TIFF"})
    os.makedirs(temp_dir, exist_ok=True)
    r = requests.get(url, timeout=300); r.raise_for_status()
    cp = os.path.join(temp_dir, "c.tif")
    if r.content.startswith(b"PK\x03\x04"):
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            tn = [n for n in z.namelist() if n.endswith(".tif")][0]
            open(cp,"wb").write(z.read(tn))
    else:
        open(cp,"wb").write(r.content)

    with rasterio.open(cp) as src:
        vv = src.read(1).astype(float)
        g  = src.read(3).astype(float)
        nir= src.read(4).astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = np.where((g+nir)>0, (g-nir)/(g+nir), 0.0)

    valid = (vv!=0) & np.isfinite(vv) & np.isfinite(ndwi)
    try: os.remove(cp)
    except: pass

    vv_v  = vv[valid];  nd_v = ndwi[valid]
    if len(vv_v) < 100:
        return {"n_valid": int(valid.sum()), "status": "insufficient"}

    scale = "dB" if np.min(vv_v)<-5.0 else "linear"
    p5,p20,p30 = (float(np.percentile(vv_v,q)) for q in [5,20,30])
    sar_t = float(np.clip(p20,-20.0,-9.0)) if scale=="dB" else float(np.clip(p20,0.001,0.12))

    nd_otsu = None
    if len(nd_v)>200:
        try:
            from skimage.filters import threshold_otsu
            nd_otsu = float(np.clip(threshold_otsu(nd_v), 0.02, 0.30))
        except: pass

    # Estimasi: berapa persen piksel yang terdeteksi air oleh adaptive threshold
    sar_water_pct = float((vv_v < sar_t).sum() / len(vv_v) * 100)
    nd_water_pct  = float((nd_v > (nd_otsu or 0.10)).sum() / len(nd_v) * 100) if nd_otsu else 0.0

    return {
        "n_valid":       int(valid.sum()),
        "status":        "ok",
        "scale":         scale,
        "vv_p5":         round(p5,4),
        "vv_p20":        round(p20,4),
        "vv_p30":        round(p30,4),
        "vv_mean":       round(float(np.mean(vv_v)),4),
        "sar_thresh":    round(sar_t,4),
        "sar_water_pct": round(sar_water_pct,2),
        "ndwi_mean":     round(float(np.mean(nd_v)),4),
        "ndwi_p80":      round(float(np.percentile(nd_v,80)),4),
        "ndwi_otsu":     round(nd_otsu,4) if nd_otsu is not None else "N/A",
        "ndwi_water_pct":round(nd_water_pct,2),
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("="*72)
    print(f"  KALIBRASI THRESHOLD ADAPTIF – {len(TEST_DATES)} Tanggal (2021-01 s/d 2026-08)")
    print("="*72)
    init_gee()
    roi = ee.Geometry.Rectangle(get_bbox())

    rows, failed = [], []
    for i,(ds,lbl,desc) in enumerate(TEST_DATES,1):
        print(f"\n[{i:02d}/{len(TEST_DATES)}] {ds} | {lbl}")
        tmp = os.path.join(backend_path,"dataset","staging","calib_temp",ds)
        t0  = time.time()
        try:
            st  = analyze_date(ds, roi, tmp)
            elapsed = round(time.time()-t0,1)
            rows.append({"date":ds,"season":lbl,"desc":desc,"elapsed_s":elapsed,**st})
            if st.get("status")=="ok":
                print(f"  scale={st['scale']} VV_p20={st['vv_p20']} SAR_t={st['sar_thresh']} "
                      f"SAR_water={st['sar_water_pct']}% NDWI_Otsu={st['ndwi_otsu']} "
                      f"NDWI_water={st['ndwi_water_pct']}% [{elapsed}s]")
            else:
                print(f"  [SKIP] {st.get('status')} | n_valid={st.get('n_valid')}")
        except Exception as e:
            elapsed = round(time.time()-t0,1)
            print(f"  [ERROR] {e}")
            rows.append({"date":ds,"season":lbl,"desc":desc,"elapsed_s":elapsed,"status":"error","error":str(e)})
            failed.append(ds)
        time.sleep(2)

    # Tulis CSV
    all_keys = list(dict.fromkeys(k for r in rows for k in r.keys()))
    with open(OUTPUT_CSV,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    # Ringkasan statistik
    ok = [r for r in rows if r.get("status")=="ok"]
    flood_ok = [r for r in ok if "FLOOD" in r["season"]]
    dry_ok   = [r for r in ok if "DRY"   in r["season"]]
    print("\n"+"="*72)
    print(f"  HASIL: {len(ok)}/{len(TEST_DATES)} sukses | {len(failed)} gagal")
    print("="*72)
    if flood_ok:
        sar_f = [r["sar_thresh"] for r in flood_ok if isinstance(r.get("sar_thresh"),float)]
        nd_f  = [r["ndwi_otsu"] for r in flood_ok if isinstance(r.get("ndwi_otsu"),float)]
        sw_f  = [r["sar_water_pct"] for r in flood_ok if isinstance(r.get("sar_water_pct"),float)]
        print(f"\n  [MUSIM HUJAN – {len(flood_ok)} bulan]")
        print(f"    SAR thresh rata2  : {np.mean(sar_f):.4f}  (min={min(sar_f):.4f} max={max(sar_f):.4f})")
        print(f"    NDWI Otsu rata2   : {np.mean(nd_f):.4f}   (min={min(nd_f):.4f} max={max(nd_f):.4f})")
        print(f"    Deteksi air SAR   : {np.mean(sw_f):.1f}%  rata2 dari seluruh piksel")
    if dry_ok:
        sar_d = [r["sar_thresh"] for r in dry_ok if isinstance(r.get("sar_thresh"),float)]
        nd_d  = [r["ndwi_otsu"] for r in dry_ok if isinstance(r.get("ndwi_otsu"),float)]
        sw_d  = [r["sar_water_pct"] for r in dry_ok if isinstance(r.get("sar_water_pct"),float)]
        print(f"\n  [MUSIM KEMARAU – {len(dry_ok)} bulan]")
        print(f"    SAR thresh rata2  : {np.mean(sar_d):.4f}  (min={min(sar_d):.4f} max={max(sar_d):.4f})")
        print(f"    NDWI Otsu rata2   : {np.mean(nd_d):.4f}   (min={min(nd_d):.4f} max={max(nd_d):.4f})")
        print(f"    Deteksi air SAR   : {np.mean(sw_d):.1f}%  rata2 dari seluruh piksel")
    print(f"\n  Laporan lengkap: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
