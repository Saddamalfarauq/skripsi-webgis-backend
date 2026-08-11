"""
Tahap 5: Inference Pipeline & GeoJSON Exporter
Menerima folder citra satelit mentah (Sentinel-1 & Sentinel-2), 
kemudian menggunakan CNN untuk Risk Level, dan Mask R-CNN untuk delineasi spasial genangan banjir,
lalu mengekspornya ke GeoJSON untuk WebGIS.
"""

import os
import sys
import json
import glob
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import timm

try:
    import rasterio
    from rasterio.features import shapes
    import geopandas as gpd
    from shapely.geometry import shape, Polygon
    HAS_GEO = True
except ImportError:
    HAS_GEO = False
    print("[!] Modul rasterio, geopandas, atau shapely belum terinstall. Ekspor GeoJSON dinonaktifkan.")

import matplotlib
matplotlib.use('Agg') # Gunakan backend non-interaktif agar aman di multi-threading
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Konfigurasi Model (Relative to this file)
current_dir = os.path.dirname(os.path.abspath(__file__))
CNN_MODEL_PATH = os.path.join(current_dir, "models_weights", "best_flood_cnn_fold1.pth")
MASKRCNN_MODEL_PATH = os.path.join(current_dir, "models_weights", "best_flood_maskrcnn_fold1.pth")

# Risk Levels Map (Sesuai FLOOD_DETECTION_PLAN.md)
RISK_LEVELS = {
    0: "Sangat Rendah",
    1: "Rendah",
    2: "Sedang",
    3: "Tinggi",
    4: "Sangat Tinggi"
}

RISK_COLORS = {
    "Sangat Rendah": "#00FF00",  # Green
    "Rendah": "#FFFF00",         # Yellow
    "Sedang": "#FFA500",         # Orange
    "Tinggi": "#FF0000",         # Red
    "Sangat Tinggi": "#8B0000"   # Dark Red
}

class FloodInferencePipeline:
    def __init__(self, cnn_path=CNN_MODEL_PATH, maskrcnn_path=MASKRCNN_MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")
        
        self._load_cnn(cnn_path)
        self._load_maskrcnn(maskrcnn_path)
        
    def _load_cnn(self, model_path):
        print(f"[INFO] Loading CNN Model from {model_path}...")
        # 9 channels: B4, B3, B2, B8, NDVI, NDWI, VV, VH, DEM
        self.cnn = timm.create_model('efficientnet_b4', pretrained=False, in_chans=9, num_classes=4)
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
            # Menghapus prefiks jika model dilatih dengan torch.compile atau DDP
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.cnn.load_state_dict(state_dict, strict=False)
        else:
            print(f"[WARNING] CNN weights not found at {model_path}. Using random weights.")
        self.cnn.to(self.device)
        self.cnn.eval()

    def _load_maskrcnn(self, model_path):
        print(f"[INFO] Loading Mask R-CNN Model from {model_path}...")
        from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
        from torchvision.models.detection import MaskRCNN
        from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
        
        # Gunakan ResNet-101 sesuai training terakhir
        backbone = resnet_fpn_backbone('resnet101', pretrained=False)
        self.maskrcnn = MaskRCNN(backbone, num_classes=91, box_detections_per_img=100)
        
        # Konfigurasi ulang RPN Anchor sesuai "Trik Level Dewa"
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = (0.2, 0.5, 1.0, 2.0, 5.0)
        self.maskrcnn.rpn.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        
        out_channels = self.maskrcnn.backbone.out_channels
        num_anchors = self.maskrcnn.rpn.anchor_generator.num_anchors_per_location()[0]
        self.maskrcnn.rpn.head = RPNHead(out_channels, num_anchors)
        
        # Customize predictors
        in_features = self.maskrcnn.roi_heads.box_predictor.cls_score.in_features
        self.maskrcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        
        in_features_mask = self.maskrcnn.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.maskrcnn.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, 2)
        
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.maskrcnn.load_state_dict(state_dict, strict=True)
        else:
            print(f"[WARNING] Mask R-CNN weights not found at {model_path}.")
            
        self.maskrcnn.to(self.device)
        self.maskrcnn.eval()

    def load_raster(self, path, target_shape=None):
        with rasterio.open(path) as src:
            data = src.read(1)
            # Resize if needed (for simplicity in this pipeline, we assume uniform shape or we resize via OpenCV)
            if target_shape is not None and data.shape != target_shape:
                import cv2
                data = cv2.resize(data, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
            return data.astype(np.float32), src.transform, src.crs

    def extract_features(self, folder_path):
        """Memuat 9 channel untuk CNN dan 3 channel untuk Mask R-CNN"""
        folder_path = Path(folder_path)
        vv_files = list(folder_path.glob("*_VV.tif"))
        vh_files = list(folder_path.glob("*_VH.tif"))
        b2_files = list(folder_path.glob("*_B02.tif"))
        b3_files = list(folder_path.glob("*_B03.tif"))
        b4_files = list(folder_path.glob("*_B04.tif"))
        b8_files = list(folder_path.glob("*_B08.tif"))
        
        if not all([vv_files, vh_files, b2_files, b3_files, b4_files, b8_files]):
            raise FileNotFoundError(f"Missing required TIF files in {folder_path}")
            
        vv_data, transform, crs = self.load_raster(vv_files[0])
        target_shape = vv_data.shape
        
        vh_data, _, _ = self.load_raster(vh_files[0], target_shape)
        b2_data, _, _ = self.load_raster(b2_files[0], target_shape)
        b3_data, _, _ = self.load_raster(b3_files[0], target_shape)
        b4_data, _, _ = self.load_raster(b4_files[0], target_shape)
        b8_data, _, _ = self.load_raster(b8_files[0], target_shape)
        
        # Calculate Indices
        ndvi = (b8_data - b4_data) / (b8_data + b4_data + 1e-8)
        ndwi = (b3_data - b8_data) / (b3_data + b8_data + 1e-8)
        dem = np.zeros_like(vv_data)  # Dummy DEM
        
        # 1. Input untuk CNN (9 channels)
        # Sesuai urutan preprocess_dataset.py: R, G, B, NIR, NDVI, NDWI, VV, VH, DEM
        cnn_input = np.stack([b4_data, b3_data, b2_data, b8_data, ndvi, ndwi, vv_data, vh_data, dem], axis=0)
        
        # 2. Input untuk Mask R-CNN (3 channels)
        # Sesuai urutan train_maskrcnn_gpu.py: VV, VH, Ratio
        ratio = vv_data / (vh_data + 1e-8)
        maskrcnn_input = np.stack([vv_data, vh_data, ratio], axis=0)
        
        return cnn_input, maskrcnn_input, transform, crs, (b4_data, b3_data, b2_data)

    def predict(self, folder_path, output_dir="D:/HASIL/dataset/staging/predictions"):
        print(f"\n[INFERENCE] Processing {folder_path}...")
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            cnn_input, maskrcnn_input, transform, crs, rgb = self.extract_features(folder_path)
        except Exception as e:
            print(f"[ERROR] Failed to load features: {e}")
            return None, 0.0, 0, None
            
        # 1. CNN Prediction (Risk Level)
        # Resize to 256x256 for CNN if it expects that, or if trained on patches, we might need to patchify.
        # Asumsi: Untuk pipeline sederhana, kita resize seluruh gambar ke 256x256 untuk CNN
        import torch.nn.functional as F
        
        cnn_tensor = torch.from_numpy(cnn_input).unsqueeze(0).to(self.device)
        cnn_tensor = F.interpolate(cnn_tensor, size=(256, 256), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            cnn_out = self.cnn(cnn_tensor)
            probs = torch.softmax(cnn_out, dim=1)[0]
            risk_idx = torch.argmax(probs).item()
            risk_confidence = probs[risk_idx].item()
            risk_label = RISK_LEVELS.get(risk_idx, "Unknown")
            print(f"  -> CNN Risk Level: {risk_label} (Confidence: {risk_confidence:.2f})")
            
        # 2. Mask R-CNN Prediction (Spatial Segments)
        # Resize to 512x512
        mask_tensor = torch.from_numpy(maskrcnn_input).unsqueeze(0).to(self.device)
        mask_tensor = F.interpolate(mask_tensor, size=(512, 512), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            predictions = self.maskrcnn(mask_tensor)[0]
            
        # === CASCADING MASK R-CNN SCORE THRESHOLD ===
        # Mulai dari 0.50, turun bertahap ke 0.35 dan 0.20 jika tidak ada deteksi
        scores = predictions['scores'].cpu().numpy()
        masks_raw = np.array([])
        used_score_thresh = 0.50
        for score_thresh in [0.50, 0.35, 0.20]:
            idx = np.where(scores > score_thresh)[0]
            if len(idx) > 0:
                masks_raw = predictions['masks'].cpu().numpy()[idx, 0]
                boxes = predictions['boxes'].cpu().numpy()[idx]
                used_score_thresh = score_thresh
                break
            boxes = predictions['boxes'].cpu().numpy()[np.array([], dtype=int)]
        
        print(f"  -> Mask R-CNN: {len(masks_raw)} region ditemukan (score > {used_score_thresh:.2f})")
        
        # Buat valid_mask dari piksel non-zero aktual (cek B04 optis ATAU VV radar)
        raw_b4 = cnn_input[0]   # Band B04 (Red reflectance)
        raw_vv = cnn_input[6]   # Band VV SAR backscatter
        # Piksel valid jika ada refleksitansi optis nyata ATAU ada sinyal radar nyata
        valid_mask = ((raw_b4 > 10) | (raw_vv < -2.0) | ((raw_vv > 0.001) & (raw_vv < 1.0))) & \
                     np.isfinite(raw_vv) & np.isfinite(raw_b4)
        
        # Resize mask Mask R-CNN kembali ke resolusi asli
        original_h, original_w = cnn_input.shape[1], cnn_input.shape[2]
        final_masks = []
        import cv2
        for m in masks_raw:
            m_resized = cv2.resize(m, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
            m_binary = (m_resized > 0.40) & valid_mask
            if np.any(m_binary):
                final_masks.append(m_binary)

        # === ADAPTIVE SPECTRAL THRESHOLDING (All-Weather, Auto-Calibrating) ===
        ndwi_data = cnn_input[5]
        vv_data   = cnn_input[6]
        
        valid_vv   = vv_data[valid_mask]
        valid_ndwi = ndwi_data[valid_mask]
        
        # ── 1. Adaptive SAR VV Threshold (Percentile ke-20 piksel valid) ──────────
        if len(valid_vv) > 200:
            vv_p5  = float(np.percentile(valid_vv, 5))
            vv_p20 = float(np.percentile(valid_vv, 20))
            if vv_p5 < -5.0:  # Skala dB (logaritmik, misal -25 s/d -3 dB)
                sar_thresh = float(np.clip(vv_p20, -20.0, -9.0))
                sar_water  = (vv_data < sar_thresh) & (vv_data > -40.0)
            else:              # Skala Linier (0.001 s/d 0.5)
                sar_thresh = float(np.clip(vv_p20, 0.001, 0.12))
                sar_water  = (vv_data < sar_thresh) & (vv_data > 0.0001)
            print(f"  -> Adaptive SAR threshold: {sar_thresh:.4f} "
                  f"({'dB' if vv_p5 < -5.0 else 'linear'} | p20 = {vv_p20:.4f})")
        else:
            sar_water = np.zeros_like(vv_data, dtype=bool)
            print("  -> SAR: tidak cukup piksel valid untuk adaptive threshold")

        # ── 2. Adaptive NDWI Threshold (Otsu Bimodal, bounded 0.02–0.30) ─────────
        if len(valid_ndwi) > 200:
            try:
                # Otsu Thresholding: cari batas optimal antara distribusi air & non-air
                from skimage.filters import threshold_otsu
                ndwi_otsu  = float(threshold_otsu(valid_ndwi))
                ndwi_thresh = float(np.clip(ndwi_otsu, 0.08, 0.30))
                method = "Otsu"
            except Exception:
                # Fallback: gunakan persentil ke-80 (top 20% paling basah = air)
                ndwi_thresh = float(np.clip(np.percentile(valid_ndwi, 80), 0.08, 0.30))
                method = "p80-fallback"
            ndwi_water = ndwi_data > ndwi_thresh
            print(f"  -> Adaptive NDWI threshold ({method}): {ndwi_thresh:.4f}")
        else:
            ndwi_water  = np.zeros_like(ndwi_data, dtype=bool)
            ndwi_thresh = 0.10
            print("  -> NDWI: tidak cukup piksel valid untuk adaptive threshold")

        # Kombinasi OR: SAR menembus awan, NDWI konfirmasi optis
        water_spectral_mask = (sar_water | ndwi_water) & valid_mask

        # Hybrid Fallback: aktif jika Mask R-CNN tidak menghasilkan polygon
        if len(final_masks) == 0 and np.any(water_spectral_mask):
            print("  -> Hybrid Fallback: menggunakan ekstraksi spektral adaptif SAR + NDWI...")
            final_masks.append(water_spectral_mask)

        # 3. Export to GeoJSON
        geojson_path = os.path.join(output_dir, f"flood-risk-latest.geojson")
        if HAS_GEO:
            if len(final_masks) > 0:
                self._export_geojson(final_masks, transform, crs, risk_label, risk_confidence, geojson_path, water_mask=water_spectral_mask)
            else:
                # Generate empty GeoJSON to indicate no floods found
                geojson = {
                    "type": "FeatureCollection",
                    "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::4326"}},
                    "features": []
                }
                with open(geojson_path, 'w') as f:
                    json.dump(geojson, f)
                print(f"[OK] Saved empty GeoJSON to {geojson_path}")
            
        # 4. Visualize
        vis_path = os.path.join(output_dir, "prediction_vis.png")
        self._visualize(rgb, boxes, masks_raw, risk_label, vis_path)
        
        return risk_label, float(risk_confidence), int(len(boxes)), geojson_path
        
    def _export_geojson(self, masks, transform, crs, risk_label, risk_confidence, output_path, water_mask=None):
        print(f"[EXPORT] Generating GeoJSON...")
        
        # Gabungkan semua mask menjadi satu master mask
        master_mask = np.zeros(masks[0].shape, dtype=np.uint8)
        for m in masks:
            master_mask |= m.astype(np.uint8)
            
        # Terapkan water_spectral_mask agar memangkas piksel daratan non-air
        if water_mask is not None:
            master_mask = master_mask & water_mask.astype(np.uint8)
            
        # Rapikan mask dengan morfologi
        import cv2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        master_mask = cv2.morphologyEx(master_mask, cv2.MORPH_CLOSE, kernel)
        master_mask = cv2.morphologyEx(master_mask, cv2.MORPH_OPEN, kernel)
        
        # Load BatasWilayah Maros untuk clipping dan penentuan daerah (kecamatan)
        try:
            import geopandas as gpd
            batas_gdf = gpd.read_file("d:/HASIL/BatasWilayah/73.09_kecamatan.geojson")
        except Exception as e:
            print(f"[WARNING] Gagal meload batas wilayah: {e}")
            batas_gdf = None
            
        features = []
        from shapely.geometry import shape, mapping
        try:
            from pyproj import Geod
            geod = Geod(ellps="WGS84")
        except:
            geod = None
        
        # Extract shapes
        for geom, val in shapes(master_mask, transform=transform):
            if val == 1: # Flooded pixels
                poly = shape(geom)
                
                # Filter noise poligon sangat kecil (< 0.5 Ha)
                poly_area_raw = poly.area
                if poly_area_raw < 1e-6:
                    continue

                if batas_gdf is not None:
                    # Potong polygon dengan masing-masing kecamatan yang beririsan
                    for idx, row in batas_gdf.iterrows():
                        kec_geom = row.geometry
                        kec_name = str(row['nm_kecamatan']).strip()
                        
                        if poly.intersects(kec_geom):
                            intersection = poly.intersection(kec_geom)
                            if intersection.is_empty:
                                continue
                                
                            # Handle hasil intersection TANPA NEGATIVE BUFFER AGAR TIDAK ADA PADDING/MARGIN BUATAN
                            if intersection.geom_type == 'MultiPolygon':
                                polys_to_add = list(intersection.geoms)
                            elif intersection.geom_type == 'GeometryCollection':
                                polys_to_add = [p for p in intersection.geoms if p.geom_type in ['Polygon', 'MultiPolygon']]
                            else:
                                polys_to_add = [intersection]
                                
                            for p in polys_to_add:
                                p_smooth = p.simplify(0.0003, preserve_topology=True)
                                if p_smooth.is_empty:
                                    continue
                                
                                area_ha = 0
                                if geod:
                                    try:
                                        area_ha = abs(geod.geometry_area_perimeter(p_smooth)[0]) / 10000.0
                                    except:
                                        pass
                                
                                # Abaikan poligon mikro NoData
                                if area_ha < 1.0:
                                    continue

                                # Hitung risiko dinamis berdasarkan luas genangan riil (ha)
                                if area_ha > 600:
                                    poly_risk = "Sangat Tinggi"
                                elif area_ha > 300:
                                    poly_risk = "Tinggi"
                                elif area_ha > 100:
                                    poly_risk = "Sedang"
                                elif area_ha > 20:
                                    poly_risk = "Rendah"
                                else:
                                    poly_risk = "Sangat Rendah"
                                
                                # Refinement Spatial Constraint Masking (BPBD Maros Historical Records & Elevation Physics):
                                centroid_y = p_smooth.centroid.y
                                centroid_x = p_smooth.centroid.x
                                
                                # 1. Moncongloe & Mandai Selatan: perbukitan, bebas banjir masif
                                #    Nama GeoJSON: 'Moncong Loe' (dengan spasi)
                                if kec_name.lower() in ["moncongloe", "moncong loe"]:
                                    poly_risk = "Sangat Rendah" if area_ha < 400 else "Rendah"
                                elif kec_name.lower() == "mandai" and centroid_y < -5.06:
                                    poly_risk = "Sangat Rendah" if area_ha < 400 else "Rendah"
                                # 2. Tanralili timur: daratan tinggi
                                elif kec_name.lower() == "tanralili" and centroid_x > 119.62:
                                    poly_risk = "Sangat Rendah" if area_ha < 400 else "Rendah"
                                # 3. Pegunungan Karst Timur: selalu Sangat Rendah
                                #    PERHATIAN: nama GeoJSON 'Malllawa' (3 huruf l), bukan 'Mallawa'
                                elif kec_name.lower() in ["camba", "cenrana", "malllawa", "mallawa", "tompobulu"]:
                                    poly_risk = "Sangat Rendah"
                                # 4. Pesisir Barat (Bontoa, Lau, Maros Baru, Marusu):
                                #    Air permanen tambak ikan/udang sepanjang tahun.
                                #    Cap berdasarkan LUAS (bukan CNN risk_label) agar aktif di kemarau:
                                #    < 2000 Ha = tambak normal → Rendah/Sangat Rendah
                                #    > 2000 Ha = banjir masif  → biarkan nilai area-based risk
                                elif (centroid_x < 119.52) and (kec_name.lower() in ["bontoa", "lau", "maros baru", "marusu"]):
                                    if area_ha < 2000:
                                        poly_risk = "Rendah" if area_ha > 150 else "Sangat Rendah"
                                        
                                feature = {
                                    "type": "Feature",
                                    "geometry": mapping(p_smooth),
                                    "properties": {
                                        "risk_level": poly_risk,
                                        "risk_confidence": float(risk_confidence),
                                        "source": "CNN + Mask R-CNN",
                                        "date_generated": datetime.now().isoformat(),
                                        "daerah": row['nm_kecamatan'],
                                        "area_ha": round(area_ha, 2)
                                    }
                                }
                                features.append(feature)
                else:
                    # Fallback jika batas kecamatan gagal diload
                    if poly.geom_type == 'MultiPolygon':
                        polys_to_add = list(poly.geoms)
                    elif poly.geom_type == 'GeometryCollection':
                        polys_to_add = [p for p in poly.geoms if p.geom_type in ['Polygon', 'MultiPolygon']]
                    else:
                        polys_to_add = [poly]
                    
                    for p in polys_to_add:
                        p_smooth = p.simplify(0.0005, preserve_topology=True)
                        if p_smooth.is_empty:
                            continue
                        
                        area_ha = 0
                        if geod:
                            try:
                                area_ha = abs(geod.geometry_area_perimeter(p_smooth)[0]) / 10000.0
                            except:
                                pass
                                
                        # Hitung risiko dinamis berdasarkan luas (ha)
                        if area_ha > 3000:
                            poly_risk = "Sangat Tinggi"
                        elif area_ha > 1500:
                            poly_risk = "Tinggi"
                        elif area_ha > 500:
                            poly_risk = "Sedang"
                        elif area_ha > 100:
                            poly_risk = "Rendah"
                        else:
                            poly_risk = "Sangat Rendah"
                                
                        feature = {
                            "type": "Feature",
                            "geometry": mapping(p_smooth),
                            "properties": {
                                "risk_level": poly_risk,
                                "risk_confidence": float(risk_confidence),
                                "source": "CNN + Mask R-CNN",
                                "date_generated": datetime.now().isoformat(),
                                "daerah": "Tidak Diketahui",
                                "area_ha": round(area_ha, 2)
                            }
                        }
                        features.append(feature)
                
        geojson = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:EPSG::4326"}
            },
            "features": features
        }
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f)
        print(f"[OK] Saved GeoJSON to {output_path}")

    def _visualize(self, rgb, boxes, masks, risk_label, output_path):
        # Normalisasi RGB untuk visualisasi
        r, g, b = rgb
        r = np.clip(r / np.max(r) * 255, 0, 255).astype(np.uint8)
        g = np.clip(g / np.max(g) * 255, 0, 255).astype(np.uint8)
        b = np.clip(b / np.max(b) * 255, 0, 255).astype(np.uint8)
        img_rgb = np.stack([r, g, b], axis=-1)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(img_rgb)
        
        # Plot Bounding Boxes dan Masks
        color = RISK_COLORS.get(risk_label, "#FF0000")
        
        for i in range(len(boxes)):
            box = boxes[i]
            # Bounding box coordinates are relative to 512x512. Need to scale to original!
            # Since masks are 512x512, we can just display the 512x512 image or scale boxes.
            # To avoid scaling math here, we will just scale the boxes to the RGB image shape.
            h_scale = img_rgb.shape[0] / 512.0
            w_scale = img_rgb.shape[1] / 512.0
            
            x1, y1, x2, y2 = box
            x1, x2 = x1 * w_scale, x2 * w_scale
            y1, y2 = y1 * h_scale, y2 * h_scale
            
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Overlay mask (semi-transparent)
            mask = masks[i]
            import cv2
            mask_resized = cv2.resize(mask, (img_rgb.shape[1], img_rgb.shape[0]))
            
            # Buat overlay biru untuk air
            overlay = np.zeros_like(img_rgb)
            overlay[mask_resized > 0.5] = [0, 150, 255] # Biru air
            
            img_rgb = cv2.addWeighted(img_rgb, 1.0, overlay, 0.4, 0)
            
        ax.imshow(img_rgb)
        ax.set_title(f"Flood Inference\nCNN Risk Level: {risk_label}", fontsize=14, color=color)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"[OK] Saved Visualization to {output_path}")

if __name__ == "__main__":
    print("="*60)
    print("FLOOD DETECTION INFERENCE PIPELINE")
    print("="*60)
    
    pipeline = FloodInferencePipeline()
    
    # Ambil sampel dari SEN12FLOOD
    sample_dir = "D:/HASIL/dataset/SEN12FLOOD"
    folders = [f.path for f in os.scandir(sample_dir) if f.is_dir()]
    
    if len(folders) > 0:
        # Gunakan folder pertama sebagai pengujian
        test_folder = folders[0]
        pipeline.predict(test_folder)
    else:
        print("[!] Tidak ada folder sampel yang ditemukan di SEN12FLOOD")
