# app.py
# Jalankan dengan: streamlit run app.py

import streamlit as st
import numpy as np
import cv2
import dlib
import joblib
import os
import time
from PIL import Image
from io import BytesIO
from scipy.signal import wiener
from skimage.color import rgb2lab, lab2rgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

def download_shape_predictor():
    """Download shape_predictor_68_face_landmarks.dat jika belum ada."""
    dat_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "shape_predictor_68_face_landmarks.dat")
    if os.path.exists(dat_path):
        return dat_path
    
    import urllib.request
    import bz2

    url      = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    bz2_path = dat_path + ".bz2"

    print("⬇️ Downloading shape_predictor_68_face_landmarks.dat...")
    urllib.request.urlretrieve(url, bz2_path)

    print("📦 Extracting...")
    with bz2.open(bz2_path, "rb") as f_in, open(dat_path, "wb") as f_out:
        f_out.write(f_in.read())
    os.remove(bz2_path)

    print("✅ shape_predictor siap")
    return dat_path

# Jalankan sebelum BASE_DIR
LANDMARK_DAT = download_shape_predictor()

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR      = os.path.join(BASE_DIR, "models")
FOUNDATION_CSV = os.path.join(BASE_DIR, "foundation_mst_full_most_updated.csv")

# Kolom fitur yang sama persis dengan training
FEATURE_COLS = [
    'cheek_L_mean', 'cheek_L_std', 'cheek_a_mean', 'cheek_a_std',
    'cheek_b_mean', 'cheek_b_std', 'cheek_ITA',
    'forehead_L_mean', 'forehead_L_std', 'forehead_a_mean', 'forehead_a_std',
    'forehead_b_mean', 'forehead_b_std', 'forehead_ITA',
    'nose_L_mean', 'nose_L_std', 'nose_a_mean', 'nose_a_std',
    'nose_b_mean', 'nose_b_std', 'nose_ITA',
    'global_L_mean', 'global_L_std', 'global_a_mean', 'global_a_std',
    'global_b_mean', 'global_b_std', 'global_ITA',
]

# ─────────────────────────────────────────────
# LOAD MODEL (cached agar tidak reload tiap klik)
# ─────────────────────────────────────────────
@st.cache_resource
def load_resources():
    detector   = dlib.get_frontal_face_detector()
    predictor  = dlib.shape_predictor(LANDMARK_DAT)
    ensemble   = joblib.load(f"{MODEL_DIR}/best_model.pkl")
    scaler     = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    
    # Cari file kmeans secara dinamis
    kmeans_path = None
    for f in os.listdir(MODEL_DIR):
        if f.startswith("kmeans_k") and f.endswith(".pkl"):
            kmeans_path = os.path.join(MODEL_DIR, f)
            break
    if kmeans_path is None:
        raise FileNotFoundError("kmeans_k*.pkl tidak ditemukan di MODEL_DIR")
    kmeans = joblib.load(kmeans_path)
    
    df_found   = pd.read_csv(FOUNDATION_CSV)
    centroids  = (
        df_found.groupby("mst_id")[["lab_L", "lab_a", "lab_b"]]
        .median()
        .rename(columns={"lab_L": "L_ref", "lab_a": "a_ref", "lab_b": "b_ref"})
        .reset_index()
    )
    mst_hex_lookup = (
        df_found.drop_duplicates("mst_id")
        .set_index("mst_id")["mst_hex"]
        .to_dict()
    )
    return detector, predictor, ensemble, scaler, kmeans, df_found, centroids, mst_hex_lookup


import math

# ─────────────────────────────────────────────
# PREPROCESSING — identik dengan notebook
# ─────────────────────────────────────────────
def preprocess_image(img):
    """normalize → gaussian → wiener, sama persis dengan training."""
    # CLAHE ringan
    lab   = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16))  # ← fix
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img_norm = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    # Gaussian
    img_blur = cv2.GaussianBlur(img_norm, (5, 5), 1.0)             # ← fix ksize & sigma
    # Wiener per channel
    result = np.zeros_like(img_blur, dtype=np.float32)
    for c in range(3):
        result[:, :, c] = wiener(img_blur[:, :, c].astype(np.float32), mysize=5)  # ← fix
    return np.clip(result, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# DETEKSI LANDMARK
# ─────────────────────────────────────────────
def detect_landmarks(img_rgb, detector, predictor):
    gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = detector(gray, 1)
    if len(faces) == 0:
        faces = detector(gray, 0)   # upsample sekali lagi jika tidak ketemu
    if len(faces) == 0:
        return None, None
    face  = faces[0]
    shape = predictor(gray, face)
    lms   = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
    return lms, face


# ─────────────────────────────────────────────
# MASK HELPERS — identik dengan notebook
# ─────────────────────────────────────────────
def make_cheek_ellipse_mask(img_shape, landmarks):
    h, w   = img_shape[:2]
    mid_y  = (landmarks[27][1] + landmarks[8][1]) // 2
    face_w = landmarks[16][0] - landmarks[0][0]
    ew, eh = int(face_w * 0.22), int(face_w * 0.15)
    mask   = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (landmarks[1][0] + ew, mid_y),  (ew, eh), 0, 0, 360, 1, -1)
    cv2.ellipse(mask, (landmarks[15][0] - ew, mid_y), (ew, eh), 0, 0, 360, 1, -1)
    return mask.astype(bool)

def make_forehead_mask(img_shape, landmarks):
    h, w    = img_shape[:2]
    brow_y  = int(np.mean([landmarks[i][1] for i in range(17, 27)]))
    brow_lx = landmarks[17][0]
    brow_rx = landmarks[26][0]
    face_h  = landmarks[8][1] - landmarks[19][1]
    top_y   = max(0, brow_y - int(face_h * 0.35))
    pts  = np.array([[brow_lx, top_y], [brow_rx, top_y],
                     [brow_rx, brow_y], [brow_lx, brow_y]], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)

def make_nose_mask(img_shape, landmarks):
    h, w     = img_shape[:2]
    nose_pts = np.array([landmarks[i] for i in range(27, 36)], dtype=np.int32)
    hull     = cv2.convexHull(nose_pts)
    mask     = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [hull], 1)
    return mask.astype(bool)

def filter_skin_pixels(lab_pixels):
    """Filter pixel non-kulit — sama dengan notebook v3.1."""
    mask = (
        (lab_pixels[:, 0] >= 20) & (lab_pixels[:, 0] <= 97) &
        (lab_pixels[:, 1] >= 3)  & (lab_pixels[:, 1] <= 30)
    )
    return lab_pixels[mask]


# ─────────────────────────────────────────────
# EKSTRAKSI FITUR — identik dengan notebook
# ─────────────────────────────────────────────
def get_skin_features(img_rgb, lms):
    from skimage.color import rgb2lab as skimage_rgb2lab
    lab        = skimage_rgb2lab(img_rgb.astype(np.float32) / 255.0)
    all_pixels = []
    feats      = {}

    zones = {
        'cheek'   : make_cheek_ellipse_mask(img_rgb.shape, lms),
        'forehead': make_forehead_mask(img_rgb.shape, lms),
        'nose'    : make_nose_mask(img_rgb.shape, lms),
    }

    for zone_name, mask in zones.items():
        if mask.sum() < 10:
            for s in ['L_mean','L_std','a_mean','a_std','b_mean','b_std','ITA']:
                feats[f"{zone_name}_{s}"] = 0.0
            continue
        px = lab[mask]
        px = filter_skin_pixels(px)
        if len(px) < 5:
            for s in ['L_mean','L_std','a_mean','a_std','b_mean','b_std','ITA']:
                feats[f"{zone_name}_{s}"] = 0.0
            continue
        all_pixels.append(px)
        for ci, ch in enumerate(['L', 'a', 'b']):
            feats[f'{zone_name}_{ch}_mean'] = float(px[:, ci].mean())
            feats[f'{zone_name}_{ch}_std']  = float(px[:, ci].std())
        feats[f'{zone_name}_ITA'] = math.degrees(
            math.atan2(px[:, 0].mean(), px[:, 2].mean())  # atan2(L_mean, b_mean)
        )

    if not all_pixels:
        return None

    combined = np.vstack(all_pixels)
    for ci, ch in enumerate(['L', 'a', 'b']):
        feats[f'global_{ch}_mean'] = float(combined[:, ci].mean())
        feats[f'global_{ch}_std']  = float(combined[:, ci].std())
    feats['global_ITA'] = math.degrees(
        math.atan2(combined[:, 0].mean(), combined[:, 2].mean())
    )
    return feats


# ─────────────────────────────────────────────
# PREDIKSI HYBRID — identik dengan notebook
# ─────────────────────────────────────────────
def predict_mst_hybrid(feats, ensemble, scaler, kmeans, centroids, feature_cols,
                        alpha=0.60, temperature=1.0, sigma_eucl=4.0, sigma_ita=8.0):
    x    = np.array([[feats.get(c, 0.0) for c in feature_cols]])
    x_sc = scaler.transform(x)
    dist = kmeans.transform(x_sc)
    x_aug = np.hstack([x_sc, dist])

    model_proba   = ensemble.predict_proba(x_aug)[0]
    model_classes = ensemble.classes_

    # Temperature scaling
    log_p = np.log(model_proba + 1e-10) / temperature
    model_proba = np.exp(log_p - log_p.max())
    model_proba = model_proba / model_proba.sum()

    L_inp   = feats.get('global_L_mean', 50)
    a_inp   = feats.get('global_a_mean', 8)
    b_inp   = feats.get('global_b_mean', 12)
    ita_inp = math.degrees(math.atan2(L_inp - 50, b_inp))

    mst_keys = centroids['mst_id'].values

    dist_arr     = np.sqrt(
        (centroids['L_ref'].values - L_inp)**2 +
        (centroids['a_ref'].values - a_inp)**2 +
        (centroids['b_ref'].values - b_inp)**2
    )
    inv_dist     = np.exp(-dist_arr / sigma_eucl)
    db_proba_lab = inv_dist / inv_dist.sum()

    ita_centroids = np.degrees(np.arctan2(
        centroids['L_ref'].values - 50,
        centroids['b_ref'].values
    ))
    ita_dist     = np.abs(ita_centroids - ita_inp)
    inv_ita      = np.exp(-ita_dist / sigma_ita)
    db_proba_ita = inv_ita / inv_ita.sum()

    db_proba = 0.60 * db_proba_lab + 0.40 * db_proba_ita

    combined = {}
    for i, mst in enumerate(mst_keys):
        idx     = np.where(model_classes == mst)[0]
        model_p = float(model_proba[idx[0]]) if len(idx) > 0 else 0.0
        combined[mst] = (1 - alpha) * model_p + alpha * float(db_proba[i])

    best_mst = max(combined, key=combined.get)
    total    = sum(combined.values())

    top3_candidates = sorted(combined.items(), key=lambda x: -x[1])
    top3 = [item for item in top3_candidates if abs(item[0] - best_mst) <= 2][:3]
    if len(top3) < 3:
        remaining = [item for item in top3_candidates if item not in top3]
        top3 += sorted(remaining, key=lambda x: abs(x[0] - best_mst))[:3 - len(top3)]

    return (
        int(best_mst),
        round(combined[best_mst] / total * 100, 1),
        [{'mst': int(m), 'conf': round(p / total * 100, 1)} for m, p in top3]
    )


# ─────────────────────────────────────────────
# REKOMENDASI — identik dengan notebook
# ─────────────────────────────────────────────
def recommend_foundation(mst_pred, L, a, b, df_found, top_n=5):
    df = df_found.copy()
    df['delta_e'] = np.sqrt(
        (df['lab_L'] - L)**2 +
        (df['lab_a'] - a)**2 +
        (df['lab_b'] - b)**2
    )
    mst_range  = [mst_pred - 1, mst_pred, mst_pred + 1]
    df_primary = df[df['mst_id'].isin(mst_range)].sort_values('delta_e')
    df_fallback= df[~df['mst_id'].isin(mst_range)].sort_values('delta_e')
    return pd.concat([df_primary, df_fallback]).head(top_n).reset_index(drop=True)


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(img_rgb, detector, predictor, ensemble, scaler,
                 kmeans, centroids, df_found, mst_hex_lookup, feature_cols):
    t0 = time.time()

    h, w = img_rgb.shape[:2]
    if max(h, w) > 512:
        scale   = 512 / max(h, w)
        img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)))

    img_pre        = preprocess_image(img_rgb)
    lms, face_rect = detect_landmarks(img_pre, detector, predictor)

    if lms is None:
        return None, "❌ Wajah tidak terdeteksi. Pastikan pencahayaan cukup dan wajah menghadap kamera."

    feats = get_skin_features(img_pre, lms)
    if feats is None:
        return None, "❌ Ekstraksi fitur gagal. Wajah terlalu kecil atau terhalang."

    mst, conf, top3 = predict_mst_hybrid(
        feats, ensemble, scaler, kmeans, centroids, feature_cols
    )

    top3_hex = [{"mst": t["mst"], "conf": t["conf"],
                 "hex": mst_hex_lookup.get(t["mst"], "#888888")} for t in top3]

    recs    = recommend_foundation(
        mst, feats["global_L_mean"], feats["global_a_mean"], feats["global_b_mean"],
        df_found, top_n=5
    )
    top_rec = recs.iloc[0]
    latency = round((time.time() - t0) * 1000, 1)

    vis = img_rgb.copy()
    if face_rect is not None:
        x1, y1, x2, y2 = face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 100), 2)
    for (px, py) in lms:
        cv2.circle(vis, (int(px), int(py)), 1, (255, 100, 0), -1)

    return {
        "mst_pred"  : mst,
        "confidence": conf,
        "top3"      : top3_hex,
        "shade_name": top_rec["Shade"],
        "brand"     : top_rec["Brand"],
        "product"   : top_rec["Product"],
        "hex_color" : top_rec["mst_hex"],
        "undertone" : top_rec["Undertone"],
        "delta_e"   : round(top_rec["delta_e"], 2),
        "top5_recs" : recs.to_dict(orient="records"),
        "cielab"    : {
            "L": round(feats["global_L_mean"], 2),
            "a": round(feats["global_a_mean"], 2),
            "b": round(feats["global_b_mean"], 2),
        },
        "latency_ms": latency,
        "vis_frame" : vis,
    }, None

# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Foundation Shade Detector",
        page_icon="🎨",
        layout="wide",
    )

    st.title("🎨 Foundation Shade Detector")
    st.caption("Monk Skin Tone (MST) Detection + Rekomendasi Foundation via Webcam")

    # ── Load resources ──
    with st.spinner("Memuat model & database foundation..."):
        try:
            (detector, predictor, ensemble, scaler,
             kmeans, df_found, centroids, mst_hex_lookup) = load_resources()
            st.success(f"✅ Model siap | Foundation DB: {len(df_found)} produk")
        except Exception as e:
            st.error(f"❌ Gagal load model: {e}")
            st.stop()

    # ── Sidebar: info & settings ──
    with st.sidebar:
        st.header("ℹ️ Petunjuk")
        st.markdown("""
        1. Klik **'Ambil Foto'** di bawah
        2. Izinkan akses kamera jika diminta
        3. Pastikan wajah terlihat jelas & cahaya cukup
        4. Klik tombol kamera untuk mengambil foto
        5. Hasil prediksi MST & rekomendasi foundation akan muncul
        """)
        st.divider()
        st.markdown("**Tentang MST (Monk Skin Tone)**")
        st.markdown("Skala 1–10 untuk mengukur warna kulit secara inklusif, "
                    "dikembangkan oleh Dr. Ellis Monk (Google).")
        
        # Warna referensi MST
        st.markdown("**Referensi Warna MST:**")
        mst_colors = {
            1: "#f6ede4", 2: "#f3e7db", 3: "#f7ead0", 4: "#eadaba",
            5: "#d7bd96", 6: "#a07850", 7: "#825c43", 8: "#604134",
            9: "#3a312a", 10: "#292420"
        }
        cols_mst = st.columns(5)
        for i, (mst_id, hex_c) in enumerate(mst_colors.items()):
            with cols_mst[i % 5]:
                st.markdown(
                    f'<div style="background:{hex_c};border-radius:6px;'
                    f'height:28px;display:flex;align-items:center;'
                    f'justify-content:center;color:{"#000" if i < 5 else "#fff"};'
                    f'font-size:11px;font-weight:bold">MST {mst_id}</div>',
                    unsafe_allow_html=True
                )

    # ── Webcam input ──
    st.subheader("📷 Kamera")
    camera_image = st.camera_input(
        "Ambil foto wajah",
        help="Klik tombol kamera di bawah preview untuk mengambil foto"
    )

    if camera_image is not None:
        # Decode gambar dari file upload
        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        st.subheader("🔍 Hasil Analisis")

        with st.spinner("Menganalisis wajah..."):
            result, error = run_pipeline(
                img_rgb, detector, predictor, ensemble, scaler,
                kmeans, centroids, df_found, mst_hex_lookup, FEATURE_COLS
            )

        if error:
            st.warning(error)
        else:
            # ── Layout hasil: 3 kolom ──
            col1, col2, col3 = st.columns([1.2, 1, 1.5])

            # Kolom 1: foto dengan landmark
            with col1:
                st.markdown("**Frame + Landmark**")
                st.image(result["vis_frame"], use_column_width=True)

            # Kolom 2: swatch warna + MST info
            with col2:
                st.markdown("**Prediksi MST**")

                # Swatch warna kulit
                skin_hex = cielab_to_hex(
                    result["cielab"]["L"],
                    result["cielab"]["a"],
                    result["cielab"]["b"]
                )
                st.markdown(
                    f'<div style="background:{skin_hex};border-radius:10px;'
                    f'height:50px;margin-bottom:6px;border:1px solid #ccc"></div>'
                    f'<p style="text-align:center;font-size:12px;margin-top:0">Warna Kulit Terdeteksi<br>{skin_hex}</p>',
                    unsafe_allow_html=True
                )

                # Swatch foundation rekomendasi
                st.markdown(
                    f'<div style="background:{result["hex_color"]};border-radius:10px;'
                    f'height:50px;margin-bottom:6px;border:1px solid #ccc"></div>'
                    f'<p style="text-align:center;font-size:12px;margin-top:0">Foundation Cocok<br>{result["hex_color"]}</p>',
                    unsafe_allow_html=True
                )

                # MST badge
                st.markdown(
                    f'<div style="text-align:center;background:#f0f0f0;'
                    f'border-radius:12px;padding:12px;margin:8px 0">'
                    f'<span style="font-size:36px;font-weight:bold">MST {result["mst_pred"]}</span><br>'
                    f'<span style="font-size:14px;color:#666">Confidence: {result["confidence"]}%</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # Confidence bar
                conf_pct = result["confidence"]
                bar_color = "#2e8b57" if conf_pct >= 60 else "#e07b39" if conf_pct >= 40 else "#cc2222"
                st.markdown(
                    f'<div style="background:#eee;border-radius:6px;height:12px;overflow:hidden">'
                    f'<div style="background:{bar_color};width:{conf_pct}%;height:100%;border-radius:6px"></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # Top-3 alternatif
                st.markdown("**Top-3 Alternatif MST:**")
                for t in result["top3"]:
                    hex_c = t["hex"]
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0">'
                        f'<div style="background:{hex_c};width:28px;height:28px;border-radius:5px;'
                        f'flex-shrink:0;border:1px solid #ccc"></div>'
                        f'<span style="font-size:13px">MST {t["mst"]} — {t["conf"]}%</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            # Kolom 3: detail rekomendasi + CIELAB
            with col3:
                st.markdown("**Rekomendasi Foundation**")
                st.markdown(f"""
                | Info | Detail |
                |------|--------|
                | 🏷️ Brand | **{result['brand']}** |
                | 💄 Produk | {result['product']} |
                | 🎨 Shade | **{result['shade_name']}** |
                | 🌡️ Undertone | {result['undertone']} |
                | 📐 Delta E | {result['delta_e']} |
                | ⚡ Latency | {result['latency_ms']} ms |
                """)

                st.markdown("**Nilai CIELAB Kulit:**")
                c1, c2, c3 = st.columns(3)
                c1.metric("L* (kecerahan)", result["cielab"]["L"])
                c2.metric("a* (merah-hijau)", result["cielab"]["a"])
                c3.metric("b* (kuning-biru)", result["cielab"]["b"])

                st.markdown("**Top-5 Rekomendasi Foundation:**")
                df_recs = pd.DataFrame(result["top5_recs"])[
                    ["Brand", "Product", "Shade", "Undertone", "delta_e"]
                ].rename(columns={"delta_e": "ΔE"})
                df_recs["ΔE"] = df_recs["ΔE"].round(2)
                st.dataframe(df_recs, use_container_width=True, hide_index=True)

            # ── Info latency di bawah ──
            st.caption(f"⏱️ Waktu analisis: {result['latency_ms']} ms")


if __name__ == "__main__":
    main()
