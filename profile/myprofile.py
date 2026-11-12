import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.ndimage as nd
from PIL import Image

st.set_page_config(
    page_title="Auto Object Counting Web App (HW#5)",
    layout="wide"
)

st.title("🤖 완전 자동화 Object Counting 웹 애플리케이션")
st.markdown(
    "이미지를 업로드하면 **각 이미지의 조명 분포와 객체 크기(반경)를 자동으로 분석**하여, 별도의 파라미터 조작 없이 전처리·HPF 에지 분리·Opening·객체 카운팅을 수행합니다."
)

# ---------------------------------------------------------
# 1. 기초 유틸리티 함수
# ---------------------------------------------------------
def compute_otsu_threshold(image_uint8):
    hist, _ = np.histogram(image_uint8.flatten(), bins=256, range=(0, 256))
    total = image_uint8.size
    sum_total = np.dot(np.arange(256), hist)

    sum_bg, weight_bg, max_var, threshold = 0.0, 0, 0.0, 127
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    return threshold

def create_disk_structure(radius):
    if radius <= 0:
        return nd.generate_binary_structure(2, 1)
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x ** 2 + y ** 2) <= (radius ** 2)

def fill_internal_holes(binary, max_hole_ratio=0.02):
    inv = ~binary
    labeled_holes, num_holes = nd.label(inv)
    if num_holes == 0:
        return binary
    hole_sizes = np.bincount(labeled_holes.ravel())
    max_hole_px = int(binary.size * max_hole_ratio)
    border_labels = np.unique(np.concatenate([
        labeled_holes[0, :], labeled_holes[-1, :],
        labeled_holes[:, 0], labeled_holes[:, -1]
    ]))
    small_holes = hole_sizes < max_hole_px
    small_holes[border_labels] = False
    return binary | small_holes[labeled_holes]

def resize_keep_aspect(img_pil, max_dim=512):
    w, h = img_pil.size
    if max(w, h) <= max_dim:
        return img_pil
    scale = max_dim / float(max(w, h))
    return img_pil.resize((int(round(w * scale)), int(round(h * scale))), Image.Resampling.BILINEAR)

# ---------------------------------------------------------
# 2. HPF (Sobel + Laplacian) 에지 추출 함수
# ---------------------------------------------------------
def extract_hpf_edges(f_img):
    f_smooth = nd.gaussian_filter(f_img, sigma=1.0)

    # 1차 미분 Sobel Gradient
    gx = nd.sobel(f_smooth, axis=1)
    gy = nd.sobel(f_smooth, axis=0)
    sobel_mag = np.hypot(gx, gy)
    sobel_norm = (sobel_mag / sobel_mag.max() * 255.0) if sobel_mag.max() > 0 else np.zeros_like(f_img)

    # 2차 미분 Laplacian HPF
    lap_kernel = np.array([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]], dtype=np.float64)
    lap_resp = np.abs(signal.convolve2d(f_smooth, lap_kernel, mode="same", boundary="symm"))
    p99 = np.percentile(lap_resp, 99.0)
    lap_norm = np.clip(lap_resp / p99 * 255.0, 0, 255) if p99 > 0 else np.zeros_like(f_img)

    edge_map = np.clip(0.65 * sobel_norm + 0.35 * lap_norm, 0, 255)
    return edge_map.astype(np.uint8)

# ---------------------------------------------------------
# 3. 2-Pass 완전 자동 파라미터 추정 및 객체 카운팅 파이프라인
# ---------------------------------------------------------
def process_image_auto(img_pil, manual_override=None):
    # [Step 1] Grayscale 변환 및 해상도 표준화
    gray_full = img_pil.convert("RGB").convert("L")
    gray_resized = resize_keep_aspect(gray_full, max_dim=512)
    f_raw = np.array(gray_resized, dtype=np.float64)

    # [Step 2] 배경/객체 극성(Polarity) 자동 판별
    # 넓게 블러링한 배경 대비 양(+)의 편차와 음(-)의 편차 에너지를 비교
    bg_rough = nd.gaussian_filter(f_raw, sigma=35.0)
    pos_dev = np.percentile(np.clip(f_raw - bg_rough, 0, None), 98)
    neg_dev = np.percentile(np.clip(bg_rough - f_raw, 0, None), 98)
    is_inverted = bool(np.median(f_raw) > 127 and neg_dev > pos_dev * 0.8)

    if manual_override and manual_override["polarity"] != "자동":
        is_inverted = (manual_override["polarity"] == "반전 (밝은 배경)")

    f = (255.0 - f_raw) if is_inverted else f_raw.copy()

    # [Step 3] 1차 패스 (Rough Pass): 이미지 내 객체들의 평균 반경(R_obj) 자동 측정
    f_med_init = signal.medfilt2d(f, kernel_size=5)
    fg_init = np.clip(f_med_init - nd.gaussian_filter(f_med_init, sigma=35.0), 0, None)
    p99_init = np.percentile(fg_init, 99.5)
    fg_init_norm = np.clip(fg_init / p99_init * 255.0, 0, 255).astype(np.uint8) if p99_init > 0 else fg_init.astype(np.uint8)
    
    t_init = max(compute_otsu_threshold(fg_init_norm), 35)
    bw_init = fill_internal_holes(fg_init_norm > t_init)
    edt_init = nd.distance_transform_edt(bw_init)
    
    # 객체 내부 거리 변환의 상위 85% 값을 해당 이미지의 대표 객체 반경(R_obj)으로 산출
    fg_distances = edt_init[edt_init > 0]
    r_obj = float(np.percentile(fg_distances, 85)) if len(fg_distances) > 0 else 6.0
    r_obj = np.clip(r_obj, 2.5, 40.0)

    # [Step 4] 측정된 객체 반경(R_obj) 기반 최적 파라미터 자동 계산
    auto_med_k = 3 if r_obj < 5.5 else (5 if r_obj < 12.0 else 7)
    auto_bg_sigma = float(np.clip(r_obj * 3.2, 20.0, 75.0))
    auto_open_radius = int(np.clip(round(r_obj * 0.35), 1, 5))
    auto_hpf_weight = float(np.clip(0.35 + (r_obj / 40.0) * 0.25, 0.35, 0.65))

    if manual_override and not manual_override["use_auto"]:
        auto_med_k = manual_override["med_k"]
        auto_bg_sigma = manual_override["bg_sigma"]
        auto_open_radius = manual_override["open_r"]
        auto_hpf_weight = manual_override["hpf_w"]

    # [Step 5] 2차 본 패스: 최적화된 커널로 노이즈 제거 및 불균일 조명 평탄화
    f_denoised = signal.medfilt2d(f, kernel_size=auto_med_k)
    bg_est = nd.gaussian_filter(f_denoised, sigma=auto_bg_sigma)
    f_corrected = np.clip(f_denoised - bg_est, 0, None)

    low_v, high_v = np.percentile(f_corrected, 2.0), np.percentile(f_corrected, 99.5)
    if high_v > low_v:
        f_norm = np.clip((f_corrected - low_v) / (high_v - low_v) * 255.0, 0, 255).astype(np.uint8)
    else:
        f_norm = np.clip(f_corrected, 0, 255).astype(np.uint8)

    # [Step 6] HPF 에지 추출 및 객체 접합부 차감
    edge_map = extract_hpf_edges(f_norm.astype(np.float64))
    f_carved = np.clip(
        f_norm.astype(np.float64) - auto_hpf_weight * edge_map.astype(np.float64),
        0, 255
    ).astype(np.uint8)

    # [Step 7] 적응형 임계값 산출 (Otsu + 배경 노이즈 바닥 하한선 결합)
    otsu_t = compute_otsu_threshold(f_norm)
    noise_floor_t = int(np.mean(f_carved) + 0.85 * np.std(f_carved))
    thresh_val = int(np.clip(max(otsu_t, noise_floor_t), 25, 230))

    if manual_override and not manual_override["use_auto"]:
        thresh_val = int(np.clip(otsu_t + manual_override["t_offset"], 5, 250))

    binary = f_carved > thresh_val
    binary = fill_internal_holes(binary, max_hole_ratio=0.02)

    # [Step 8] 자동 산출된 원형(Disk) 커널로 Morphological Opening 수행 (붙은 객체 1차 분리)
    disk_struct = create_disk_structure(auto_open_radius)
    binary_opened = nd.binary_opening(binary, structure=disk_struct, iterations=1)

    # [Step 9] Opening 후에도 넓게 붙어 있는 덩어리만 선별하여 거리 피크(Watershed)로 2차 분리
    label_struct = nd.generate_binary_structure(2, 2)
    lbl_temp, n_temp = nd.label(binary_opened, structure=label_struct)
    if n_temp > 1:
        dist_map = nd.distance_transform_edt(binary_opened)
        peak_size = max(5, int(round(r_obj * 1.35)))
        dist_smooth = nd.gaussian_filter(dist_map, sigma=max(1.0, r_obj * 0.25))
        local_max = (dist_smooth == nd.maximum_filter(dist_smooth, size=peak_size)) & (dist_map > r_obj * 0.45)
        markers, num_seeds = nd.label(local_max, structure=label_struct)

        if num_seeds > n_temp:
            markers_ws = markers.astype(np.int32)
            markers_ws[~binary_opened] = num_seeds + 1
            cost = (255 - (dist_smooth / (dist_smooth.max() + 1e-6) * 255)).astype(np.uint8)
            ws = nd.watershed_ift(cost, markers_ws)
            ws_fg = np.where((ws >= 1) & (ws <= num_seeds), ws, 0)
            max_nb = nd.maximum_filter(ws_fg, size=3)
            min_nb = nd.minimum_filter(np.where(ws_fg > 0, ws_fg, num_seeds + 2), size=3)
            ridge = (ws_fg > 0) & (max_nb != min_nb)
            binary_opened[ridge] = False

    # [Step 10] 최종 라벨링 및 중앙값 면적 대비 자동 노이즈 제거
    labeled_array, num_features = nd.label(binary_opened, structure=label_struct)
    min_abs_px = max(8, int(np.pi * (r_obj ** 2) * 0.18))

    if num_features > 0:
        sizes = np.bincount(labeled_array.ravel())
        fg_sizes = sizes[1:]
        valid_fg = fg_sizes[fg_sizes >= min_abs_px]
        median_area = np.median(valid_fg) if len(valid_fg) > 0 else np.median(fg_sizes)
        min_allowed_area = max(min_abs_px, median_area * 0.22)

        keep_mask = sizes >= min_allowed_area
        keep_mask[0] = False
        binary_final = keep_mask[labeled_array]
        labeled_array, num_features = nd.label(binary_final, structure=label_struct)
    else:
        binary_final = binary_opened

    centroids = []
    if num_features > 0:
        centroids = nd.center_of_mass(binary_final, labeled_array, range(1, num_features + 1))

    return {
        "original_gray": np.array(gray_resized),
        "corrected": f_norm,
        "edge_map": edge_map,
        "binary_final": binary_final,
        "labeled": labeled_array,
        "count": num_features,
        "centroids": centroids,
        "auto_info": {
            "is_inverted": is_inverted,
            "r_obj": round(r_obj, 1),
            "med_k": auto_med_k,
            "bg_sigma": round(auto_bg_sigma, 1),
            "hpf_weight": round(auto_hpf_weight, 2),
            "thresh": thresh_val,
            "open_r": auto_open_radius
        }
    }

# ---------------------------------------------------------
# 4. 사이드바: 기본 완전 자동 모드 (선택적 수동 제어 제공)
# ---------------------------------------------------------
st.sidebar.header("🤖 작동 모드 설정")
use_auto = st.sidebar.toggle("✨ 100% 완전 자동화 모드 (권장)", value=True)

manual_override = {"use_auto": use_auto, "polarity": "자동"}
if not use_auto:
    st.sidebar.subheader("수동 파라미터 조절")
    manual_override["polarity"] = st.sidebar.selectbox("극성 선택", ["자동", "원본 (어두운 배경)", "반전 (밝은 배경)"])
    manual_override["med_k"] = st.sidebar.slider("Median 커널", 1, 11, 5, 2)
    manual_override["bg_sigma"] = st.sidebar.slider("배경 Sigma", 15.0, 80.0, 35.0, 5.0)
    manual_override["hpf_w"] = st.sidebar.slider("HPF 차감 강도", 0.0, 1.2, 0.45, 0.05)
    manual_override["open_r"] = st.sidebar.slider("Opening 반경", 1, 6, 2, 1)
    manual_override["t_offset"] = st.sidebar.slider("임계값 Offset", -40, 40, 0, 1)
else:
    st.sidebar.success(
        "현재 **완전 자동 모드**입니다.\n\n"
        "업로드된 각 이미지의 **배경 밝기, 객체 평균 반경, 에지 강도**를 실시간으로 분석해 최적의 필터 크기를 자동 적용합니다."
    )

# ---------------------------------------------------------
# 5. 메인 화면: 다중 이미지 업로드 및 자동 카운팅 출력
# ---------------------------------------------------------
uploaded_files = st.file_uploader(
    "카운팅할 이미지 파일들을 업로드하세요 (여러 장 동시 선택 가능)",
    type=["tif", "tiff", "png", "jpg", "jpeg", "bmp"],
    accept_multiple_files=True
)

if uploaded_files:
    summary_data = []
    results_list = []

    for uploaded_file in uploaded_files:
        img_pil = Image.open(uploaded_file)
        res = process_image_auto(img_pil, manual_override)
        results_list.append((uploaded_file.name, img_pil, res))
        info = res["auto_info"]

        summary_data.append({
            "파일명": uploaded_file.name,
            "원본 해상도": f"{img_pil.width} x {img_pil.height}",
            "검출된 객체 수 (개)": res["count"],
            "자동 판별 배경": "밝은 배경 (반전)" if info["is_inverted"] else "어두운 배경 (원본)",
            "추정 객체 반경 (px)": info["r_obj"],
            "자동 Opening 반경": f"R = {info['open_r']}",
            "자동 임계값 (T)": info["thresh"]
        })

    # 전체 요약표를 최상단에 배치하여 여러 이미지 결과를 한눈에 확인
    st.subheader("📊 전체 이미지 자동 Object Counting 결과 요약")
    st.dataframe(summary_data, use_container_width=True)

    st.divider()
    st.subheader("🖼️ 이미지별 5단계 처리 과정 및 자동 산출 파라미터")
    tabs = st.tabs([item[0] for item in results_list])

    for tab, (fname, img_pil, res) in zip(tabs, results_list):
        with tab:
            info = res["auto_info"]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("검출된 총 객체 수", f"{res['count']} 개")
            c2.metric("추정 객체 반경", f"{info['r_obj']} px")
            c3.metric("자동 Opening 커널", f"R = {info['open_r']} (Median {info['med_k']}x{info['med_k']})")
            c4.metric("자동 임계값 / HPF", f"T = {info['thresh']} (w={info['hpf_weight']})")
            c5.metric("자동 극성 판별", "밝은 배경 (반전)" if info["is_inverted"] else "어두운 배경 (유지)")

            fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))

            axes[0].imshow(res["original_gray"], cmap="gray")
            axes[0].set_title("1. Grayscale Input")
            axes[0].axis("off")

            axes[1].imshow(res["corrected"], cmap="gray")
            axes[1].set_title(f"2. Auto Bg Flat (σ={info['bg_sigma']})")
            axes[1].axis("off")

            axes[2].imshow(res["edge_map"], cmap="inferno")
            axes[2].set_title(f"3. HPF Edge (w={info['hpf_weight']})")
            axes[2].axis("off")

            axes[3].imshow(res["binary_final"], cmap="gray")
            axes[3].set_title(f"4. Auto Open (R={info['open_r']}, T={info['thresh']})")
            axes[3].axis("off")

            axes[4].imshow(res["labeled"], cmap="nipy_spectral")
            axes[4].set_title(f"5. Counted: {res['count']}")
            axes[4].axis("off")

            for cy, cx in res["centroids"]:
                axes[4].plot(cx, cy, "w+", markersize=6, markeredgewidth=1.2)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
else:
    st.info("👆 상단의 'Browse files' 버튼을 눌러 분석할 이미지들을 업로드하면 조작 없이 자동으로 카운팅됩니다.")
