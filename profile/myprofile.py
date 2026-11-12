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
    "밀집 객체 보존형 배경 평탄화, 감마 대비 보정, **중심부 보호형 HPF 에지 절단 + Opening + 거리 변환 분할**을 통해 다양한 조건의 이미지를 자동으로 카운팅합니다."
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

def fill_internal_holes(binary, max_hole_ratio=0.03):
    """객체 내부의 무늬나 반사로 인해 뚫린 구멍만 메우고, 배경 틈새는 유지"""
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
# 2. 밀집 객체 보존형 배경 조명 추정 함수
# ---------------------------------------------------------
def estimate_background_illumination(f_denoised, window_scale=85):
    """
    가우시안 평균 대신 하위 10% 백분위수(Percentile)를 사용해,
    동전처럼 객체가 빽빽하게 밀집한 이미지에서도 객체가 지워지지 않고 실제 배경만 추출
    """
    h, w = f_denoised.shape
    # 연산 속도와 대구경 수용 영역을 위해 1/4로 다운샘플링하여 배경 포락선 추출
    step = 4
    f_small = f_denoised[::step, ::step]
    win_small = max(15, int(round(window_scale / step)))
    if win_small % 2 == 0:
        win_small += 1

    # 각 영역의 하위 10% 밝기(객체 사이 틈새 배경) 추출 후 부드럽게 스무딩
    bg_small = nd.percentile_filter(f_small, percentile=10, size=win_small)
    bg_small = nd.gaussian_filter(bg_small, sigma=win_small * 0.3)

    # 원래 해상도로 복원
    bg_pil = Image.fromarray(bg_small.astype(np.float32)).resize((w, h), Image.Resampling.BILINEAR)
    return np.array(bg_pil, dtype=np.float64)

# ---------------------------------------------------------
# 3. HPF (Sobel + Laplacian) 에지 추출 함수
# ---------------------------------------------------------
def extract_hpf_edges(f_img):
    f_smooth = nd.gaussian_filter(f_img, sigma=1.2)

    # Sobel 1차 미분 경계
    gx = nd.sobel(f_smooth, axis=1)
    gy = nd.sobel(f_smooth, axis=0)
    sobel_mag = np.hypot(gx, gy)
    sobel_norm = (sobel_mag / (np.percentile(sobel_mag, 99.0) + 1e-6) * 255.0)
    sobel_norm = np.clip(sobel_norm, 0, 255)

    # Laplacian 2차 미분 고주파 (객체들이 맞닿은 어두운 골짜기 강조)
    lap_kernel = np.array([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]], dtype=np.float64)
    lap_resp = signal.convolve2d(f_smooth, lap_kernel, mode="same", boundary="symm")
    # 밝은 객체 사이의 어두운 경계 골짜기(음의 응답)와 강한 에지 결합
    valley_resp = np.clip(-lap_resp, 0, None)
    v99 = np.percentile(valley_resp, 99.0)
    valley_norm = np.clip(valley_resp / (v99 + 1e-6) * 255.0, 0, 255)

    edge_map = np.clip(0.55 * sobel_norm + 0.45 * valley_norm, 0, 255)
    return edge_map.astype(np.uint8)

# ---------------------------------------------------------
# 4. 완전 자동화 객체 카운팅 파이프라인
# ---------------------------------------------------------
def process_image_auto(img_pil, manual_override=None):
    # [Step 1] Grayscale 변환 및 해상도 표준화
    gray_full = img_pil.convert("RGB").convert("L")
    gray_resized = resize_keep_aspect(gray_full, max_dim=512)
    f_raw = np.array(gray_resized, dtype=np.float64)

    # [Step 2] 테두리 및 전체 밝기 분포를 이용한 극성(Polarity) 자동 판별
    h, w = f_raw.shape
    pad_h, pad_w = max(2, int(h * 0.04)), max(2, int(w * 0.04))
    border_pixels = np.concatenate([
        f_raw[:pad_h, :].ravel(), f_raw[-pad_h:, :].ravel(),
        f_raw[:, :pad_w].ravel(), f_raw[:, -pad_w:].ravel()
    ])
    border_median = np.median(border_pixels)
    overall_otsu = compute_otsu_threshold(f_raw.astype(np.uint8))
    # 테두리 중앙값이 밝으면(>127) 밝은 배경 위 어두운 객체로 판별
    is_inverted = bool(border_median > max(125, overall_otsu))

    if manual_override and manual_override["polarity"] != "자동":
        is_inverted = (manual_override["polarity"] == "반전 (밝은 배경)")

    f = (255.0 - f_raw) if is_inverted else f_raw.copy()

    # [Step 3] 노이즈 제거 및 밀집 객체 보존형 배경 평탄화
    f_med_init = signal.medfilt2d(f, kernel_size=5)
    bg_init = estimate_background_illumination(f_med_init, window_scale=95)
    f_corr_init = np.clip(f_med_init - bg_init, 0, None)

    p99 = np.percentile(f_corr_init, 99.5)
    f_norm_linear = np.clip(f_corr_init / (p99 + 1e-6) * 255.0, 0, 255).astype(np.uint8)

    # 감마 보정(Gamma = 0.65)으로 옅은 회색 동전 등 저대비 객체의 밝기를 끌어올림
    f_norm = np.clip(((f_norm_linear / 255.0) ** 0.65) * 255.0, 0, 255).astype(np.uint8)

    # [Step 4] 적응형 임계값 자동 산출 (배경 노이즈 분포 기반)
    otsu_t = compute_otsu_threshold(f_norm)
    bg_sample = f_norm[f_norm <= otsu_t]
    bg_noise_limit = float(np.percentile(bg_sample, 95)) + 12.0 if len(bg_sample) > 0 else 30.0
    # Otsu가 밝기 편차 큰 객체들 사이를 가르는 것을 방지하기 위해 보수적 임계값 선택
    thresh_val = int(np.clip(min(otsu_t * 0.85, max(otsu_t * 0.65, bg_noise_limit)), 22, 165))

    if manual_override and not manual_override["use_auto"]:
        thresh_val = int(np.clip(otsu_t + manual_override["t_offset"], 5, 250))

    # [Step 5] 내부 무늬로 인한 깨짐 방지를 위해 '속이 꽉 찬 전경 마스크(Solid Mask)' 선행 생성
    binary_solid = f_norm > thresh_val
    binary_solid = fill_internal_holes(binary_solid, max_hole_ratio=0.03)
    # 미세한 내부 갈라짐 봉합
    binary_solid = nd.binary_closing(binary_solid, structure=create_disk_structure(1), iterations=1)
    binary_solid = fill_internal_holes(binary_solid, max_hole_ratio=0.03)

    # [Step 6] 거리 변환으로 이미지 내 객체들의 대표 반경(R_obj) 자동 측정
    edt_full = nd.distance_transform_edt(binary_solid)
    # 국소 최대값(Local Maxima) 위치의 거리값들을 샘플링해 실제 객체 반경 추정
    local_peaks = (edt_full == nd.maximum_filter(edt_full, size=9)) & (edt_full > 2.0)
    peak_radii = edt_full[local_peaks]
    r_obj = float(np.median(peak_radii)) if len(peak_radii) > 0 else 8.0
    r_obj = float(np.clip(r_obj, 3.0, 45.0))

    auto_open_radius = int(np.clip(round(r_obj * 0.28), 1, 4))
    auto_hpf_weight = 0.55

    if manual_override and not manual_override["use_auto"]:
        auto_open_radius = manual_override["open_r"]
        auto_hpf_weight = manual_override["hpf_w"]

    # [Step 7] HPF 에지 추출 및 '외곽 접합부(Neck) 한정의 선택적 에지 절단'
    edge_map = extract_hpf_edges(f_norm.astype(np.float64))

    # 핵심: 동전 중심부(edt_full > r_obj * 0.55)는 내부 무늬에 깨지지 않도록 100% 보호하고,
    # 객체들이 맞닿는 외곽 경계 부근(0 < edt_full <= r_obj * 0.55)에서만 HPF 에지로 깎아냄
    neck_zone = (edt_full > 0) & (edt_full <= r_obj * 0.55)
    f_carved = f_norm.astype(np.float64).copy()
    f_carved[neck_zone] -= (auto_hpf_weight * 1.3) * edge_map[neck_zone].astype(np.float64)

    binary_carved = binary_solid.copy()
    binary_carved[neck_zone] = f_carved[neck_zone] > thresh_val

    # [Step 8] 원형(Disk) 커널 기반 Morphological Opening (좁은 접합부 1차 분리)
    disk_struct = create_disk_structure(auto_open_radius)
    binary_opened = nd.binary_opening(binary_carved, structure=disk_struct, iterations=1)
    binary_opened = fill_internal_holes(binary_opened, max_hole_ratio=0.02)

    # [Step 9] HPF 에지 골짜기가 반영된 거리 변환 Watershed (밀집 동전 2차 정밀 분리)
    label_struct = nd.generate_binary_structure(2, 2)
    dist_opened = nd.distance_transform_edt(binary_opened)
    # 거리 변환 값에서 HPF 에지 강도를 감산하여, 맞닿은 동전 경계에 깊은 계곡 형성
    edge_penalty = np.clip(edge_map.astype(np.float64) / 255.0, 0, 1)
    dist_guided = dist_opened * (1.0 - 0.35 * edge_penalty)
    dist_smooth = nd.gaussian_filter(dist_guided, sigma=max(1.2, r_obj * 0.22))

    # 각 객체의 중심 시드(Seed) 검출
    seed_footprint = max(5, int(round(r_obj * 1.25)))
    local_max = (dist_smooth == nd.maximum_filter(dist_smooth, size=seed_footprint)) & (dist_opened > r_obj * 0.38)
    # 시드들이 너무 가까우면 하나로 병합
    local_max = nd.binary_dilation(local_max, structure=create_disk_structure(max(1, int(r_obj * 0.2))))
    markers, num_seeds = nd.label(local_max, structure=label_struct)

    if num_seeds > 1:
        markers_ws = markers.astype(np.int32)
        markers_ws[~binary_opened] = num_seeds + 1
        cost = (255 - np.clip(dist_smooth / (dist_smooth.max() + 1e-6) * 255.0, 0, 255)).astype(np.uint8)
        ws = nd.watershed_ift(cost, markers_ws)
        ws_fg = np.where((ws >= 1) & (ws <= num_seeds), ws, 0)

        # 서로 다른 라벨이 맞닿는 경계선(Watershed Ridge) 분리
        max_nb = nd.maximum_filter(ws_fg, size=3)
        min_nb = nd.minimum_filter(np.where(ws_fg > 0, ws_fg, num_seeds + 2), size=3)
        ridge = (ws_fg > 0) & (max_nb != min_nb)
        binary_separated = binary_opened & (~ridge)
    else:
        binary_separated = binary_opened

    # [Step 10] 최종 라벨링 및 중앙값 면적 대비 자동 노이즈 필터링
    labeled_array, num_features = nd.label(binary_separated, structure=label_struct)
    min_abs_px = max(8, int(np.pi * (r_obj ** 2) * 0.15))

    if num_features > 0:
        sizes = np.bincount(labeled_array.ravel())
        fg_sizes = sizes[1:]
        valid_fg = fg_sizes[fg_sizes >= min_abs_px]
        median_area = np.median(valid_fg) if len(valid_fg) > 0 else np.median(fg_sizes)
        min_allowed_area = max(min_abs_px, median_area * 0.20)

        keep_mask = sizes >= min_allowed_area
        keep_mask[0] = False
        binary_final = keep_mask[labeled_array]
        labeled_array, num_features = nd.label(binary_final, structure=label_struct)
    else:
        binary_final = binary_separated

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
            "hpf_weight": round(auto_hpf_weight, 2),
            "thresh": thresh_val,
            "open_r": auto_open_radius
        }
    }

# ---------------------------------------------------------
# 5. 사이드바 UI
# ---------------------------------------------------------
st.sidebar.header("🤖 작동 모드 설정")
use_auto = st.sidebar.toggle("✨ 100% 완전 자동화 모드 (권장)", value=True)

manual_override = {"use_auto": use_auto, "polarity": "자동"}
if not use_auto:
    st.sidebar.subheader("수동 파라미터 조절")
    manual_override["polarity"] = st.sidebar.selectbox("극성 선택", ["자동", "원본 (어두운 배경)", "반전 (밝은 배경)"])
    manual_override["hpf_w"] = st.sidebar.slider("HPF 차감 강도", 0.0, 1.2, 0.55, 0.05)
    manual_override["open_r"] = st.sidebar.slider("Opening 반경", 1, 6, 2, 1)
    manual_override["t_offset"] = st.sidebar.slider("임계값 Offset", -40, 40, -10, 1)
else:
    st.sidebar.success(
        "현재 **완전 자동 모드**입니다.\n\n"
        "- **밀집 객체 보존**: 하위 10% 백분위수 배경 추출\n"
        "- **저대비 보정**: 감마 보정으로 옅은 객체 보존\n"
        "- **내부 보호 HPF**: 객체 중심부 무늬 깨짐 방지"
    )

# ---------------------------------------------------------
# 6. 메인 화면: 다중 이미지 업로드 및 결과 시각화
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
            c3.metric("자동 Opening 커널", f"R = {info['open_r']}")
            c4.metric("자동 임계값 / HPF", f"T = {info['thresh']} (w={info['hpf_weight']})")
            c5.metric("자동 극성 판별", "밝은 배경 (반전)" if info["is_inverted"] else "어두운 배경 (유지)")

            fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))

            axes[0].imshow(res["original_gray"], cmap="gray")
            axes[0].set_title("1. Grayscale Input")
            axes[0].axis("off")

            axes[1].imshow(res["corrected"], cmap="gray")
            axes[1].set_title("2. Percentile Bg Flat + Gamma")
            axes[1].axis("off")

            axes[2].imshow(res["edge_map"], cmap="inferno")
            axes[2].set_title(f"3. HPF Edge (w={info['hpf_weight']})")
            axes[2].axis("off")

            axes[3].imshow(res["binary_final"], cmap="gray")
            axes[3].set_title(f"4. Neck-Cut + Open (R={info['open_r']}, T={info['thresh']})")
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
