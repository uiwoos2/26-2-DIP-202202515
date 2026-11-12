import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.ndimage as nd
from PIL import Image

# 페이지 기본 설정
st.set_page_config(
    page_title="Object Counting Web App (HW#5)",
    layout="wide"
)

st.title("🔍 영상처리 Object Counting 웹 애플리케이션")
st.markdown(
    "여러 장의 이미지를 입력받아 **그레이스케일 변환 → 노이즈 제거 → 배경 보정 → 이진화 → 형태학적 처리 → 객체 카운팅(Labeling)** 과정을 수행합니다."
)

# ---------------------------------------------------------
# 유틸리티 함수: Otsu 자동 임계값 계산 (NumPy 기반)
# ---------------------------------------------------------
def compute_otsu_threshold(image_uint8):
    hist, _ = np.histogram(image_uint8.flatten(), bins=256, range=(0, 256))
    total = image_uint8.size
    sum_total = np.dot(np.arange(256), hist)
    
    sum_bg, weight_bg, max_var, threshold = 0.0, 0, 0.0, 0
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

# ---------------------------------------------------------
# 영상처리 파이프라인 함수
# ---------------------------------------------------------
def process_image(img_pil, params):
    # 1. Grayscale 변환
    gray_img = img_pil.convert('L')
    f = np.array(gray_img, dtype=np.float64)
    
    # 밝은 배경에 어두운 객체가 있는 경우 이미지 반전
    if params["invert"]:
        f = 255.0 - f

    # 2. 노이즈 제거 (Median Filter)
    if params["med_kernel"] > 1:
        f_denoised = signal.medfilt2d(f, kernel_size=params["med_kernel"])
    else:
        f_denoised = f.copy()

    # 3. 불균일 배경 보정 (Top-hat: 원본 - Grey Opening 배경 추정)
    if params["use_bg_sub"]:
        bg = nd.grey_opening(f_denoised, size=(params["bg_kernel"], params["bg_kernel"]))
        f_corrected = np.clip(f_denoised - bg, 0, 255)
    else:
        f_corrected = f_denoised.copy()

    # 정규화 (0 ~ 255)
    if f_corrected.max() > f_corrected.min():
        f_norm = ((f_corrected - f_corrected.min()) / (f_corrected.max() - f_corrected.min()) * 255.0).astype(np.uint8)
    else:
        f_norm = f_corrected.astype(np.uint8)

    # 4. 이진화 (Thresholding)
    if params["thresh_mode"] == "자동 (Otsu)":
        thresh_val = compute_otsu_threshold(f_norm) + params["thresh_offset"]
        thresh_val = int(np.clip(thresh_val, 0, 255))
    else:
        thresh_val = params["manual_thresh"]

    binary = f_norm > thresh_val

    # 5. 형태학적 연산 (Morphological Operations)
    struct = nd.generate_binary_structure(2, 2)
    if params["fill_holes"]:
        binary = nd.binary_fill_holes(binary)
    if params["opening_iter"] > 0:
        binary = nd.binary_opening(binary, structure=struct, iterations=params["opening_iter"])
    if params["erosion_iter"] > 0:
        binary = nd.binary_erosion(binary, structure=struct, iterations=params["erosion_iter"])

    # 6. Connected Component Labeling 및 최소 크기 필터링
    labeled_array, num_features = nd.label(binary, structure=struct)
    
    if params["min_area"] > 1 and num_features > 0:
        component_sizes = np.bincount(labeled_array.ravel())
        too_small = component_sizes < params["min_area"]
        too_small_mask = too_small[labeled_array]
        binary_filtered = binary.copy()
        binary_filtered[too_small_mask] = False
        labeled_array, num_features = nd.label(binary_filtered, structure=struct)
    else:
        binary_filtered = binary

    # 객체 중심점(Centroid) 계산
    centroids = []
    if num_features > 0:
        centroids = nd.center_of_mass(binary_filtered, labeled_array, range(1, num_features + 1))

    return {
        "original_gray": np.array(gray_img),
        "denoised": f_denoised,
        "corrected": f_norm,
        "binary": binary_filtered,
        "labeled": labeled_array,
        "count": num_features,
        "threshold_used": thresh_val,
        "centroids": centroids
    }

# ---------------------------------------------------------
# 사이드바: 파라미터 설정
# ---------------------------------------------------------
st.sidebar.header("⚙️ 영상처리 파라미터 설정")

st.sidebar.subheader("1. 전처리 (Preprocessing)")
invert = st.sidebar.checkbox("명암 반전 (밝은 배경 / 어두운 객체)", value=False)
med_kernel = st.sidebar.slider("Median Filter 커널 크기 (노이즈 제거)", min_value=1, max_value=15, value=7, step=2)
use_bg_sub = st.sidebar.checkbox("불균일 조명 배경 제거 (Top-hat)", value=True)
bg_kernel = st.sidebar.slider("배경 추정 커널 크기", min_value=15, max_value=101, value=51, step=2, disabled=not use_bg_sub)

st.sidebar.subheader("2. 이진화 (Thresholding)")
thresh_mode = st.sidebar.radio("임계값 설정 방식", ["자동 (Otsu)", "수동 지정"])
if thresh_mode == "자동 (Otsu)":
    thresh_offset = st.sidebar.slider("Otsu 임계값 미세 조정 (Offset)", min_value=-50, max_value=50, value=0, step=1)
    manual_thresh = 128
else:
    manual_thresh = st.sidebar.slider("수동 임계값", min_value=0, max_value=255, value=100, step=1)
    thresh_offset = 0

st.sidebar.subheader("3. 형태학적 후처리 (Morphology)")
fill_holes = st.sidebar.checkbox("객체 내부 구멍 채우기 (Fill Holes)", value=True)
opening_iter = st.sidebar.slider("Binary Opening 반복 횟수 (작은 노이즈 제거)", min_value=0, max_value=5, value=1)
erosion_iter = st.sidebar.slider("Binary Erosion 반복 횟수 (붙은 객체 분리)", min_value=0, max_value=5, value=0)
min_area = st.sidebar.slider("최소 객체 면적 (픽셀 수 필터링)", min_value=1, max_value=500, value=20)

params = {
    "invert": invert,
    "med_kernel": med_kernel,
    "use_bg_sub": use_bg_sub,
    "bg_kernel": bg_kernel,
    "thresh_mode": thresh_mode,
    "thresh_offset": thresh_offset,
    "manual_thresh": manual_thresh,
    "fill_holes": fill_holes,
    "opening_iter": opening_iter,
    "erosion_iter": erosion_iter,
    "min_area": min_area
}

# ---------------------------------------------------------
# 메인 화면: 다중 이미지 업로드 및 결과 출력
# ---------------------------------------------------------
uploaded_files = st.file_uploader(
    "카운팅할 이미지 파일들을 업로드하세요 (여러 장 동시 선택 가능)",
    type=["tif", "tiff", "png", "jpg", "jpeg", "bmp"],
    accept_multiple_files=True
)

if uploaded_files:
    summary_data = []
    results_dict = {}

    for uploaded_file in uploaded_files:
        img_pil = Image.open(uploaded_file)
        res = process_image(img_pil, params)
        results_dict[uploaded_file.name] = res
        summary_data.append({
            "파일명": uploaded_file.name,
            "이미지 해상도": f"{img_pil.width} x {img_pil.height}",
            "적용된 임계값": res["threshold_used"],
            "검출된 객체 수 (개)": res["count"]
        })

    # 전체 이미지 카운팅 요약 테이블
    st.subheader("📊 전체 이미지 Object Counting 요약")
    st.dataframe(summary_data, use_container_width=True)

    st.divider()

    # 개별 이미지 단계별 시각화 탭
    st.subheader("🖼️ 이미지별 단계별 처리 결과")
    tabs = st.tabs([file.name for file in uploaded_files])

    for tab, uploaded_file in zip(tabs, uploaded_files):
        with tab:
            res = results_dict[uploaded_file.name]
            
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("검출된 총 객체 수", f"{res['count']} 개")
            col_m2.metric("적용 임계값 (Threshold)", f"{res['threshold_used']}")
            col_m3.metric("Median Filter 커널", f"{params['med_kernel']} x {params['med_kernel']}")

            # Matplotlib을 이용한 4단계 비교 시각화
            fig, axes = plt.subplots(1, 4, figsize=(18, 5))

            # Step 1: Grayscale
            axes[0].imshow(res["original_gray"], cmap="gray")
            axes[0].set_title("1. Grayscale Input")
            axes[0].axis("off")

            # Step 2: Denoised & Background Corrected
            axes[1].imshow(res["corrected"], cmap="gray")
            axes[1].set_title("2. Denoised & Corrected")
            axes[1].axis("off")

            # Step 3: Binary & Morphology
            axes[2].imshow(res["binary"], cmap="gray")
            axes[2].set_title(f"3. Binarized (T={res['threshold_used']})")
            axes[2].axis("off")

            # Step 4: Labeled Objects
            axes[3].imshow(res["labeled"], cmap="nipy_spectral")
            axes[3].set_title(f"4. Counted Objects: {res['count']}")
            axes[3].axis("off")

            # 중심점 마커 표시
            for cy, cx in res["centroids"]:
                axes[3].plot(cx, cy, "w+", markersize=5)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
else:
    st.info("👆 상단의 'Browse files' 버튼을 눌러 분석할 이미지 파일(예: rice.tif 등)을 1장 이상 업로드해주세요.")
