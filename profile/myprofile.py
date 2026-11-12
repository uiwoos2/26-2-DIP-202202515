import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.ndimage as nd
from PIL import Image

st.set_page_config(
    page_title="Universal Auto Object Counting (HW#5)",
    layout="wide"
)

st.title("🤖 범용 완전 자동화 Object Counting 웹 애플리케이션")
st.markdown(
    "**양방향 극성·다단 임계값 자동 탐색**과 **형상 인식 조건부 HPF + Opening 알고리즘**을 통해 "
    "작은 객체(낙엽·쌀알), 가방 속 숫자, 키보드 배열, 얇고 긴 객체(면봉), 밀집 동전을 모두 자동으로 판별하여 카운팅합니다."
)

# ---------------------------------------------------------
# 1. 기초 유틸리티 함수
# ---------------------------------------------------------
def compute_otsu_threshold(arr_uint8):
    if arr_uint8.size == 0:
        return 127
    hist, _ = np.histogram(arr_uint8.flatten(), bins=256, range=(0, 256))
    total = arr_uint8.size
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
    return int(threshold)

def create_disk_structure(radius):
    if radius <= 0:
        return nd.generate_binary_structure(2, 1)
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x ** 2 + y ** 2) <= (radius ** 2)

def fill_small_internal_holes(binary, max_hole_px=150):
    """객체 내부의 아주 작은 반사 구멍만 메우고, 키보드 격자나 배경은 절대 메우지 않음"""
    inv = ~binary
    labeled_holes, num_holes = nd.label(inv)
    if num_holes == 0:
        return binary
    hole_sizes = np.bincount(labeled_holes.ravel())
    border_labels = np.unique(np.concatenate([
        labeled_holes[0, :], labeled_holes[-1, :],
        labeled_holes[:, 0], labeled_holes[:, -1]
    ]))
    small_holes = hole_sizes <= max_hole_px
    small_holes[0] = False
    small_holes[border_labels] = False
    return binary | small_holes[labeled_holes]

def remove_border_background_slabs(binary):
    """
    사진 외곽 테두리(흰 나무 바닥, 노트북 본체 베젤/그림자 등)에 걸친 거대 배경 덩어리를 제거하되,
    가장자리에 살짝 닿은 정상적인 작은 객체(동전, 쌀알 등)는 보존
    """
    h, w = binary.shape
    struct = nd.generate_binary_structure(2, 2)
    lbl, num = nd.label(binary, structure=struct)
    if num == 0:
        return binary

    sizes = np.bincount(lbl.ravel())
    top_ids = set(np.unique(lbl[0, :])) - {0}
    bot_ids = set(np.unique(lbl[-1, :])) - {0}
    left_ids = set(np.unique(lbl[:, 0])) - {0}
    right_ids = set(np.unique(lbl[:, -1])) - {0}
    border_ids = top_ids | bot_ids | left_ids | right_ids

    border_mask = np.zeros_like(binary, dtype=bool)
    border_mask[0, :] = border_mask[-1, :] = border_mask[:, 0] = border_mask[:, -1] = True
    border_counts = np.bincount(lbl[border_mask].ravel(), minlength=num + 1)

    remove_ids = []
    for bid in border_ids:
        sides_touched = sum([bid in top_ids, bid in bot_ids, bid in left_ids, bid in right_ids])
        # 2개 이상의 외곽 변에 동시에 닿거나, 테두리 접촉 길이가 길거나, 화면의 3.5% 이상을 차지하는 외곽 덩어리는 배경으로 제거
        if sides_touched >= 2 or border_counts[bid] > max(h, w) * 0.12 or sizes[bid] > (h * w * 0.035):
            remove_ids.append(bid)

    if remove_ids:
        keep = np.ones(num + 1, dtype=bool)
        keep[0] = False
        keep[remove_ids] = False
        return keep[lbl]
    return binary

def resize_keep_aspect(img_pil, max_dim=512):
    w, h = img_pil.size
    if max(w, h) <= max_dim:
        return img_pil
    scale = max_dim / float(max(w, h))
    return img_pil.resize((int(round(w * scale)), int(round(h * scale))), Image.Resampling.BILINEAR)

# ---------------------------------------------------------
# 2. 배경 조명 추정 및 HPF 에지 추출
# ---------------------------------------------------------
def estimate_local_background(f_img, percentile=50, window_scale=75):
    """
    - 밝은 타겟 검출 시: percentile=50 (보도블럭 표면, 가방 천 등 중위 표면만 추출해 차감)
    - 어두운 타겟 검출 시: percentile=88 (키보드 은색 격자, 동전 사이 흰 배경 등 상위 배경 포락선 추출)
    """
    h, w = f_img.shape
    step = 4
    f_small = f_img[::step, ::step]
    win_s = max(11, int(round(window_scale / step)))
    if win_s % 2 == 0:
        win_s += 1

    bg_small = nd.percentile_filter(f_small, percentile=percentile, size=win_s)
    bg_small = nd.gaussian_filter(bg_small, sigma=win_s * 0.25)
    bg_pil = Image.fromarray(bg_small.astype(np.float32)).resize((w, h), Image.Resampling.BILINEAR)
    return np.array(bg_pil, dtype=np.float64)

def extract_hpf_edges(f_norm):
    """Sobel 경계와 Laplacian 골짜기(인접 객체가 맞닿은 오목한 경계)를 결합한 HPF 맵 생성"""
    f_s = nd.gaussian_filter(f_norm.astype(np.float64), sigma=1.0)

    gx = nd.sobel(f_s, axis=1)
    gy = nd.sobel(f_s, axis=0)
    sobel_mag = np.hypot(gx, gy)
    s99 = np.percentile(sobel_mag, 99.0)
    sobel_norm = np.clip(sobel_mag / (s99 + 1e-6) * 255.0, 0, 255)

    lap_kernel = np.array([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]], dtype=np.float64)
    lap_resp = signal.convolve2d(f_s, lap_kernel, mode="same", boundary="symm")
    # 두 밝은 객체 사이의 어두운 골짜기(음의 라플라시안) 추출
    valley = np.clip(-lap_resp, 0, None)
    v99 = np.percentile(valley, 99.0)
    valley_norm = np.clip(valley / (v99 + 1e-6) * 255.0, 0, 255)

    edge_map = np.clip(0.5 * sobel_norm + 0.5 * valley_norm, 0, 255).astype(np.uint8)
    return edge_map, valley_norm.astype(np.uint8)

# ---------------------------------------------------------
# 3. 형상 인식 조건부 HPF + Opening + 선별 분할기
# ---------------------------------------------------------
def refine_and_split_mask(binary_raw, f_norm, edge_map, valley_map, hpf_weight=0.5, force_open_r=None):
    struct = nd.generate_binary_structure(2, 2)
    # 1) 외곽에 걸친 대형 배경/프레임(가방 주변 바닥, 노트북 하단 그림자 등) 우선 제거
    binary = remove_border_background_slabs(binary_raw)

    # 아주 작은 점 잡음(6px 미만) 1차 제거
    lbl_0, n_0 = nd.label(binary, structure=struct)
    if n_0 == 0:
        return binary, 0, 0.0, 1, "객체 없음"
    sizes_0 = np.bincount(lbl_0.ravel())
    keep_0 = sizes_0 >= 6
    keep_0[0] = False
    binary = keep_0[lbl_0]

    # 2) 객체 형상(두께 R_obj 및 세장비 Elongation) 분석
    edt_0 = nd.distance_transform_edt(binary)
    lbl_1, n_1 = nd.label(binary, structure=struct)
    if n_1 == 0:
        return binary, 0, 0.0, 1, "객체 없음"

    comp_areas = np.bincount(lbl_1.ravel())[1:]
    comp_max_r = nd.maximum(edt_0, lbl_1, index=range(1, n_1 + 1))
    r_obj = float(np.median(comp_max_r)) if len(comp_max_r) > 0 else 4.0
    r_obj = float(np.clip(r_obj, 1.5, 45.0))

    # 세장비(Elongation): 원형은 Area / (pi * R^2) ≈ 1.0~2.2, 면봉처럼 가늘고 긴 객체는 > 2.8
    elongation_ratios = comp_areas / (np.pi * (np.maximum(comp_max_r, 1.0) ** 2))
    median_elongation = float(np.median(elongation_ratios))

    # 형상 분류:
    # - 세장형(면봉 등): median_elongation > 2.7
    # - 미세/획 타겟(가방 숫자, 작은 낙엽 등): r_obj <= 4.2
    # - 일반/밀집 블록·원형(동전, 키보드 자판, 블록 등): 그 외
    is_elongated = bool(median_elongation > 2.7)
    is_thin_stroke = bool(r_obj <= 4.2)

    if is_elongated:
        shape_type = "세장형 (면봉·막대 등)"
        open_r = 1 if r_obj >= 2.8 else 0
    elif is_thin_stroke:
        shape_type = "미세·선형 (숫자·작은 낙엽 등)"
        open_r = 1 if r_obj >= 2.5 else 0
    else:
        shape_type = "블록·원형 (동전·자판·블록 등)"
        open_r = int(np.clip(round(r_obj * 0.25), 1, 3))

    if force_open_r is not None:
        open_r = force_open_r

    # 3) 블록·원형 객체일 경우 접합부(Neck)에 HPF 골짜기 차감 및 내부 미세 구멍 메우기
    if not is_elongated and not is_thin_stroke:
        binary = fill_small_internal_holes(binary, max_hole_px=int(np.pi * (r_obj ** 2) * 0.35))
        edt_solid = nd.distance_transform_edt(binary)
        # 중심부는 보호하고 외곽 접합부(0 < edt <= 0.55 * r_obj)에서만 강한 HPF 골짜기 절단
        neck_zone = (edt_solid > 0) & (edt_solid <= r_obj * 0.55)
        cut_line = (valley_map > np.percentile(valley_map, 86)) & neck_zone & (hpf_weight > 0.1)
        binary = binary & (~cut_line)

    # 4) 형상 맞춤 Morphological Opening 수행
    if open_r > 0:
        binary_opened = nd.binary_opening(binary, structure=create_disk_structure(open_r), iterations=1)
    else:
        binary_opened = binary.copy()

    binary_opened = remove_border_background_slabs(binary_opened)

    # 5) 복합 덩어리 선별 분할 (Selective Clump Watershed)
    # 면봉이나 가방 숫자는 쪼개지지 않도록 건너뛰고, 둥근 객체가 1.55배 이상 뭉친 덩어리만 분할
    lbl_2, n_2 = nd.label(binary_opened, structure=struct)
    if not is_elongated and not is_thin_stroke and n_2 >= 3:
        sizes_2 = np.bincount(lbl_2.ravel())[1:]
        med_area_2 = float(np.median(sizes_2))
        clump_ids = [i + 1 for i, s in enumerate(sizes_2) if s > med_area_2 * 1.55]

        if len(clump_ids) > 0:
            clump_mask = np.isin(lbl_2, clump_ids)
            # 스페이스바처럼 두께가 얇고 균일한 직사각형 단일 키는 분할에서 제외
            dist_all = nd.distance_transform_edt(binary_opened)
            edge_pen = np.clip(edge_map.astype(np.float64) / 255.0, 0, 1)
            dist_guided = dist_all * (1.0 - 0.35 * hpf_weight * edge_pen)
            dist_smooth = nd.gaussian_filter(dist_guided, sigma=max(1.0, r_obj * 0.22))

            seed_fp = max(5, int(round(r_obj * 1.3)))
            local_max = (dist_smooth == nd.maximum_filter(dist_smooth, size=seed_fp)) & (dist_all > r_obj * 0.42)
            local_max = nd.binary_dilation(local_max, structure=create_disk_structure(max(1, int(r_obj * 0.18))))
            markers, num_seeds = nd.label(local_max, structure=struct)

            if num_seeds > 1:
                markers_ws = markers.astype(np.int32)
                markers_ws[~binary_opened] = num_seeds + 1
                cost = (255 - np.clip(dist_smooth / (dist_smooth.max() + 1e-6) * 255.0, 0, 255)).astype(np.uint8)
                ws = nd.watershed_ift(cost, markers_ws)
                ws_fg = np.where((ws >= 1) & (ws <= num_seeds), ws, 0)
                max_nb = nd.maximum_filter(ws_fg, size=3)
                min_nb = nd.minimum_filter(np.where(ws_fg > 0, ws_fg, num_seeds + 2), size=3)
                ridge = (ws_fg > 0) & (max_nb != min_nb) & clump_mask
                binary_opened = binary_opened & (~ridge)

    # 6) 최종 면적 이상치(잔잡음 및 비정상 거대 배경 덩어리) 필터링
    lbl_3, n_3 = nd.label(binary_opened, structure=struct)
    if n_3 == 0:
        return binary_opened, 0, r_obj, open_r, shape_type

    sizes_3 = np.bincount(lbl_3.ravel())
    fg_3 = sizes_3[1:]
    min_abs = max(8, int(binary_opened.size * 0.00018))
    valid_fg = fg_3[fg_3 >= min_abs]
    if len(valid_fg) == 0:
        return np.zeros_like(binary_opened), 0, r_obj, open_r, shape_type

    med_area = float(np.median(valid_fg))
    min_allow = max(min_abs, med_area * 0.18)
    # 중앙값 면적의 9.5배를 넘는 거대 배경 그림자나 프레임 덩어리 배제 (스페이스바 등은 ~5배이므로 통과)
    max_allow = max(med_area * 9.5, binary_opened.size * 0.08)

    keep_final = (sizes_3 >= min_allow) & (sizes_3 <= max_allow)
    keep_final[0] = False
    binary_final = keep_final[lbl_3]
    _, final_count = nd.label(binary_final, structure=struct)

    return binary_final, final_count, r_obj, open_r, shape_type

# ---------------------------------------------------------
# 4. 후보 분할 품질 자동 평가 함수 (Target Plausibility Score)
# ---------------------------------------------------------
def score_candidate_mask(binary_final, f_norm):
    """
    검출된 객체들의 면적 규칙성, 배경 범람 여부, 밀실도(보도블럭 금/그물망 배제), 경계 대비를 종합 평가
    """
    struct = nd.generate_binary_structure(2, 2)
    lbl, count = nd.label(binary_final, structure=struct)
    if count < 2 or count > 450:
        return -1e6 if count == 0 else -500.0

    fg_ratio = float(np.mean(binary_final))
    # 화면의 55% 이상이 하얗게 덮이면 배경이 범람한 것(예: 보도블럭 전체가 잡힌 경우)이므로 큰 감점
    if fg_ratio > 0.55 or fg_ratio < 0.001:
        return -300.0

    sizes = np.bincount(lbl.ravel())[1:].astype(np.float64)
    med_area = np.median(sizes)

    # 1) 면적 균일도 점수 (규칙적인 타겟일수록 중앙값 주변에 밀집)
    inliers = np.mean((sizes >= med_area * 0.30) & (sizes <= med_area * 3.2))
    log_std = np.std(np.log1p(sizes))
    uniformity_score = (inliers * 45.0) / (1.0 + 0.6 * log_std)

    # 2) 형태 충실도(Bounding Box 대비 채움 비율): 보도블럭 균열 같은 거미줄 형태는 점수가 매우 낮음
    slices = nd.find_objects(lbl)
    fill_ratios = []
    for idx, sl in enumerate(slices):
        if sl is None:
            continue
        box_h = sl[0].stop - sl[0].start
        box_w = sl[1].stop - sl[1].start
        box_area = max(1, box_h * box_w)
        fill_ratios.append(sizes[idx] / box_area)
    med_fill = float(np.median(fill_ratios)) if fill_ratios else 0.1
    # 너무 텅 빈 거미줄(균열, 격자 테두리) 감점
    shape_score = 25.0 * med_fill if med_fill >= 0.18 else -80.0

    # 3) 전경-배경 명암 대비 점수
    dilated = nd.binary_dilation(binary_final, iterations=2)
    ring = dilated & (~binary_final)
    contrast = float(np.mean(f_norm[binary_final]) - np.mean(f_norm[ring])) if np.any(ring) else 0.0
    contrast_score = np.clip(contrast * 0.35, 0.0, 35.0)

    # 4) 적정 객체 수 가산점 (3개 ~ 180개 사이 선호)
    count_bonus = 15.0 if 4 <= count <= 180 else 5.0

    return uniformity_score + shape_score + contrast_score + count_bonus

# ---------------------------------------------------------
# 5. 메인 자동화 파이프라인 (4대 후보 모드 탐색 및 최적 결과 반환)
# ---------------------------------------------------------
def process_image_universal(img_pil, mode_choice="자동 최적 탐색 (Auto)", manual_params=None):
    gray_full = img_pil.convert("RGB").convert("L")
    gray_resized = resize_keep_aspect(gray_full, max_dim=512)
    f_raw = np.array(gray_resized, dtype=np.float64)

    # 미세 노이즈 제거 (3x3 Median으로 얇은 면봉, 숫자 획, 키보드 틈새 보존)
    f_denoised = signal.medfilt2d(f_raw, kernel_size=3)

    # [후보군 A: 밝은 타겟 모드 (쌀알, 보도블럭 위 낙엽, 검은 가방 위 흰 숫자 등)]
    bg_dark = estimate_local_background(f_denoised, percentile=50, window_scale=65)
    f_bright_diff = np.clip(f_denoised - bg_dark, 0, None)
    p99_b = np.percentile(f_bright_diff, 99.5)
    f_bright_norm = np.clip(f_bright_diff / (p99_b + 1e-6) * 255.0, 0, 255).astype(np.uint8)

    # [후보군 B: 어두운 타겟 모드 (밝은 배경 위 블록, 동전, 키보드 자판 등)]
    bg_bright = estimate_local_background(f_denoised, percentile=88, window_scale=75)
    f_dark_diff = np.clip(bg_bright - f_denoised, 0, None)
    p99_d = np.percentile(f_dark_diff, 99.5)
    f_dark_norm = np.clip(f_dark_diff / (p99_d + 1e-6) * 255.0, 0, 255).astype(np.uint8)

    # 수동 모드일 경우 지정된 설정으로 즉시 실행
    if manual_params is not None:
        is_inv = (manual_params["polarity"] == "어두운 타겟 (반전)")
        f_use = f_dark_norm if is_inv else f_bright_norm
        edge_map, valley_map = extract_hpf_edges(f_use)
        t_otsu = compute_otsu_threshold(f_use)
        t_val = int(np.clip(t_otsu + manual_params["t_offset"], 10, 245))
        bw_raw = f_use > t_val
        bw_final, cnt, r_obj, open_r, s_type = refine_and_split_mask(
            bw_raw, f_use, edge_map, valley_map,
            hpf_weight=manual_params["hpf_w"],
            force_open_r=manual_params["open_r"]
        )
    else:
        # 4가지 자동 후보 생성:
        # 1) 밝은 타겟 (표준 1단 Otsu) - 예: rice.tif, 가방 위 흰 숫자
        # 2) 밝은 고휘도 타겟 (2단 상위 Otsu) - 예: 회색 보도블럭 위 밝은 낙엽
        # 3) 어두운 타겟 (표준 1단 Otsu) - 예: 흰 배경 위 블록, 밀집 동전
        # 4) 어두운 고대비 타겟 (2단 상위 Otsu) - 예: 은색 노트북 위 검은 키보드 자판
        candidates = []
        for pol_name, is_inv, f_use in [
            ("밝은 타겟 (원본 극성)", False, f_bright_norm),
            ("어두운 타겟 (반전 극성)", True, f_dark_norm)
        ]:
            edge_map, valley_map = extract_hpf_edges(f_use)
            t1 = max(28, compute_otsu_threshold(f_use))
            upper_pixels = f_use[f_use > t1]
            t2 = int(np.clip(compute_otsu_threshold(upper_pixels), t1 + 15, 235)) if len(upper_pixels) > 50 else t1 + 25

            # 동전처럼 밝기 편차가 큰 밀집 객체를 위한 완화 임계값 후보(t0)도 함께 점검
            t0 = max(25, int(t1 * 0.78))

            for t_label, t_cand in [("표준 Otsu", t1), ("고대비 2단 Otsu", t2), ("저대비 포괄 Otsu", t0)]:
                if mode_choice == "밝은 타겟만 탐색" and is_inv:
                    continue
                if mode_choice == "어두운 타겟만 탐색" and not is_inv:
                    continue

                bw_raw = f_use > t_cand
                bw_final, cnt, r_obj, open_r, s_type = refine_and_split_mask(
                    bw_raw, f_use, edge_map, valley_map, hpf_weight=0.55
                )
                sc = score_candidate_mask(bw_final, f_use)
                # 표준 Otsu에 약간의 기본 우선순위 부여 (동점 방지)
                if t_label == "표준 Otsu":
                    sc += 4.0
                candidates.append({
                    "score": sc,
                    "pol_name": f"{pol_name} - {t_label}",
                    "is_inv": is_inv,
                    "f_use": f_use,
                    "edge_map": edge_map,
                    "bw_final": bw_final,
                    "count": cnt,
                    "thresh": t_cand,
                    "r_obj": r_obj,
                    "open_r": open_r,
                    "shape_type": s_type
                })

        # 가장 품질 점수가 높은 후보 자동 선택
        best = max(candidates, key=lambda x: x["score"])
        f_use = best["f_use"]
        edge_map = best["edge_map"]
        bw_final = best["bw_final"]
        cnt = best["count"]
        t_val = best["thresh"]
        r_obj = best["r_obj"]
        open_r = best["open_r"]
        s_type = best["shape_type"]
        is_inv = best["is_inv"]

    struct = nd.generate_binary_structure(2, 2)
    labeled_array, final_count = nd.label(bw_final, structure=struct)
    centroids = nd.center_of_mass(bw_final, labeled_array, range(1, final_count + 1)) if final_count > 0 else []

    return {
        "original_gray": np.array(gray_resized),
        "corrected": f_use,
        "edge_map": edge_map,
        "binary_final": bw_final,
        "labeled": labeled_array,
        "count": final_count,
        "centroids": centroids,
        "info": {
            "is_inverted": is_inv,
            "r_obj": round(r_obj, 1),
            "open_r": open_r,
            "thresh": t_val,
            "shape_type": s_type
        }
    }

# ---------------------------------------------------------
# 6. 사이드바 UI 구성
# ---------------------------------------------------------
st.sidebar.header("🤖 자동화 모드 설정")
auto_mode = st.sidebar.toggle("✨ 100% 완전 자동 판별 모드", value=True)

if auto_mode:
    mode_filter = st.sidebar.radio(
        "탐색 범위 설정",
        ["자동 최적 탐색 (Auto)", "밝은 타겟만 탐색", "어두운 타겟만 탐색"],
        index=0
    )
    manual_params = None
    st.sidebar.info(
        "**작동 중인 자동화 기술:**\n"
        "1. **양방향 극성 & 다단 Otsu 평가**: 가방 속 흰 숫자, 보도블럭 위 낙엽, 키보드 자판 중 실제 타겟 군집을 자동 선택\n"
        "2. **외곽 배경 자동 소거**: 사진 테두리에 걸친 흰 바닥·노트북 그림자 배제\n"
        "3. **형상 적응형 Opening**: 면봉·숫자처럼 얇은 타겟은 침식을 최소화(`R=0~1`)하고, 뭉툭한 타겟만 HPF+Opening 분리"
    )
else:
    mode_filter = "수동"
    st.sidebar.subheader("수동 파라미터 제어")
    manual_params = {
        "polarity": st.sidebar.selectbox("타겟 밝기", ["밝은 타겟 (원본)", "어두운 타겟 (반전)"]),
        "t_offset": st.sidebar.slider("임계값 미세 조정 (Offset)", -60, 60, 0, 2),
        "hpf_w": st.sidebar.slider("HPF 에지 절단 강도", 0.0, 1.2, 0.5, 0.05),
        "open_r": st.sidebar.slider("Opening 커널 반경 (R)", 0, 5, 1, 1)
    }

# ---------------------------------------------------------
# 7. 메인 화면: 다중 이미지 업로드 및 결과 출력
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
        res = process_image_universal(img_pil, mode_choice=mode_filter, manual_params=manual_params)
        results_list.append((uploaded_file.name, img_pil, res))
        info = res["info"]

        summary_data.append({
            "파일명": uploaded_file.name,
            "원본 해상도": f"{img_pil.width} x {img_pil.height}",
            "검출된 객체 수 (개)": res["count"],
            "자동 인식 형상": info["shape_type"],
            "적용 극성": "어두운 타겟 (반전)" if info["is_inverted"] else "밝은 타겟 (원본)",
            "추정 두께 반경 (px)": info["r_obj"],
            "자동 Opening (R)": f"R = {info['open_r']}",
            "선택 임계값 (T)": info["thresh"]
        })

    st.subheader("📊 전체 이미지 자동 Object Counting 요약")
    st.dataframe(summary_data, use_container_width=True)

    st.divider()
    st.subheader("🖼️ 이미지별 5단계 처리 결과 및 자동 인식 정보")
    tabs = st.tabs([item[0] for item in results_list])

    for idx, (tab, (fname, img_pil, res)) in enumerate(zip(tabs, results_list)):
        with tab:
            # 특정 탭에서만 수동으로 오버라이드하고 싶을 때 사용할 수 있는 개별 조정창
            with st.expander("🔧 이 이미지만 수동으로 파라미터 변경하기 (선택 사항)"):
                col_a, col_b, col_c, col_d = st.columns(4)
                use_ind = col_a.checkbox("수동 변경 적용", value=False, key=f"ind_{idx}")
                ind_pol = col_b.selectbox("타겟 극성", ["밝은 타겟 (원본)", "어두운 타겟 (반전)"], key=f"ipol_{idx}", disabled=not use_ind)
                ind_off = col_c.slider("임계값 Offset", -60, 60, 0, 2, key=f"ioff_{idx}", disabled=not use_ind)
                ind_rad = col_d.slider("Opening 반경 (R)", 0, 5, 1, 1, key=f"irad_{idx}", disabled=not use_ind)

            if use_ind:
                res = process_image_universal(
                    img_pil,
                    manual_params={"polarity": ind_pol, "t_offset": ind_off, "hpf_w": 0.5, "open_r": ind_rad}
                )

            info = res["info"]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("검출된 총 객체 수", f"{res['count']} 개")
            c2.metric("인식된 타겟 형태", info["shape_type"])
            c3.metric("추정 반경 / Opening", f"{info['r_obj']} px / R={info['open_r']}")
            c4.metric("자동 선택 임계값", f"T = {info['thresh']}")
            c5.metric("타겟 극성", "어두운 타겟 (반전)" if info["is_inverted"] else "밝은 타겟 (원본)")

            fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))

            axes[0].imshow(res["original_gray"], cmap="gray")
            axes[0].set_title("1. Grayscale Input")
            axes[0].axis("off")

            axes[1].imshow(res["corrected"], cmap="gray")
            axes[1].set_title("2. Local Surface Bg Sub")
            axes[1].axis("off")

            axes[2].imshow(res["edge_map"], cmap="inferno")
            axes[2].set_title("3. HPF (Sobel + Valley)")
            axes[2].axis("off")

            axes[3].imshow(res["binary_final"], cmap="gray")
            axes[3].set_title(f"4. Adaptive Open (R={info['open_r']}, T={info['thresh']})")
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
    st.info("👆 상단의 'Browse files' 버튼을 눌러 이미지들을 업로드하면, 각 이미지의 타겟 형태와 배경을 자동 분석하여 카운팅합니다.")
