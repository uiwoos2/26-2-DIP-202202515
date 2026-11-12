import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.ndimage as nd
from PIL import Image

st.set_page_config(
    page_title="Scene-Adaptive Object Counting (HW#5)",
    layout="wide"
)

st.title("🧠 장면 인식 기반 멀티 알고리즘 Object Counting 웹 앱")
st.markdown(
    "입력된 이미지의 **배경 질감·타겟 크기·단위 면적($A_{\\text{unit}}$)**을 자동 분석하여 "
    "**① 가방 속 숫자(끈 제거 ROI), ② 교차·접촉 면봉(K-Means 단위면적 분할), ③ 보도블럭·아스팔트 낙엽(줄눈 봉합 + 히스테리시스 복원), ④ 키보드 격자, ⑤ 겹친 입체 블록·동전** 알고리즘을 자동으로 다르게 적용합니다."
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

def resize_keep_aspect(img_pil, max_dim=512):
    w, h = img_pil.size
    if max(w, h) <= max_dim:
        return img_pil
    scale = max_dim / float(max(w, h))
    return img_pil.resize((int(round(w * scale)), int(round(h * scale))), Image.Resampling.BILINEAR)

def fill_small_holes(binary, max_hole_px=200):
    inv = ~binary
    lbl, num = nd.label(inv)
    if num == 0:
        return binary
    sizes = np.bincount(lbl.ravel())
    border_ids = np.unique(np.concatenate([lbl[0, :], lbl[-1, :], lbl[:, 0], lbl[:, -1]]))
    small = sizes <= max_hole_px
    small[0] = False
    small[border_ids] = False
    return binary | small[lbl]

def remove_border_slabs(binary, max_area_ratio=0.035):
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

    b_mask = np.zeros_like(binary, dtype=bool)
    b_mask[0, :] = b_mask[-1, :] = b_mask[:, 0] = b_mask[:, -1] = True
    b_counts = np.bincount(lbl[b_mask].ravel(), minlength=num + 1)

    remove_ids = []
    for bid in border_ids:
        sides = sum([bid in top_ids, bid in bot_ids, bid in left_ids, bid in right_ids])
        if sides >= 2 or b_counts[bid] > max(h, w) * 0.10 or sizes[bid] > (h * w * max_area_ratio):
            remove_ids.append(bid)

    if remove_ids:
        keep = np.ones(num + 1, dtype=bool)
        keep[0] = False
        keep[remove_ids] = False
        return keep[lbl]
    return binary

def extract_hpf_edges(f_img):
    """Sobel 1차 미분 에지와 Laplacian 2차 미분 골짜기(Valley) 추출"""
    f_s = nd.gaussian_filter(f_img.astype(np.float64), sigma=1.0)
    gx = nd.sobel(f_s, axis=1)
    gy = nd.sobel(f_s, axis=0)
    sobel = np.hypot(gx, gy)
    s99 = np.percentile(sobel, 99.0)
    sobel_norm = np.clip(sobel / (s99 + 1e-6) * 255.0, 0, 255)

    lap_k = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float64)
    lap = signal.convolve2d(f_s, lap_k, mode="same", boundary="symm")
    valley = np.clip(-lap, 0, None)
    v99 = np.percentile(valley, 99.0)
    valley_norm = np.clip(valley / (v99 + 1e-6) * 255.0, 0, 255)

    edge_map = np.clip(0.55 * sobel_norm + 0.45 * valley_norm, 0, 255).astype(np.uint8)
    return edge_map, valley_norm.astype(np.uint8)

# ---------------------------------------------------------
# 2. 단위 면적(Unit Area) 기반 가중 K-Means 클러스터링 분할기
#    (X자 교차 면봉, 접촉 면봉, 중앙에 겹친 블록 무리를 정확히 K개로 분할)
# ---------------------------------------------------------
def partition_clumps_into_k_objects(binary, edge_map, unit_area, clump_ratio=1.42, min_ratio=0.38):
    """
    1) unit_area * min_ratio 미만의 그림자/파편 제거
    2) unit_area * clump_ratio 미만의 단일 객체는 K=1로 그대로 보존
    3) unit_area * clump_ratio 이상의 겹친/교차 덩어리는 K = round(Area / unit_area)개의
       독립 라벨로 분할하여 X자 교차 면봉이나 겹친 블록도 정확히 K개로 카운팅
    """
    struct = nd.generate_binary_structure(2, 2)
    lbl, num = nd.label(binary, structure=struct)
    if num == 0 or unit_area <= 8:
        return binary, lbl, num

    sizes = np.bincount(lbl.ravel())
    slices = nd.find_objects(lbl)

    final_labeled = np.zeros_like(lbl, dtype=np.int32)
    final_binary = np.zeros_like(binary, dtype=bool)
    current_label = 0

    for cid in range(1, num + 1):
        c_area = sizes[cid]
        # 그림자 조각이나 미세 파편 배제
        if c_area < unit_area * min_ratio:
            continue

        sl = slices[cid - 1]
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        sub_mask = (lbl[y0:y1, x0:x1] == cid)

        # 단일 객체인 경우 (K = 1)
        if c_area < unit_area * clump_ratio:
            current_label += 1
            final_labeled[y0:y1, x0:x1][sub_mask] = current_label
            final_binary[y0:y1, x0:x1][sub_mask] = True
            continue

        # 겹치거나 교차된 복합 덩어리인 경우: K개 객체로 분할
        k_targets = int(np.clip(round(c_area / unit_area), 2, 35))
        ys, xs = np.nonzero(sub_mask)
        coords = np.column_stack((ys.astype(np.float64), xs.astype(np.float64)))

        if len(coords) < k_targets * 5:
            current_label += 1
            final_labeled[y0:y1, x0:x1][sub_mask] = current_label
            final_binary[y0:y1, x0:x1][sub_mask] = True
            continue

        # 중심부일수록, HPF 에지가 낮을수록 높은 가중치 부여
        sub_edt = nd.distance_transform_edt(sub_mask)
        sub_edge = edge_map[y0:y1, x0:x1].astype(np.float64) / 255.0
        weights = sub_edt[ys, xs] * (1.0 - 0.55 * sub_edge[ys, xs]) + 0.1

        # 결정론적 Farthest-Point (K-Means++) 초기화:
        # 면봉처럼 긴 객체에서도 시드들이 한쪽 끝에 몰리지 않고 서로 멀리 배치됨
        centers = np.zeros((k_targets, 2), dtype=np.float64)
        first_idx = int(np.argmax(weights))
        centers[0] = coords[first_idx]

        min_dist_sq = np.sum((coords - centers[0]) ** 2, axis=1)
        for k in range(1, k_targets):
            score_init = min_dist_sq * (weights ** 0.5)
            next_idx = int(np.argmax(score_init))
            centers[k] = coords[next_idx]
            new_dist_sq = np.sum((coords - centers[k]) ** 2, axis=1)
            min_dist_sq = np.minimum(min_dist_sq, new_dist_sq)

        # 가중 K-Means 반복으로 K개 객체의 중심 수렴
        labels_k = np.zeros(len(coords), dtype=np.int32)
        for _ in range(12):
            dists = np.sum((coords[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels_k = np.argmin(dists, axis=1)
            for k in range(k_targets):
                mask_k = (labels_k == k)
                if np.any(mask_k):
                    w_k = weights[mask_k]
                    centers[k, 0] = np.sum(coords[mask_k, 0] * w_k) / np.sum(w_k)
                    centers[k, 1] = np.sum(coords[mask_k, 1] * w_k) / np.sum(w_k)

        # 클러스터 맵 구성 및 시각적 경계선(Ridge) 1픽셀 절단
        sub_lbl_map = np.zeros(sub_mask.shape, dtype=np.int32)
        for k in range(k_targets):
            mask_k = (labels_k == k)
            if np.any(mask_k):
                current_label += 1
                sub_lbl_map[ys[mask_k], xs[mask_k]] = current_label

        max_nb = nd.maximum_filter(sub_lbl_map, size=3)
        min_nb = nd.minimum_filter(np.where(sub_lbl_map > 0, sub_lbl_map, current_label + 10), size=3)
        ridge = (sub_lbl_map > 0) & (max_nb != min_nb)

        sub_bin = sub_lbl_map > 0
        sub_bin[ridge] = False

        final_labeled[y0:y1, x0:x1][sub_lbl_map > 0] = sub_lbl_map[sub_lbl_map > 0]
        final_binary[y0:y1, x0:x1][sub_bin] = True

    return final_binary, final_labeled, current_label

# ---------------------------------------------------------
# 3. 5가지 장면별 전용 알고리즘 파이프라인
# ---------------------------------------------------------

# [알고리즘 1] 컨테이너 내부 ROI 타겟 (예: 흰 바닥 위 검은 가방 속 흰 숫자 - image_95d066)
def pipeline_container_roi(f_raw, f_denoised, params_override=None):
    otsu_g = compute_otsu_threshold(f_denoised.astype(np.uint8))
    dark_mask = f_denoised < (otsu_g * 0.92)
    dark_mask = remove_border_slabs(dark_mask, max_area_ratio=0.15)

    lbl_d, num_d = nd.label(dark_mask)
    if num_d > 0:
        sizes_d = np.bincount(lbl_d.ravel())
        sizes_d[0] = 0
        bag_id = int(np.argmax(sizes_d))
        bag_raw = (lbl_d == bag_id)
        # 핵심: 가방 내부의 작은 흰색 숫자 구멍(최대 2500px)만 먼저 메우고,
        # 가방 어깨끈 사이의 거대한 흰색 바닥 구멍은 메우지 않은 상태에서 R=11 Opening을 수행해 어깨끈을 완전히 잘라냄!
        bag_filled_nums = fill_small_holes(bag_raw, max_hole_px=2500)
        bag_body_only = nd.binary_opening(bag_filled_nums, structure=create_disk_structure(11))
        bag_body_only = nd.binary_fill_holes(bag_body_only)
        # 가방 테두리 안쪽으로 7px 축소(Erosion)하여 가방 바깥/어깨끈 근처 흰 바닥을 100% 차단
        bag_roi = nd.binary_erosion(bag_body_only, structure=create_disk_structure(7))
    else:
        bag_roi = np.ones_like(f_denoised, dtype=bool)

    bg_local = nd.percentile_filter(f_denoised, percentile=30, size=25)
    f_diff = np.clip(f_denoised - bg_local, 0, None)
    f_diff[~bag_roi] = 0.0

    p99 = np.percentile(f_diff, 99.5)
    f_norm = np.clip(f_diff / (p99 + 1e-6) * 255.0, 0, 255).astype(np.uint8)
    edge_map, _ = extract_hpf_edges(f_norm)

    t_val = max(50, compute_otsu_threshold(f_norm[bag_roi]) if np.any(bag_roi) else 85)
    if params_override:
        t_val = int(np.clip(t_val + params_override["t_offset"], 10, 245))

    bw = (f_norm > t_val) & bag_roi
    open_r = params_override["open_r"] if params_override else 0
    if open_r > 0:
        bw = nd.binary_opening(bw, structure=create_disk_structure(open_r))

    struct = nd.generate_binary_structure(2, 2)
    lbl, num = nd.label(bw, structure=struct)
    if num > 0:
        sizes = np.bincount(lbl.ravel())
        med_a = float(np.median(sizes[1:]))
        keep = (sizes >= max(15, med_a * 0.25)) & (sizes <= med_a * 3.5)
        keep[0] = False
        bw = keep[lbl]
        lbl, num = nd.label(bw, structure=struct)

    return f_norm, edge_map, bw, lbl, num, t_val, open_r, False, "알고리즘 1: 끈 제거 ROI 마스킹 (가방 속 숫자 모드)"


# [알고리즘 2] 어두운 배경 위 세장형/교차 객체 및 입자 (예: 면봉 image_9563e6, 쌀알 rice.tif)
def pipeline_dark_bg_elongated(f_raw, f_denoised, params_override=None):
    bg_wide = nd.percentile_filter(f_denoised, percentile=20, size=45)
    bg_wide = nd.gaussian_filter(bg_wide, sigma=15.0)
    f_diff = np.clip(f_denoised - bg_wide, 0, None)

    p99 = np.percentile(f_diff, 99.5)
    f_norm = np.clip(f_diff / (p99 + 1e-6) * 255.0, 0, 255).astype(np.uint8)
    edge_map, valley_map = extract_hpf_edges(f_norm)

    otsu_t = compute_otsu_threshold(f_norm)
    t_val = max(38, int(otsu_t * 0.85))
    if params_override:
        t_val = int(np.clip(otsu_t + params_override["t_offset"], 10, 245))

    # 나란히 붙은 면봉 사이 골짜기(Valley)를 차감한 뒤 이진화
    f_carved = np.clip(f_norm.astype(np.float64) - 0.45 * valley_map.astype(np.float64), 0, 255)
    bw = f_carved > t_val
    bw = fill_small_holes(bw, max_hole_px=35)

    open_r = params_override["open_r"] if params_override else 1
    if open_r > 0:
        bw = nd.binary_opening(bw, structure=create_disk_structure(open_r))

    # 단일 면봉 기준 면적(unit_area) 산출 후 X자 교차 면봉 및 접촉 면봉을 K개로 정확히 분할!
    struct = nd.generate_binary_structure(2, 2)
    lbl_init, num_init = nd.label(bw, structure=struct)
    if num_init > 0:
        sizes = np.bincount(lbl_init.ravel())[1:]
        valid_s = sizes[sizes >= 20]
        if len(valid_s) > 0:
            # 하위 20%~65% 구간의 중앙값으로 '단일 면봉 1개'의 기준 면적 산출
            q20, q65 = np.percentile(valid_s, [20, 65])
            single_s = valid_s[(valid_s >= q20) & (valid_s <= q65)]
            unit_area = float(np.median(single_s)) if len(single_s) > 0 else float(np.median(valid_s))

            bw_final, lbl_final, count_final = partition_clumps_into_k_objects(
                bw, edge_map, unit_area, clump_ratio=1.42, min_ratio=0.35
            )
            return f_norm, edge_map, bw_final, lbl_final, count_final, t_val, open_r, False, "알고리즘 2: 단위면적 K-Means 교차 분할 (면봉·입자 모드)"

    return f_norm, edge_map, bw, lbl_init, num_init, t_val, open_r, False, "알고리즘 2: 단위면적 K-Means 교차 분할 (면봉·입자 모드)"


# [알고리즘 3] 텍스처 바닥 위 소형 산재 타겟 (예: 보도블럭·아스팔트 위 낙엽 - image_9563a3, 9560a3, 956084 등)
def pipeline_textured_small_targets(f_raw, f_denoised, params_override=None):
    # 핵심 1: 5x5 Median 필터로 아스팔트 자갈/모래알 점 잡음(1~3px)을 먼저 평탄화
    f_smooth = signal.medfilt2d(f_raw, kernel_size=5)

    # 핵심 2: 보도블럭의 어두운 줄눈(균열)을 grey_closing으로 먼저 메운 뒤 상위 72% 배경 표면을 추정하여
    # 줄눈 옆 보도블럭 테두리가 하얗게 뜨는 현상(Halo)과 밝은 타일 오검출을 동시에 완벽 제거!
    h, w = f_smooth.shape
    f_small = f_smooth[::2, ::2]
    f_closed_s = nd.grey_closing(f_small, size=(7, 7))
    bg_s = nd.percentile_filter(f_closed_s, percentile=72, size=15)
    bg_s = nd.gaussian_filter(bg_s, sigma=3.0)
    bg_surface = np.array(Image.fromarray(bg_s.astype(np.float32)).resize((w, h), Image.Resampling.BILINEAR), dtype=np.float64)

    f_diff = np.clip(f_smooth - bg_surface, 0, None)
    f_diff = nd.gaussian_filter(f_diff, sigma=0.8)

    # 핵심 3: 낙엽이 적은 이미지에서 배경 자갈이 255로 증폭되지 않도록 최소 스케일 보장 정규화
    max_ref = max(float(np.percentile(f_diff, 99.9)), 28.0)
    f_norm = np.clip(f_diff / max_ref * 255.0, 0, 255).astype(np.uint8)
    edge_map, _ = extract_hpf_edges(f_norm)

    # 핵심 4: 히스테리시스(Hysteresis) 이중 임계값 + R=1 Opening으로 자갈 노이즈 폭증(354/432/928개) 원천 차단!
    otsu_t = compute_otsu_threshold(f_norm)
    med_v = float(np.median(f_norm))
    mad_v = float(np.median(np.abs(f_norm.astype(np.float64) - med_v))) + 1.0

    # T_low: 낙엽의 전체 외형을 보존하는 임계값 / T_high: 실제 낙엽으로 인정할 중심부(Core) 임계값
    t_low = int(np.clip(max(med_v + 4.5 * mad_v, otsu_t * 0.72, 38), 35, 115))
    if params_override:
        t_low = int(np.clip(t_low + params_override["t_offset"], 15, 240))
    t_high = int(np.clip(max(t_low + 22, otsu_t * 1.05, 65), 55, 175))

    bw_low = f_norm > t_low
    open_r = params_override["open_r"] if params_override else 1
    # 3x3 십자/원형 Opening으로 미세한 아스팔트 돌기 제거
    if open_r > 0:
        bw_low = nd.binary_opening(bw_low, structure=create_disk_structure(open_r))

    # T_high 이상인 확실한 밝기 코어를 가진 객체만 복원 (Morphological Reconstruction)
    struct = nd.generate_binary_structure(2, 2)
    core_markers = (f_norm > t_high) & bw_low
    bw_recon = nd.binary_propagation(core_markers, mask=bw_low, structure=struct)

    lbl, num = nd.label(bw_recon, structure=struct)
    if num > 0:
        sizes = np.bincount(lbl.ravel())
        # 최소 10px 이상이면서 거대 타일 잔상이 아닌 실제 낙엽만 유지
        keep = (sizes >= 10) & (sizes <= (bw_recon.size * 0.015))
        keep[0] = False
        bw_recon = keep[lbl]
        lbl, num = nd.label(bw_recon, structure=struct)

    return f_norm, edge_map, bw_recon, lbl, num, t_low, open_r, False, "알고리즘 3: 줄눈 봉합 표면차감 + 히스테리시스 복원 (낙엽 모드)"


# [알고리즘 4] 격자 배열 타겟 (예: 키보드 자판 - image_963d62)
def pipeline_grid_array(f_raw, f_denoised, params_override=None):
    bg_bright = nd.percentile_filter(f_denoised, percentile=88, size=45)
    bg_bright = nd.gaussian_filter(bg_bright, sigma=8.0)
    f_diff = np.clip(bg_bright - f_denoised, 0, None)

    p99 = np.percentile(f_diff, 99.5)
    f_norm = np.clip(f_diff / (p99 + 1e-6) * 255.0, 0, 255).astype(np.uint8)
    edge_map, valley_map = extract_hpf_edges(f_norm)

    otsu_t = compute_otsu_threshold(f_norm)
    t_val = int(np.clip(otsu_t * 1.05, 70, 160))
    if params_override:
        t_val = int(np.clip(otsu_t + params_override["t_offset"], 10, 245))

    f_cut = np.clip(f_norm.astype(np.float64) - 0.65 * valley_map.astype(np.float64), 0, 255)
    bw = f_cut > t_val
    bw = remove_border_slabs(bw, max_area_ratio=0.03)

    open_r = params_override["open_r"] if params_override else 2
    if open_r > 0:
        bw = nd.binary_opening(bw, structure=create_disk_structure(open_r))
    bw = fill_small_holes(bw, max_hole_px=80)

    struct = nd.generate_binary_structure(2, 2)
    lbl, num = nd.label(bw, structure=struct)
    if num > 0:
        sizes = np.bincount(lbl.ravel())
        med_a = float(np.median(sizes[1:]))
        keep = (sizes >= max(20, med_a * 0.25)) & (sizes <= med_a * 7.5)
        keep[0] = False
        bw = keep[lbl]
        lbl, num = nd.label(bw, structure=struct)

    return f_norm, edge_map, bw, lbl, num, t_val, open_r, True, "알고리즘 4: HPF 격자 절단 + 그림자 배제 (키보드·배열 모드)"


# [알고리즘 5] 밝은 배경 위 입체 블록·동전 및 중앙 겹침 군집 (예: 겹친 블록 image_95cce0, 20개 블록 95c93b, 동전 IMG_9637)
def pipeline_solid_blocks_and_coins(f_raw, f_denoised, params_override=None):
    h, w = f_denoised.shape
    f_small = f_denoised[::4, ::4]
    bg_s = nd.percentile_filter(f_small, percentile=92, size=25)
    bg_s = nd.gaussian_filter(bg_s, sigma=6.0)
    bg_full = np.array(Image.fromarray(bg_s.astype(np.float32)).resize((w, h), Image.Resampling.BILINEAR), dtype=np.float64)

    f_diff = np.clip(bg_full - f_denoised, 0, None)
    p99 = np.percentile(f_diff, 99.5)
    f_norm_lin = np.clip(f_diff / (p99 + 1e-6) * 255.0, 0, 255)
    # 밝은 회색 레고 블록(image_95cce0)이나 옅은 동전(IMG_9637)도 손실되지 않도록 약한 감마(0.8) 보정 적용
    f_norm = np.clip(((f_norm_lin / 255.0) ** 0.80) * 255.0, 0, 255).astype(np.uint8)
    edge_map, valley_map = extract_hpf_edges(f_norm)

    otsu_t = compute_otsu_threshold(f_norm)
    t_val = int(np.clip(otsu_t * 0.85, 50, 130))
    if params_override:
        t_val = int(np.clip(otsu_t + params_override["t_offset"], 10, 245))

    bw = f_norm > t_val
    bw = remove_border_slabs(bw, max_area_ratio=0.05)
    bw = fill_small_holes(bw, max_hole_px=180)

    edt = nd.distance_transform_edt(bw)
    peaks = (edt == nd.maximum_filter(edt, size=9)) & (edt > 3.0)
    r_unit = float(np.median(edt[peaks])) if np.any(peaks) else 10.0
    r_unit = float(np.clip(r_unit, 4.0, 35.0))

    # 블록/동전 사이 접합부 골짜기(Valley) 절단 후 Opening
    cut_pixels = (valley_map > np.percentile(valley_map, 84)) & (edt > 0) & (edt <= r_unit * 0.65)
    bw_carved = bw & (~cut_pixels)

    open_r = params_override["open_r"] if params_override else int(np.clip(round(r_unit * 0.22), 1, 3))
    if open_r > 0:
        bw_opened = nd.binary_opening(bw_carved, structure=create_disk_structure(open_r))
    else:
        bw_opened = bw_carved.copy()
    bw_opened = fill_small_holes(bw_opened, max_hole_px=180)

    # 외곽 단일 블록들의 중앙값으로부터 '단일 객체 기준 면적(unit_area)'을 구해
    # 중앙에 겹친 블록 무리(image_95cce0)와 맞닿은 동전들을 K-Means 단위 면적으로 분할!
    struct = nd.generate_binary_structure(2, 2)
    lbl_init, num_init = nd.label(bw_opened, structure=struct)
    if num_init > 0:
        sizes = np.bincount(lbl_init.ravel())[1:]
        min_prelim = max(25, int(np.pi * (r_unit ** 2) * 0.25))
        valid_s = sizes[sizes >= min_prelim]
        if len(valid_s) > 0:
            q25, q65 = np.percentile(valid_s, [25, 65])
            single_s = valid_s[(valid_s >= q25) & (valid_s <= q65)]
            unit_area = float(np.median(single_s)) if len(single_s) > 0 else float(np.median(valid_s))

            bw_final, lbl_final, count_final = partition_clumps_into_k_objects(
                bw_opened, edge_map, unit_area, clump_ratio=1.48, min_ratio=0.42
            )
            return f_norm, edge_map, bw_final, lbl_final, count_final, t_val, open_r, True, "알고리즘 5: HPF + 단위면적 K-Means 분할 (입체 블록·동전 모드)"

    return f_norm, edge_map, bw_opened, lbl_init, num_init, t_val, open_r, True, "알고리즘 5: HPF + 단위면적 K-Means 분할 (입체 블록·동전 모드)"


# ---------------------------------------------------------
# 4. 이미지 자동 진단 및 알고리즘 라우터 (Scene Classifier)
# ---------------------------------------------------------
def diagnose_and_route(img_pil, selected_pipeline="자동 감지 (Auto Router)", params_override=None):
    gray_full = img_pil.convert("RGB").convert("L")
    gray_resized = resize_keep_aspect(gray_full, max_dim=512)
    f_raw = np.array(gray_resized, dtype=np.float64)
    f_denoised = signal.medfilt2d(f_raw, kernel_size=3)

    h, w = f_denoised.shape
    pad_h, pad_w = max(2, int(h * 0.04)), max(2, int(w * 0.04))
    border_pixels = np.concatenate([
        f_denoised[:pad_h, :].ravel(), f_denoised[-pad_h:, :].ravel(),
        f_denoised[:, :pad_w].ravel(), f_denoised[:, -pad_w:].ravel()
    ])
    border_med = float(np.median(border_pixels))
    global_med = float(np.median(f_denoised))
    otsu_g = compute_otsu_threshold(f_denoised.astype(np.uint8))

    if selected_pipeline != "자동 감지 (Auto Router)":
        if "알고리즘 1" in selected_pipeline:
            res_tuple = pipeline_container_roi(f_raw, f_denoised, params_override)
        elif "알고리즘 2" in selected_pipeline:
            res_tuple = pipeline_dark_bg_elongated(f_raw, f_denoised, params_override)
        elif "알고리즘 3" in selected_pipeline:
            res_tuple = pipeline_textured_small_targets(f_raw, f_denoised, params_override)
        elif "알고리즘 4" in selected_pipeline:
            res_tuple = pipeline_grid_array(f_raw, f_denoised, params_override)
        else:
            res_tuple = pipeline_solid_blocks_and_coins(f_raw, f_denoised, params_override)
    else:
        # [자동 진단 1] 어두운 배경 (면봉 image_9563e6, 쌀알 rice.tif 등)
        if border_med < 85 and global_med < 95:
            res_tuple = pipeline_dark_bg_elongated(f_raw, f_denoised, params_override)
        else:
            # [자동 진단 2] 밝은 바닥 위 거대 어두운 가방 안에 밝은 숫자가 있는 경우 (image_95d066)
            dark_check = remove_border_slabs(f_denoised < (otsu_g * 0.92), max_area_ratio=0.15)
            lbl_d, num_d = nd.label(dark_check)
            max_dark_ratio = (np.max(np.bincount(lbl_d.ravel())[1:]) / float(h * w)) if num_d > 0 else 0.0

            is_container_bag = False
            if border_med > 140 and 0.16 <= max_dark_ratio <= 0.75:
                bag_id = int(np.argmax(np.bincount(lbl_d.ravel())[1:])) + 1
                bag_hull = nd.binary_fill_holes(lbl_d == bag_id)
                bag_eroded = nd.binary_erosion(bag_hull, structure=create_disk_structure(6))
                inner_bright = (f_denoised > otsu_g) & bag_eroded
                _, n_inner = nd.label(inner_bright)
                if 5 <= n_inner <= 60 and np.mean(f_denoised[bag_eroded] < otsu_g) > 0.65:
                    is_container_bag = True

            if is_container_bag:
                res_tuple = pipeline_container_roi(f_raw, f_denoised, params_override)
            # [자동 진단 3] 보도블럭·아스팔트 등 중간 톤 텍스처 바닥 위 소형 낙엽 (image_9563a3, 9560a3, 956084 등)
            elif border_med < 175 and global_med < 175:
                res_tuple = pipeline_textured_small_targets(f_raw, f_denoised, params_override)
            # [자동 진단 4 & 5] 밝은 배경 위 키보드 자판 vs 입체 블록·동전
            else:
                bw_test = remove_border_slabs(f_denoised < (otsu_g * 0.95))
                lbl_t, num_t = nd.label(bw_test)
                rect_fills = []
                if num_t >= 25:
                    sizes_t = np.bincount(lbl_t.ravel())[1:]
                    for idx, sl in enumerate(nd.find_objects(lbl_t)):
                        if sl is not None and sizes_t[idx] > 40:
                            ba = (sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start)
                            rect_fills.append(sizes_t[idx] / float(max(1, ba)))
                if len(rect_fills) >= 25 and np.median(rect_fills) > 0.74:
                    res_tuple = pipeline_grid_array(f_raw, f_denoised, params_override)
                else:
                    res_tuple = pipeline_solid_blocks_and_coins(f_raw, f_denoised, params_override)

    f_norm, edge_map, bw_final, labeled_array, final_count, t_val, open_r, is_inv, algo_name = res_tuple

    centroids = []
    if final_count > 0:
        unique_labels = [i for i in range(1, int(labeled_array.max()) + 1) if np.any(labeled_array == i)]
        final_count = len(unique_labels)
        centroids = nd.center_of_mass(labeled_array > 0, labeled_array, unique_labels)

    return {
        "original_gray": np.array(gray_resized),
        "corrected": f_norm,
        "edge_map": edge_map,
        "binary_final": bw_final,
        "labeled": labeled_array,
        "count": final_count,
        "centroids": centroids,
        "info": {
            "algo_name": algo_name,
            "is_inverted": is_inv,
            "thresh": t_val,
            "open_r": open_r
        }
    }

# ---------------------------------------------------------
# 5. 사이드바 및 메인 UI
# ---------------------------------------------------------
st.sidebar.header("🧠 멀티 알고리즘 라우터")
global_router_mode = st.sidebar.selectbox(
    "기본 알고리즘 선택 방식",
    [
        "자동 감지 (Auto Router)",
        "알고리즘 1: 끈 제거 ROI 마스킹 (가방 숫자)",
        "알고리즘 2: 단위면적 K-Means 교차 분할 (면봉·쌀알)",
        "알고리즘 3: 줄눈 봉합 + 히스테리시스 복원 (보도블럭 낙엽)",
        "알고리즘 4: HPF 격자 절단 (키보드 자판)",
        "알고리즘 5: 단위면적 K-Means 군집 분할 (블록·동전)"
    ]
)

st.sidebar.info(
    "**핵심 개선 사항:**\n"
    "- **면봉 교차·접촉**: 단일 면봉 면적($A_{\\text{unit}}$)을 기준으로 `X`자 교차 및 붙은 면봉을 K-Means 공간 클러스터링으로 정확히 $K$개로 분할\n"
    "- **낙엽 노이즈 폭증 해결**: 보도블럭 줄눈을 `grey_closing`으로 먼저 메워 테두리 할로를 없애고, 이중 임계값(`T_high` 코어 복원)으로 아스팔트 자갈 오검출 차단\n"
    "- **가방 숫자**: 어깨끈을 `R=11` Opening으로 잘라낸 뒤 가방 몸통 내부만 마스킹"
)

uploaded_files = st.file_uploader(
    "카운팅할 이미지 파일들을 업로드하세요 (여러 장 동시 선택 가능)",
    type=["tif", "tiff", "png", "jpg", "jpeg", "bmp"],
    accept_multiple_files=True
)

if uploaded_files:
    summary_data = []

    st.subheader("🖼️ 이미지별 자동 선택 알고리즘 및 5단계 처리 결과")
    tabs = st.tabs([file.name for file in uploaded_files])

    for idx, (tab, uploaded_file) in enumerate(zip(tabs, uploaded_files)):
        img_pil = Image.open(uploaded_file)

        with tab:
            with st.expander("🔧 이 이미지의 알고리즘 또는 임계값 수동 변경하기 (선택 사항)"):
                c_a, c_b, c_c = st.columns(3)
                ind_pipe = c_a.selectbox(
                    "적용할 알고리즘 선택",
                    [
                        "자동 감지 (Auto Router)",
                        "알고리즘 1: 끈 제거 ROI 마스킹 (가방 숫자)",
                        "알고리즘 2: 단위면적 K-Means 교차 분할 (면봉·쌀알)",
                        "알고리즘 3: 줄눈 봉합 + 히스테리시스 복원 (보도블럭 낙엽)",
                        "알고리즘 4: HPF 격자 절단 (키보드 자판)",
                        "알고리즘 5: 단위면적 K-Means 군집 분할 (블록·동전)"
                    ],
                    index=0,
                    key=f"pipe_{idx}"
                )
                use_custom_param = c_b.checkbox("임계값/Opening 수동 미세 조정", value=False, key=f"cust_{idx}")
                ind_off = c_b.slider("임계값 Offset", -50, 50, 0, 2, key=f"off_{idx}", disabled=not use_custom_param)
                ind_rad = c_c.slider("Opening 반경 (R)", 0, 5, 1, 1, key=f"rad_{idx}", disabled=not use_custom_param)

            chosen_mode = ind_pipe if ind_pipe != "자동 감지 (Auto Router)" else global_router_mode
            override = {"t_offset": ind_off, "open_r": ind_rad} if use_custom_param else None

            res = diagnose_and_route(img_pil, selected_pipeline=chosen_mode, params_override=override)
            info = res["info"]

            st.success(f"**자동 진단 적용 알고리즘**: `{info['algo_name']}`")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("검출된 총 객체 수", f"{res['count']} 개")
            m2.metric("적용 임계값 (Threshold)", f"T = {info['thresh']}")
            m3.metric("적용 Opening 반경", f"R = {info['open_r']}")
            m4.metric("타겟 극성", "어두운 타겟 (반전)" if info["is_inverted"] else "밝은 타겟 (원본)")

            fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))

            axes[0].imshow(res["original_gray"], cmap="gray")
            axes[0].set_title("1. Grayscale Input")
            axes[0].axis("off")

            axes[1].imshow(res["corrected"], cmap="gray")
            axes[1].set_title("2. Scene-Specific Bg Sub")
            axes[1].axis("off")

            axes[2].imshow(res["edge_map"], cmap="inferno")
            axes[2].set_title("3. HPF (Sobel + Valley)")
            axes[2].axis("off")

            axes[3].imshow(res["binary_final"], cmap="gray")
            axes[3].set_title(f"4. Segmented (R={info['open_r']}, T={info['thresh']})")
            axes[3].axis("off")

            axes[4].imshow(res["labeled"], cmap="nipy_spectral")
            axes[4].set_title(f"5. Counted: {res['count']}")
            axes[4].axis("off")

            for cy, cx in res["centroids"]:
                axes[4].plot(cx, cy, "w+", markersize=6, markeredgewidth=1.2)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        summary_data.append({
            "파일명": uploaded_file.name,
            "원본 해상도": f"{img_pil.width} x {img_pil.height}",
            "자동 선택된 알고리즘": info["algo_name"],
            "검출된 객체 수 (개)": res["count"],
            "적용 임계값 (T)": info["thresh"],
            "Opening (R)": f"R = {info['open_r']}"
        })

    st.divider()
    st.subheader("📊 전체 이미지 자동 Object Counting 요약")
    st.dataframe(summary_data, use_container_width=True)
else:
    st.info("👆 상단의 'Browse files' 버튼을 눌러 이미지들을 업로드하면, 각 이미지의 장면 특성을 진단하여 최적의 알고리즘을 자동 적용합니다.")
