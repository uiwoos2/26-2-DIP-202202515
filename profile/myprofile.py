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
    "입력된 이미지의 **배경 구조·타겟 크기·형상(세장비)**을 먼저 진단하여 "
    "**① 가방 속 숫자(ROI 내부), ② 면봉(교차 세장형), ③ 보도블럭 낙엽(소구경 Top-Hat + MAD), ④ 키보드 격자, ⑤ 입체 블록·동전(HPF + 단위면적 분할)** 중 최적의 알고리즘을 자동으로 다르게 적용합니다."
)

# ---------------------------------------------------------
# 1. 공통 유틸리티 및 수학적 기초 함수
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
    """이미지 외곽 테두리에 길게 걸친 배경 덩어리(바닥, 베젤, 큰 그림자) 제거"""
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
    """Sobel 1차 미분 에지와 Laplacian 2차 미분 골짜기(Valley) 결합"""
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
# 2. 단위 면적(Unit Area) 기반 겹친 객체 강제 분할 알고리즘
#    (겹친 블록 image_95cce0, 교차 면봉 image_95cd9f, 밀집 동전 해결 핵심)
# ---------------------------------------------------------
def split_clumps_by_unit_area(binary, edge_map, unit_area, unit_radius, clump_ratio=1.48):
    """
    단일 객체 기준 면적(unit_area)보다 1.48배 이상 큰 덩어리(겹친 블록, 교차된 면봉 등)를 찾아,
    덩어리 면적 / 기준 면적 = K 개로 HPF 에지 및 거리 피크를 따라 정확히 분할합니다.
    (단일 블록은 K=1이므로 절대 쪼개지지 않습니다.)
    """
    struct = nd.generate_binary_structure(2, 2)
    lbl, num = nd.label(binary, structure=struct)
    if num == 0 or unit_area <= 10:
        return binary

    sizes = np.bincount(lbl.ravel())
    slices = nd.find_objects(lbl)
    output_bw = binary.copy()

    for cid in range(1, num + 1):
        c_area = sizes[cid]
        if c_area <= unit_area * clump_ratio:
            continue

        # 이 덩어리 안에 몇 개의 객체가 뭉쳐 있는지 계산
        k_targets = int(np.clip(round(c_area / unit_area), 2, 30))
        sl = slices[cid - 1]
        # 패딩을 주어 슬라이스 추출
        y0, y1 = max(0, sl[0].start - 2), min(binary.shape[0], sl[0].stop + 2)
        x0, x1 = max(0, sl[1].start - 2), min(binary.shape[1], sl[1].stop + 2)

        sub_mask = (lbl[y0:y1, x0:x1] == cid)
        sub_edge = edge_map[y0:y1, x0:x1].astype(np.float64) / 255.0

        # 거리 변환 + HPF 내부 에지 페널티 결합 (블록 경계선에서 값이 낮아지도록 유도)
        edt = nd.distance_transform_edt(sub_mask)
        score_map = edt * (1.0 - 0.50 * sub_edge)
        score_smooth = nd.gaussian_filter(score_map, sigma=max(0.8, unit_radius * 0.18))
        score_smooth[~sub_mask] = 0.0

        # Greedy Peak Picking: 최소 이격 거리(r_sep)를 유지하며 K개의 중심 시드 추출
        seeds = []
        r_sep = max(3.0, unit_radius * 0.72)
        temp_score = score_smooth.copy()

        for _ in range(k_targets):
            max_idx = np.argmax(temp_score)
            yy, xx = np.unravel_index(max_idx, temp_score.shape)
            if temp_score[yy, xx] <= 0:
                break
            seeds.append((yy, xx))
            # 선택된 시드 주변 r_sep 반경 영점 처리
            Y, X = np.ogrid[:temp_score.shape[0], :temp_score.shape[1]]
            dist_sq = (Y - yy) ** 2 + (X - xx) ** 2
            temp_score[dist_sq <= (r_sep ** 2)] = 0.0

        # 만약 겹침이 심해 시드가 부족하면 이격 거리를 좁혀 나머지 시드 보충
        if len(seeds) < k_targets:
            temp_score2 = score_smooth.copy()
            r_sep_small = max(2.0, unit_radius * 0.42)
            for sy, sx in seeds:
                Y, X = np.ogrid[:temp_score2.shape[0], :temp_score2.shape[1]]
                temp_score2[((Y - sy) ** 2 + (X - sx) ** 2) <= (r_sep_small ** 2)] = 0.0
            while len(seeds) < k_targets:
                max_idx = np.argmax(temp_score2)
                yy, xx = np.unravel_index(max_idx, temp_score2.shape)
                if temp_score2[yy, xx] <= 0:
                    break
                seeds.append((yy, xx))
                Y, X = np.ogrid[:temp_score2.shape[0], :temp_score2.shape[1]]
                temp_score2[((Y - yy) ** 2 + (X - sx) ** 2) <= (r_sep_small ** 2)] = 0.0

        if len(seeds) >= 2:
            markers = np.zeros(sub_mask.shape, dtype=np.int32)
            for s_idx, (sy, sx) in enumerate(seeds, start=1):
                markers[sy, sx] = s_idx
            markers = nd.grey_dilation(markers, size=(3, 3))
            markers[~sub_mask] = len(seeds) + 1

            # HPF 에지가 높은 곳을 높은 언덕(Cost)으로 설정하여 블록 테두리를 따라 분할 유도
            inv_dist = 255.0 - np.clip(score_smooth / (score_smooth.max() + 1e-6) * 200.0, 0, 200)
            cost = np.clip(inv_dist + sub_edge * 55.0, 0, 255).astype(np.uint8)
            ws = nd.watershed_ift(cost, markers)
            ws_fg = np.where((ws >= 1) & (ws <= len(seeds)), ws, 0)

            # 서로 다른 시드 영역이 맞닿는 경계선(Ridge) 1픽셀 절단
            max_nb = nd.maximum_filter(ws_fg, size=3)
            min_nb = nd.minimum_filter(np.where(ws_fg > 0, ws_fg, len(seeds) + 2), size=3)
            ridge = (ws_fg > 0) & (max_nb != min_nb)

            sub_out = output_bw[y0:y1, x0:x1]
            sub_out[ridge] = False
            output_bw[y0:y1, x0:x1] = sub_out

    return output_bw

# ---------------------------------------------------------
# 3. 5가지 장면별 전용 알고리즘 파이프라인
# ---------------------------------------------------------

# [알고리즘 1] 컨테이너 내부 ROI 타겟 (예: 흰 바닥 위 검은 가방 속 흰 숫자 - image_95d066)
def pipeline_container_roi(f_raw, f_denoised, params_override=None):
    otsu_g = compute_otsu_threshold(f_denoised.astype(np.uint8))
    dark_mask = f_denoised < (otsu_g * 0.95)
    dark_mask = remove_border_slabs(dark_mask, max_area_ratio=0.15)

    # 가장 큰 어두운 객체(가방 본체) 추출 후 내부 구멍 메우기
    lbl_d, num_d = nd.label(dark_mask)
    if num_d > 0:
        sizes_d = np.bincount(lbl_d.ravel())
        sizes_d[0] = 0
        bag_id = np.argmax(sizes_d)
        bag_hull = nd.binary_fill_holes(lbl_d == bag_id)
        # 핵심: 큰 반경 Opening(R=8)과 Erosion(R=6)으로 가방 어깨끈과 외곽 테두리 흰 바닥 유입을 100% 차단!
        bag_roi = nd.binary_opening(bag_hull, structure=create_disk_structure(8))
        bag_roi = nd.binary_erosion(bag_roi, structure=create_disk_structure(6))
    else:
        bag_roi = np.ones_like(f_denoised, dtype=bool)

    # 가방 내부에서 밝은 숫자만 추출
    bg_local = nd.percentile_filter(f_denoised, percentile=30, size=25)
    f_diff = np.clip(f_denoised - bg_local, 0, None)
    f_diff[~bag_roi] = 0.0  # 가방 본체 바깥은 완벽히 0으로 마스킹

    p99 = np.percentile(f_diff, 99.5)
    f_norm = np.clip(f_diff / (p99 + 1e-6) * 255.0, 0, 255).astype(np.uint8)
    edge_map, _ = extract_hpf_edges(f_norm)

    t_val = max(45, compute_otsu_threshold(f_norm[bag_roi]) if np.any(bag_roi) else 80)
    if params_override:
        t_val = int(np.clip(t_val + params_override["t_offset"], 10, 245))

    bw = (f_norm > t_val) & bag_roi
    # 숫자 획 보존을 위해 아주 약한 Opening만 수행
    open_r = params_override["open_r"] if params_override else 0
    if open_r > 0:
        bw = nd.binary_opening(bw, structure=create_disk_structure(open_r))

    struct = nd.generate_binary_structure(2, 2)
    lbl, num = nd.label(bw, structure=struct)
    if num > 0:
        sizes = np.bincount(lbl.ravel())
        med_a = np.median(sizes[1:])
        keep = (sizes >= max(12, med_a * 0.22)) & (sizes <= med_a * 4.0)
        keep[0] = False
        bw = keep[lbl]

    return f_norm, edge_map, bw, t_val, open_r, False, "알고리즘 1: 내부 컨테이너 ROI (가방 속 숫자 모드)"


# [알고리즘 2] 어두운 배경 위 세장형/교차 객체 및 밝은 입자 (예: 면봉 image_95cd9f, 쌀알 rice.tif)
def pipeline_dark_bg_elongated(f_raw, f_denoised, params_override=None):
    # 어두운 배경이므로 명암 반전 절대 금지! 넓은 배경만 완만하게 차감하여 교차부 할로(Halo) 방지
    bg_wide = nd.percentile_filter(f_denoised, percentile=20, size=45)
    bg_wide = nd.gaussian_filter(bg_wide, sigma=15.0)
    f_diff = np.clip(f_denoised - bg_wide, 0, None)

    p99 = np.percentile(f_diff, 99.5)
    f_norm = np.clip(f_diff / (p99 + 1e-6) * 255.0, 0, 255).astype(np.uint8)
    edge_map, valley_map = extract_hpf_edges(f_norm)

    otsu_t = compute_otsu_threshold(f_norm)
    t_val = max(35, int(otsu_t * 0.82))
    if params_override:
        t_val = int(np.clip(otsu_t + params_override["t_offset"], 10, 245))

    bw = f_norm > t_val
    bw = fill_small_holes(bw, max_hole_px=40)

    # 나란히 붙은 면봉 사이 골짜기(Valley) 절단
    edt = nd.distance_transform_edt(bw)
    r_med = float(np.percentile(edt[edt > 0], 80)) if np.any(edt > 0) else 3.0
    neck_cut = (valley_map > np.percentile(valley_map, 88)) & (edt > 0) & (edt <= max(2.2, r_med * 0.6))
    bw_cut = bw & (~neck_cut)

    open_r = params_override["open_r"] if params_override else 1
    if open_r > 0:
        bw_cut = nd.binary_opening(bw_cut, structure=create_disk_structure(open_r))

    # 단일 면봉/입자의 기준 면적 산출 후 X자 교차 면봉 및 붙은 면봉 분할
    struct = nd.generate_binary_structure(2, 2)
    lbl, num = nd.label(bw_cut, structure=struct)
    if num > 0:
        sizes = np.bincount(lbl.ravel())[1:]
        valid_s = sizes[sizes >= 15]
        if len(valid_s) > 0:
            # 하위 65% 백분위수로 '단일 면봉 1개'의 면적 추정
            unit_area = float(np.percentile(valid_s, 55))
            bw_split = split_clumps_by_unit_area(bw_cut, edge_map, unit_area, r_med, clump_ratio=1.42)
            lbl2, num2 = nd.label(bw_split, structure=struct)
            sizes2 = np.bincount(lbl2.ravel())
            keep = sizes2 >= max(12, unit_area * 0.28)
            keep[0] = False
            bw_cut = keep[lbl2]

    return f_norm, edge_map, bw_cut, t_val, open_r, False, "알고리즘 2: 어두운 배경 세장형·교차 분할 (면봉·입자 모드)"


# [알고리즘 3] 텍스처 바닥 위 소형 산재 타겟 (예: 보도블럭·아스팔트 위 낙엽 3종 - image_95cd41, 95c9da, 95c99b)
def pipeline_textured_small_targets(f_raw, f_denoised, params_override=None):
    # 핵심 1: 소구경(17x17) 중위 필터와 White Top-Hat을 결합하여
    # 밝은 보도블럭 타일(image_95c99b)과 검은 줄눈을 동시에 제거하고 작은 낙엽만 추출!
    bg_local_med = nd.percentile_filter(f_denoised, percentile=55, size=17)
    bg_tophat = nd.grey_opening(f_denoised, size=(15, 15))
    f_tophat = np.minimum(
        np.clip(f_denoised - bg_local_med, 0, None),
        np.clip(f_denoised - bg_tophat, 0, None)
    )
    # 아스팔트의 미세한 점 잡음 스무딩
    f_tophat = nd.gaussian_filter(f_tophat, sigma=0.9)

    p995 = np.percentile(f_tophat, 99.6)
    f_norm_lin = np.clip(f_tophat / (p995 + 1e-6) * 255.0, 0, 255)
    # 감마 보정(0.75)으로 어두운 낙엽(image_95c9da)과 멀리 있는 작은 낙엽(image_95cd41, 95c99b) 밝기 부스트
    f_norm = np.clip(((f_norm_lin / 255.0) ** 0.75) * 255.0, 0, 255).astype(np.uint8)
    edge_map, _ = extract_hpf_edges(f_norm)

    # 핵심 2: 2단 Otsu(T=178) 대신 배경의 MAD(중앙절대편차) 기반 통계적 임계값 사용!
    med_v = float(np.median(f_norm))
    mad_v = float(np.median(np.abs(f_norm.astype(np.float64) - med_v))) + 1.0
    t_val = int(np.clip(med_v + 4.2 * mad_v, 36, 82))
    if params_override:
        t_val = int(np.clip(t_val + params_override["t_offset"], 10, 245))

    bw = f_norm > t_val
    open_r = params_override["open_r"] if params_override else 0
    if open_r > 0:
        bw = nd.binary_opening(bw, structure=create_disk_structure(open_r))

    # 핵심 3: 원근감을 고려하여 사진 위쪽 멀리 있는 작은 낙엽(최소 5px)까지 카운팅에 포함
    struct = nd.generate_binary_structure(2, 2)
    lbl, num = nd.label(bw, structure=struct)
    if num > 0:
        sizes = np.bincount(lbl.ravel())
        # 각 컴포넌트 내부의 최대 밝기가 충분히 뚜렷한 경우 5px 이상 모두 보존
        max_intensities = nd.maximum(f_norm, lbl, index=range(1, num + 1))
        keep = np.zeros(num + 1, dtype=bool)
        for i in range(1, num + 1):
            if (sizes[i] >= 5 and max_intensities[i - 1] >= t_val + 14) or sizes[i] >= 11:
                if sizes[i] <= (bw.size * 0.02):  # 거대 보도블럭 잔상 배제
                    keep[i] = True
        bw = keep[lbl]

    return f_norm, edge_map, bw, t_val, open_r, False, "알고리즘 3: 소구경 Top-Hat + MAD 통계 검출 (낙엽·소형 타겟 모드)"


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

    # 자판 사이의 은색 격자선(Valley)을 강하게 절단
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

    return f_norm, edge_map, bw, t_val, open_r, True, "알고리즘 4: HPF 격자 절단 + 그림자 배제 (키보드·배열 모드)"


# [알고리즘 5] 밝은 배경 위 입체 블록·동전 및 겹친 군집 (예: 겹친 블록 image_95cce0, 20개 블록 95c93b, 동전 IMG_9637)
def pipeline_solid_blocks_and_coins(f_raw, f_denoised, params_override=None):
    # 대구경 상위 백분위수(90%) 포락선으로 흰 종이/바닥의 조명 그라데이션 추정
    h, w = f_denoised.shape
    f_small = f_denoised[::4, ::4]
    bg_s = nd.percentile_filter(f_small, percentile=90, size=25)
    bg_s = nd.gaussian_filter(bg_s, sigma=6.0)
    bg_full = np.array(Image.fromarray(bg_s.astype(np.float32)).resize((w, h), Image.Resampling.BILINEAR), dtype=np.float64)

    f_diff = np.clip(bg_full - f_denoised, 0, None)
    p99 = np.percentile(f_diff, 99.5)
    f_norm = np.clip(f_diff / (p99 + 1e-6) * 255.0, 0, 255).astype(np.uint8)
    edge_map, valley_map = extract_hpf_edges(f_norm)

    otsu_t = compute_otsu_threshold(f_norm)
    # 블록 주변의 옅은 그림자가 붙지 않도록 임계값 설정
    t_val = int(np.clip(otsu_t * 0.88, 55, 135))
    if params_override:
        t_val = int(np.clip(otsu_t + params_override["t_offset"], 10, 245))

    bw = f_norm > t_val
    bw = remove_border_slabs(bw, max_area_ratio=0.05)
    bw = fill_small_holes(bw, max_hole_px=120)

    # 초기 반경 측정
    edt = nd.distance_transform_edt(bw)
    peaks = (edt == nd.maximum_filter(edt, size=9)) & (edt > 3.0)
    r_unit = float(np.median(edt[peaks])) if np.any(peaks) else 10.0
    r_unit = float(np.clip(r_unit, 4.0, 35.0))

    # 접합부(Neck)에 HPF 골짜기 차감 후 Opening 수행
    neck_zone = (edt > 0) & (edt <= r_unit * 0.55)
    cut_pixels = (valley_map > np.percentile(valley_map, 85)) & neck_zone
    bw_carved = bw & (~cut_pixels)

    open_r = params_override["open_r"] if params_override else int(np.clip(round(r_unit * 0.22), 1, 3))
    if open_r > 0:
        bw_opened = nd.binary_opening(bw_carved, structure=create_disk_structure(open_r))
    else:
        bw_opened = bw_carved.copy()
    bw_opened = fill_small_holes(bw_opened, max_hole_px=120)

    # 핵심: 외곽의 단일 블록들(하위 25%~65% 면적 분포)로부터 '블록 1개의 기준 면적(unit_area)'을 정확히 계산!
    struct = nd.generate_binary_structure(2, 2)
    lbl, num = nd.label(bw_opened, structure=struct)
    if num > 0:
        sizes = np.bincount(lbl.ravel())[1:]
        min_prelim = max(20, int(np.pi * (r_unit ** 2) * 0.22))
        valid_s = sizes[sizes >= min_prelim]

        if len(valid_s) > 0:
            # 겹친 거대 덩어리를 제외한 단일 블록들의 중앙값 면적 산출
            q25, q65 = np.percentile(valid_s, [25, 65])
            single_candidates = valid_s[(valid_s >= q25) & (valid_s <= q65)]
            unit_area = float(np.median(single_candidates)) if len(single_candidates) > 0 else float(np.median(valid_s))

            # 1) 겹쳐 있는 중앙 블록 무리(image_95cce0)나 붙은 동전(IMG_9637)을 기준 면적(unit_area) 단위로 강제 분할!
            #    (20개 단일 블록 사진 image_95c93b는 모든 블록이 1.48 * unit_area 미만이므로 절대 쪼개지지 않음!)
            bw_split = split_clumps_by_unit_area(bw_opened, edge_map, unit_area, r_unit, clump_ratio=1.48)

            # 2) 단일 블록 면적의 42% 미만인 그림자 파편 제거 (image_95c93b에서 21개로 카운팅되던 그림자 조각 완벽 제거!)
            lbl_f, num_f = nd.label(bw_split, structure=struct)
            sizes_f = np.bincount(lbl_f.ravel())
            keep_f = sizes_f >= (unit_area * 0.42)
            keep_f[0] = False
            bw_opened = keep_f[lbl_f]

    return f_norm, edge_map, bw_opened, t_val, open_r, True, "알고리즘 5: HPF + 단위면적 비례 군집 분할 (입체 블록·동전 모드)"


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

    # 수동으로 특정 알고리즘을 지정한 경우
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
        # [자동 진단 1] 어두운 배경 (면봉 image_95cd9f, 쌀알 rice.tif 등)
        if border_med < 85 and global_med < 95:
            res_tuple = pipeline_dark_bg_elongated(f_raw, f_denoised, params_override)

        else:
            # [자동 진단 2] 밝은 외곽 배경 안에 거대 어두운 컨테이너(검은 가방 image_95d066)가 있는지 검사
            dark_Check = remove_border_slabs(f_denoised < (otsu_g * 0.92), max_area_ratio=0.15)
            lbl_d, num_d = nd.label(dark_Check)
            max_dark_ratio = (np.max(np.bincount(lbl_d.ravel())[1:]) / float(h * w)) if num_d > 0 else 0.0

            is_container_bag = False
            if border_med > 140 and 0.16 <= max_dark_ratio <= 0.75:
                bag_id = np.argmax(np.bincount(lbl_d.ravel())[1:]) + 1
                bag_hull = nd.binary_fill_holes(lbl_d == bag_id)
                bag_eroded = nd.binary_erosion(bag_hull, structure=create_disk_structure(6))
                # 가방 내부에 밝은 객체(흰 숫자)들이 독립적으로 존재하는지 확인
                inner_bright = (f_denoised > otsu_g) & bag_eroded
                _, n_inner = nd.label(inner_bright)
                if 5 <= n_inner <= 60 and np.mean(f_denoised[bag_eroded] < otsu_g) > 0.65:
                    is_container_bag = True

            if is_container_bag:
                res_tuple = pipeline_container_roi(f_raw, f_denoised, params_override)

            # [자동 진단 3] 중간 톤 텍스처 바닥 위 소형 산재 타겟 (보도블럭·아스팔트 낙엽 3종)
            elif border_med < 175 and global_med < 175:
                res_tuple = pipeline_textured_small_targets(f_raw, f_denoised, params_override)

            # [자동 진단 4 & 5] 밝은 배경 위 어두운 타겟 (키보드 배열 vs 입체 블록·동전)
            else:
                # 직사각형 격자 밀집도 검사 (키보드 판별)
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

    f_norm, edge_map, bw_final, t_val, open_r, is_inv, algo_name = res_tuple

    struct = nd.generate_binary_structure(2, 2)
    labeled_array, final_count = nd.label(bw_final, structure=struct)
    centroids = nd.center_of_mass(bw_final, labeled_array, range(1, final_count + 1)) if final_count > 0 else []

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
# 5. 사이드바 안내 및 메인 화면 구성
# ---------------------------------------------------------
st.sidebar.header("🧠 멀티 알고리즘 라우터")
global_router_mode = st.sidebar.selectbox(
    "기본 알고리즘 선택 방식",
    [
        "자동 감지 (Auto Router)",
        "알고리즘 1: 내부 컨테이너 ROI (가방 숫자)",
        "알고리즘 2: 어두운 배경 세장형·교차 (면봉·쌀알)",
        "알고리즘 3: 소구경 Top-Hat + MAD (보도블럭 낙엽)",
        "알고리즘 4: HPF 격자 절단 (키보드 자판)",
        "알고리즘 5: 단위면적 비례 군집 분할 (블록·동전)"
    ]
)

st.sidebar.info(
    "**자동 분기 작동 원리:**\n"
    "- **가방 숫자**: 가방 본체 내부만 마스킹해 어깨끈·바닥 유입 차단\n"
    "- **면봉**: 어두운 배경 고정 + 교차된 면봉을 단일 면봉 면적 비례로 분할\n"
    "- **낙엽**: `17x17` 소구경 Top-Hat으로 밝은 타일 제거 + `MAD` 임계값으로 작은/중간 밝기 낙엽 모두 검출\n"
    "- **블록·동전**: `42%` 미만 그림자 파편 제거 + 중앙 겹친 블록을 단위 블록 면적($A_{\\text{unit}}$)으로 분할"
)

uploaded_files = st.file_uploader(
    "카운팅할 이미지 파일들을 업로드하세요 (여러 장 동시 선택 가능)",
    type=["tif", "tiff", "png", "jpg", "jpeg", "bmp"],
    accept_multiple_files=True
)

if uploaded_files:
    summary_data = []
    results_list = []

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
                        "알고리즘 1: 내부 컨테이너 ROI (가방 숫자)",
                        "알고리즘 2: 어두운 배경 세장형·교차 (면봉·쌀알)",
                        "알고리즘 3: 소구경 Top-Hat + MAD (보도블럭 낙엽)",
                        "알고리즘 4: HPF 격자 절단 (키보드 자판)",
                        "알고리즘 5: 단위면적 비례 군집 분할 (블록·동전)"
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
