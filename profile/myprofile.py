import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.ndimage as nd
from PIL import Image

st.set_page_config(page_title="Object Counting", layout="wide")
st.title("Object Counting")

# ---------------------------------------------------------
# 유틸리티 함수
# ---------------------------------------------------------
def get_otsu_threshold(img_uint8):
    if img_uint8.size == 0:
        return 127
    hist, _ = np.histogram(img_uint8.flatten(), bins=256, range=(0, 256))
    total = img_uint8.size
    sum_total = np.dot(np.arange(256), hist)

    sum_bg, w_bg, max_var, thresh = 0.0, 0, 0.0, 127
    for t in range(256):
        w_bg += hist[t]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_total - sum_bg) / w_fg
        var_between = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        if var_between > max_var:
            max_var = var_between
            thresh = t
    return int(thresh)

def disk_kernel(radius):
    if radius <= 0:
        return nd.generate_binary_structure(2, 1)
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x ** 2 + y ** 2) <= (radius ** 2)

def resize_image(img_pil, max_dim=512):
    w, h = img_pil.size
    if max(w, h) <= max_dim:
        return img_pil
    scale = max_dim / float(max(w, h))
    return img_pil.resize((int(round(w * scale)), int(round(h * scale))), Image.Resampling.BILINEAR)

def fill_holes_limited(binary, max_hole_px=350):
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

def remove_border_artifacts(binary, max_area_ratio=0.04):
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
        if sides >= 2 or b_counts[bid] > max(h, w) * 0.12 or sizes[bid] > (h * w * max_area_ratio):
            remove_ids.append(bid)

    if remove_ids:
        keep = np.ones(num + 1, dtype=bool)
        keep[0] = False
        keep[remove_ids] = False
        return keep[lbl]
    return binary

def get_hpf_edge(f_img):
    f_s = nd.gaussian_filter(f_img.astype(np.float64), sigma=1.2)
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
# 겹친 덩어리 조건부 분할 함수
# ---------------------------------------------------------
def split_overlapping_clumps(binary, edge_map, r_med, med_area):
    """
    단일 객체는 보존하고, 기준 면적의 1.38배를 초과하며 내부에 실제 거리 피크/경계가 존재하는
    겹친 블록·인접 블록 덩어리만 Watershed로 분할
    """
    struct = nd.generate_binary_structure(2, 2)
    lbl, num = nd.label(binary, structure=struct)
    if num == 0:
        return binary

    sizes = np.bincount(lbl.ravel())
    slices = nd.find_objects(lbl)
    out_bw = binary.copy()

    for cid in range(1, num + 1):
        c_area = sizes[cid]
        if c_area <= med_area * 1.38:
            continue

        sl = slices[cid - 1]
        y0, y1 = max(0, sl[0].start - 2), min(binary.shape[0], sl[0].stop + 2)
        x0, x1 = max(0, sl[1].start - 2), min(binary.shape[1], sl[1].stop + 2)
        sub_mask = (lbl[y0:y1, x0:x1] == cid)

        # 내부 에지 밀도가 매우 낮고 두께가 균일한 단일 대형 부품은 분할하지 않음
        sub_edt = nd.distance_transform_edt(sub_mask)
        sub_edge = edge_map[y0:y1, x0:x1].astype(np.float64) / 255.0
        inner_zone = sub_edt > (r_med * 0.35)
        inner_edge_mean = float(np.mean(sub_edge[inner_zone])) if np.any(inner_zone) else 0.0

        # 2개 블록 수준(1.38~2.3배)은 잘록한 목이나 에지가 있을 때 분할, 대형 군집(>2.3배)은 내부 에지가 뚜렷할 때만 분할
        if c_area > med_area * 2.3 and inner_edge_mean < 0.16:
            continue

        max_targets = int(np.clip(round(c_area / (med_area * 0.92)), 2, 30))
        score_map = sub_edt * (1.0 - 0.45 * sub_edge)
        score_smooth = nd.gaussian_filter(score_map, sigma=max(1.0, r_med * 0.22))
        score_smooth[~sub_mask] = 0.0

        seeds = []
        r_sep = max(4.0, r_med * 0.95)
        temp = score_smooth.copy()

        for _ in range(max_targets):
            idx = int(np.argmax(temp))
            yy, xx = np.unravel_index(idx, temp.shape)
            if temp[yy, xx] < r_med * 0.38:
                break
            seeds.append((yy, xx))
            Y, X = np.ogrid[:temp.shape[0], :temp.shape[1]]
            temp[((Y - yy) ** 2 + (X - xx) ** 2) <= (r_sep ** 2)] = 0.0

        if len(seeds) >= 2:
            markers = np.zeros(sub_mask.shape, dtype=np.int32)
            for s_idx, (sy, sx) in enumerate(seeds, start=1):
                markers[sy, sx] = s_idx
            markers = nd.grey_dilation(markers, size=(3, 3))
            markers[~sub_mask] = len(seeds) + 1

            inv_d = 255.0 - np.clip(score_smooth / (score_smooth.max() + 1e-6) * 200.0, 0, 200)
            cost = np.clip(inv_d + sub_edge * 55.0, 0, 255).astype(np.uint8)
            ws = nd.watershed_ift(cost, markers)
            ws_fg = np.where((ws >= 1) & (ws <= len(seeds)), ws, 0)

            max_nb = nd.maximum_filter(ws_fg, size=3)
            min_nb = nd.minimum_filter(np.where(ws_fg > 0, ws_fg, len(seeds) + 5), size=3)
            ridge = (ws_fg > 0) & (max_nb != min_nb)

            sub_out = out_bw[y0:y1, x0:x1]
            sub_out[ridge] = False
            out_bw[y0:y1, x0:x1] = sub_out

    return out_bw

def split_crossed_sticks(binary, edge_map, unit_area, clump_ratio=1.28):
    """세장비(Elongation)가 매우 큰 막대/면봉이 교차하거나 맞닿은 경우에만 K개로 분할"""
    struct = nd.generate_binary_structure(2, 2)
    lbl, num = nd.label(binary, structure=struct)
    if num == 0:
        return binary, lbl, 0

    sizes = np.bincount(lbl.ravel())
    slices = nd.find_objects(lbl)
    final_lbl = np.zeros_like(lbl, dtype=np.int32)
    final_bin = np.zeros_like(binary, dtype=bool)
    curr_id = 0

    for cid in range(1, num + 1):
        c_area = sizes[cid]
        if c_area < unit_area * 0.35:
            continue
        sl = slices[cid - 1]
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        sub_mask = (lbl[y0:y1, x0:x1] == cid)

        if c_area < unit_area * clump_ratio:
            curr_id += 1
            final_lbl[y0:y1, x0:x1][sub_mask] = curr_id
            final_bin[y0:y1, x0:x1][sub_mask] = True
            continue

        k_num = int(np.clip(round(c_area / unit_area), 2, 6))
        ys, xs = np.nonzero(sub_mask)
        coords = np.column_stack((ys.astype(np.float64), xs.astype(np.float64)))
        sub_edt = nd.distance_transform_edt(sub_mask)
        weights = sub_edt[ys, xs] + 0.1

        centers = np.zeros((k_num, 2), dtype=np.float64)
        centers[0] = coords[int(np.argmax(weights))]
        min_d2 = np.sum((coords - centers[0]) ** 2, axis=1)
        for k in range(1, k_num):
            centers[k] = coords[int(np.argmax(min_d2))]
            min_d2 = np.minimum(min_d2, np.sum((coords - centers[k]) ** 2, axis=1))

        labels_k = np.zeros(len(coords), dtype=np.int32)
        for _ in range(10):
            d2 = np.sum((coords[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels_k = np.argmin(d2, axis=1)
            for k in range(k_num):
                mk = (labels_k == k)
                if np.any(mk):
                    centers[k, 0] = np.mean(coords[mk, 0])
                    centers[k, 1] = np.mean(coords[mk, 1])

        sub_map = np.zeros(sub_mask.shape, dtype=np.int32)
        for k in range(k_num):
            mk = (labels_k == k)
            if np.any(mk):
                curr_id += 1
                sub_map[ys[mk], xs[mk]] = curr_id

        max_nb = nd.maximum_filter(sub_map, size=3)
        min_nb = nd.minimum_filter(np.where(sub_map > 0, sub_map, curr_id + 5), size=3)
        ridge = (sub_map > 0) & (max_nb != min_nb)
        sub_b = (sub_map > 0) & (~ridge)

        final_lbl[y0:y1, x0:x1][sub_map > 0] = sub_map[sub_map > 0]
        final_bin[y0:y1, x0:x1][sub_b] = True

    return final_bin, final_lbl, curr_id

# ---------------------------------------------------------
# 메인 영상처리 파이프라인
# ---------------------------------------------------------
def process_image(img_pil, manual=None):
    gray_full = img_pil.convert("RGB").convert("L")
    gray_resized = resize_image(gray_full, max_dim=512)
    f_raw = np.array(gray_resized, dtype=np.float64)

    med_k = manual["med_k"] if manual is not None else 3
    f_denoised = signal.medfilt2d(f_raw, kernel_size=med_k) if med_k > 1 else f_raw.copy()

    h, w = f_denoised.shape
    pad_h, pad_w = max(2, int(h * 0.04)), max(2, int(w * 0.04))
    border_px = np.concatenate([
        f_denoised[:pad_h, :].ravel(), f_denoised[-pad_h:, :].ravel(),
        f_denoised[:, :pad_w].ravel(), f_denoised[:, -pad_w:].ravel()
    ])
    border_med = float(np.median(border_px))
    border_mad = float(np.median(np.abs(border_px - border_med))) + 1.0
    otsu_g = get_otsu_threshold(f_denoised.astype(np.uint8))

    # [수동 설정 모드]
    if manual is not None:
        is_inv = manual["invert"]
        f_base = (255.0 - f_denoised) if is_inv else f_denoised.copy()
        if manual["use_bg"]:
            bg_est = nd.gaussian_filter(f_base, sigma=manual["bg_sigma"])
            f_corr = np.clip(f_base - bg_est, 0, None)
        else:
            f_corr = f_base

        p99 = np.percentile(f_corr, 99.5)
        f_norm = np.clip(f_corr / (p99 + 1e-6) * 255.0, 0, 255).astype(np.uint8)
        edge_map, valley_map = get_hpf_edge(f_norm)

        f_cut = np.clip(f_norm.astype(np.float64) - manual["hpf_w"] * valley_map.astype(np.float64), 0, 255)
        t_val = manual["thresh"]
        bw = f_cut > t_val
        bw = fill_holes_limited(bw, max_hole_px=300)

        open_r = manual["open_r"]
        if open_r > 0:
            bw = nd.binary_opening(bw, structure=disk_kernel(open_r))

        struct = nd.generate_binary_structure(2, 2)
        lbl, num = nd.label(bw, structure=struct)
        if num > 0:
            sizes = np.bincount(lbl.ravel())
            keep = sizes >= manual["min_area"]
            keep[0] = False
            bw = keep[lbl]
            lbl, num = nd.label(bw, structure=struct)
            if manual["split"] and num > 0:
                edt_m = nd.distance_transform_edt(bw)
                r_m = float(np.percentile(edt_m[edt_m > 0], 80)) if np.any(edt_m > 0) else 6.0
                med_a = float(np.median(np.bincount(lbl.ravel())[1:]))
                bw = split_overlapping_clumps(bw, edge_map, r_m, med_a)
                lbl, num = nd.label(bw, structure=struct)

        centroids = nd.center_of_mass(lbl > 0, lbl, range(1, num + 1)) if num > 0 else []
        return {
            "original_gray": np.array(gray_resized), "corrected": f_norm, "edge_map": edge_map,
            "binary_final": bw, "labeled": lbl, "count": num, "centroids": centroids,
            "thresh": t_val, "open_r": open_r, "is_inverted": is_inv
        }

    # [자동 설정 모드]
    # 1. 가방 속 흰색 숫자 여부 확인 (균일한 검정 가방 본체 내부에 밝은 숫자가 있는 경우)
    dark_reg = remove_border_artifacts(f_denoised < (otsu_g * 0.90), max_area_ratio=0.15)
    lbl_d, num_d = nd.label(dark_reg)
    max_dark_ratio = (np.max(np.bincount(lbl_d.ravel())[1:]) / float(h * w)) if num_d > 0 else 0.0

    if border_med > 140 and 0.16 <= max_dark_ratio <= 0.72:
        bag_id = int(np.argmax(np.bincount(lbl_d.ravel())[1:])) + 1
        bag_mask = (lbl_d == bag_id)
        if float(np.median(f_denoised[bag_mask])) < 60.0 and float(np.std(f_denoised[bag_mask])) < 32.0:
            bag_filled = fill_holes_limited(bag_mask, max_hole_px=2500)
            bag_body = nd.binary_opening(bag_filled, structure=disk_kernel(11))
            bag_body = nd.binary_fill_holes(bag_body)
            bag_roi = nd.binary_erosion(bag_body, structure=disk_kernel(7))

            bg_loc = nd.percentile_filter(f_denoised, percentile=30, size=25)
            f_diff = np.clip(f_denoised - bg_loc, 0, None)
            f_diff[~bag_roi] = 0.0
            p99 = np.percentile(f_diff, 99.5)
            f_norm = np.clip(f_diff / (p99 + 1e-6) * 255.0, 0, 255).astype(np.uint8)
            edge_map, _ = get_hpf_edge(f_norm)

            t_val = max(50, get_otsu_threshold(f_norm[bag_roi]) if np.any(bag_roi) else 85)
            bw = (f_norm > t_val) & bag_roi
            struct = nd.generate_binary_structure(2, 2)
            lbl, num = nd.label(bw, structure=struct)
            if num > 0:
                sizes = np.bincount(lbl.ravel())
                med_a = float(np.median(sizes[1:]))
                keep = (sizes >= max(15, med_a * 0.25)) & (sizes <= med_a * 3.5)
                keep[0] = False
                bw = keep[lbl]
                lbl, num = nd.label(bw, structure=struct)
            centroids = nd.center_of_mass(bw, lbl, range(1, num + 1)) if num > 0 else []
            return {
                "original_gray": np.array(gray_resized), "corrected": f_norm, "edge_map": edge_map,
                "binary_final": bw, "labeled": lbl, "count": num, "centroids": centroids,
                "thresh": t_val, "open_r": 0, "is_inverted": False
            }

    # 2. 타겟 극성(밝은 타겟 vs 어두운 타겟) 자동 판별
    # 얇은 보도블럭 줄눈(음의 틈새)을 먼저 닫아준 뒤 대구경 가우시안 배경과 비교
    f_closed = nd.grey_closing(f_denoised, size=(7, 7))
    bg_broad = nd.gaussian_filter(f_closed, sigma=45.0)

    bright_salience = float(np.percentile(np.clip(f_denoised - bg_broad, 0, None), 99.0))
    dark_salience = float(np.percentile(np.clip(bg_broad - f_closed, 0, None), 99.0))

    # 테두리가 밝고(border_med > 135) 어두운 돌출 에너지가 더 크면 어두운 타겟(귤, 블록, 부품, 동전, 자판)으로 판별
    is_inverted = bool(border_med > 135 and dark_salience > bright_salience * 0.85)

    # 3-A. 밝은 타겟 처리 (면봉, 흰 종이 조각, 보도블럭/아스팔트 위 낙엽, 쌀알 등)
    if not is_inverted:
        f_smooth5 = signal.medfilt2d(f_raw, kernel_size=5)
        f_closed5 = nd.grey_closing(f_smooth5, size=(7, 7))
        bg_surface = nd.gaussian_filter(f_closed5, sigma=38.0)

        # 어두운 배경(면봉, 종이 조각)은 3x3 노이즈 제거 영상 사용, 거친 아스팔트(낙엽)는 5x5 스무딩 영상 사용
        if border_med < 95:
            f_diff = np.clip(f_denoised - bg_surface, 0, None)
            p99 = np.percentile(f_diff, 99.5)
            f_norm = np.clip(f_diff / (p99 + 1e-6) * 255.0, 0, 255).astype(np.uint8)
            edge_map, valley_map = get_hpf_edge(f_norm)

            otsu_t = get_otsu_threshold(f_norm)
            t_val = max(40, int(otsu_t * 0.85))
            bw = fill_holes_limited(f_norm > t_val, max_hole_px=400)

            open_r = 1
            bw = nd.binary_opening(bw, structure=disk_kernel(open_r))
            bw = fill_holes_limited(bw, max_hole_px=400)

            struct = nd.generate_binary_structure(2, 2)
            lbl, num = nd.label(bw, structure=struct)
            if num > 0:
                edt = nd.distance_transform_edt(bw)
                comp_areas = np.bincount(lbl.ravel())[1:]
                comp_r = nd.maximum(edt, lbl, index=range(1, num + 1))
                elongations = comp_areas / (np.pi * (np.maximum(comp_r, 1.0) ** 2))

                # 세장비가 4.0 이상인 얇고 긴 막대(면봉)인 경우에만 교차/접촉 분할 수행 (종이 조각은 분할 안 함)
                if float(np.median(elongations)) > 4.0:
                    cut_px = (valley_map > np.percentile(valley_map, 88)) & (edt > 0) & (edt <= 2.5)
                    bw = nd.binary_opening(bw & (~cut_px), structure=disk_kernel(1))
                    lbl, num = nd.label(bw, structure=struct)
                    sizes = np.bincount(lbl.ravel())[1:]
                    valid_s = sizes[sizes >= 20]
                    if len(valid_s) > 0:
                        unit_a = float(np.percentile(valid_s, 40))
                        bw, lbl, num = split_crossed_sticks(bw, edge_map, unit_a, clump_ratio=1.28)
                else:
                    med_a = float(np.median(comp_areas))
                    keep = np.bincount(lbl.ravel()) >= max(15, med_a * 0.22)
                    keep[0] = False
                    bw = keep[lbl]
                    lbl, num = nd.label(bw, structure=struct)
        else:
            # 아스팔트·보도블럭 위 낙엽 처리 (단일 낙엽 쪼개짐 방지 및 자갈 노이즈 억제)
            f_leaf = nd.gaussian_filter(f_smooth5, sigma=1.0)
            f_diff = np.clip(f_leaf - bg_surface, 0, None)

            bg_floor = f_diff[f_diff <= np.percentile(f_diff, 85)]
            noise_std = float(np.std(bg_floor)) + 1.0
            raw_thresh = max(18.0, float(np.mean(bg_floor)) + 5.0 * noise_std)

            max_scale = max(float(np.percentile(f_diff, 99.8)), 42.0)
            f_norm = np.clip(f_diff / max_scale * 255.0, 0, 255).astype(np.uint8)
            edge_map, _ = get_hpf_edge(f_norm)

            t_val = int(np.clip(raw_thresh / max_scale * 255.0, 42, 125))
            bw = fill_holes_limited(f_norm > t_val, max_hole_px=300)

            open_r = 1
            bw = nd.binary_opening(bw, structure=disk_kernel(open_r))
            bw = fill_holes_limited(bw, max_hole_px=300)

            struct = nd.generate_binary_structure(2, 2)
            lbl, num = nd.label(bw, structure=struct)
            if num > 0:
                sizes = np.bincount(lbl.ravel())
                valid_fg = sizes[1:][sizes[1:] >= 12]
                med_a = float(np.median(valid_fg)) if len(valid_fg) > 0 else 16.0
                keep = (sizes >= max(12, int(med_a * 0.22))) & (sizes <= (bw.size * 0.03))
                keep[0] = False
                bw = keep[lbl]
                lbl, num = nd.label(bw, structure=struct)

        centroids = nd.center_of_mass(lbl > 0, lbl, range(1, num + 1)) if num > 0 else []
        return {
            "original_gray": np.array(gray_resized), "corrected": f_norm, "edge_map": edge_map,
            "binary_final": bw, "labeled": lbl, "count": num, "centroids": centroids,
            "thresh": t_val, "open_r": open_r, "is_inverted": False
        }

    # 3-B. 어두운 타겟 처리 (귤, 블록, 소형 부품, 키보드 자판, 밀집 동전)
    # 배경 조명 그라데이션 추정
    bg_high = nd.gaussian_filter(nd.grey_dilation(f_denoised, size=(15, 15)), sigma=45.0)
    bg_high = np.maximum(bg_high, bg_broad)
    f_diff = np.clip(bg_high - f_denoised, 0, None)

    p99 = np.percentile(f_diff, 99.5)
    f_norm = np.clip(f_diff / (p99 + 1e-6) * 255.0, 0, 255).astype(np.uint8)
    edge_map, valley_map = get_hpf_edge(f_norm)

    otsu_t = get_otsu_threshold(f_norm)
    t_val = int(np.clip(otsu_t * 0.88, 45, 140))

    bw = remove_border_artifacts(f_norm > t_val, max_area_ratio=0.05)

    # 평탄한 회색 바닥 위에 어두운 부품과 밝은 부품이 함께 있는 경우(image_94e0fd) 밝은 부품 병합
    if 135 <= border_med <= 200 and border_mad < 12.0:
        bright_parts = remove_border_artifacts((f_denoised - bg_broad) > 38.0, max_area_ratio=0.01)
        bright_parts = nd.binary_opening(bright_parts, structure=disk_kernel(1))
        bw = bw | bright_parts

    bw = fill_holes_limited(bw, max_hole_px=250)

    # 객체 두께(내접 반경) 측정
    edt = nd.distance_transform_edt(bw)
    peaks = (edt == nd.maximum_filter(edt, size=7)) & (edt > 2.0)
    r_med = float(np.median(edt[peaks])) if np.any(peaks) else 7.0
    r_med = float(np.clip(r_med, 2.5, 38.0))

    # 접합부 목(Neck) 부분에만 HPF 골짜기 차감 적용
    neck_zone = (edt > 0) & (edt <= r_med * 0.55)
    cut_px = (valley_map > np.percentile(valley_map, 85)) & neck_zone
    bw_carved = bw & (~cut_px)

    open_r = 1 if r_med < 8.0 else 2
    bw_opened = nd.binary_opening(bw_carved, structure=disk_kernel(open_r))
    bw_opened = fill_holes_limited(bw_opened, max_hole_px=250)

    # 얇은 그림자 파편 제거 (정상 객체 두께 r_med의 50% 미만인 얇은 초승달 그림자 소거)
    struct = nd.generate_binary_structure(2, 2)
    lbl_1, num_1 = nd.label(bw_opened, structure=struct)
    if num_1 > 0:
        edt_1 = nd.distance_transform_edt(bw_opened)
        comp_max_r = np.array(nd.maximum(edt_1, lbl_1, index=range(1, num_1 + 1)))
        comp_sizes = np.bincount(lbl_1.ravel())[1:]

        thick_mask = comp_max_r >= (r_med * 0.50)
        valid_areas = comp_sizes[thick_mask] if np.any(thick_mask) else comp_sizes
        med_area = float(np.percentile(valid_areas, 45)) if len(valid_areas) > 0 else 25.0

        keep_ids = np.zeros(num_1 + 1, dtype=bool)
        for idx_c in range(num_1):
            if comp_max_r[idx_c] >= (r_med * 0.50) and comp_sizes[idx_c] >= (med_area * 0.38):
                keep_ids[idx_c + 1] = True
        bw_clean = keep_ids[lbl_1]

        # 겹친 블록이나 맞닿은 두 블록만 조건부 분할 (단일 블록이나 내부 에지 없는 큰 부품은 보존)
        bw_final = split_overlapping_clumps(bw_clean, edge_map, r_med, med_area)
        lbl_f, num_f = nd.label(bw_final, structure=struct)
        sizes_f = np.bincount(lbl_f.ravel())
        keep_f = sizes_f >= (med_area * 0.35)
        keep_f[0] = False
        bw_final = keep_f[lbl_f]
        lbl_f, num_f = nd.label(bw_final, structure=struct)
    else:
        bw_final, lbl_f, num_f = bw_opened, lbl_1, 0

    centroids = nd.center_of_mass(lbl_f > 0, lbl_f, range(1, num_f + 1)) if num_f > 0 else []
    return {
        "original_gray": np.array(gray_resized), "corrected": f_norm, "edge_map": edge_map,
        "binary_final": bw_final, "labeled": lbl_f, "count": num_f, "centroids": centroids,
        "thresh": t_val, "open_r": open_r, "is_inverted": True
    }

# ---------------------------------------------------------
# 사이드바 및 메인 화면 UI
# ---------------------------------------------------------
st.sidebar.header("파라미터 설정")
mode = st.sidebar.radio("작동 모드", ["자동 설정", "수동 설정"], index=0)

manual_cfg = None
if mode == "수동 설정":
    invert_opt = st.sidebar.checkbox("명암 반전 (밝은 배경 / 어두운 객체)", value=True)
    med_k_opt = st.sidebar.slider("Median Filter 커널 크기", 1, 11, 3, step=2)
    use_bg_opt = st.sidebar.checkbox("배경 조명 보정", value=True)
    bg_sig_opt = st.sidebar.slider("배경 가우시안 Sigma", 15.0, 80.0, 40.0, step=5.0, disabled=not use_bg_opt)
    hpf_w_opt = st.sidebar.slider("HPF 에지 차감 강도", 0.0, 1.2, 0.45, step=0.05)
    thresh_opt = st.sidebar.slider("이진화 임계값 (Threshold)", 5, 250, 75, step=1)
    open_r_opt = st.sidebar.slider("Opening 커널 반경 (R)", 0, 6, 1, step=1)
    min_area_opt = st.sidebar.slider("최소 객체 면적 (px)", 5, 300, 20, step=5)
    split_opt = st.sidebar.checkbox("붙은 객체 분할", value=True)

    manual_cfg = {
        "invert": invert_opt, "med_k": med_k_opt, "use_bg": use_bg_opt,
        "bg_sigma": bg_sig_opt, "hpf_w": hpf_w_opt, "thresh": thresh_opt,
        "open_r": open_r_opt, "min_area": min_area_opt, "split": split_opt
    }

uploaded_files = st.file_uploader(
    "이미지 파일 업로드",
    type=["tif", "tiff", "png", "jpg", "jpeg", "bmp"],
    accept_multiple_files=True
)

if uploaded_files:
    summary_data = []
    tabs = st.tabs([file.name for file in uploaded_files])

    for idx, (tab, uploaded_file) in enumerate(zip(tabs, uploaded_files)):
        img_pil = Image.open(uploaded_file)

        with tab:
            with st.expander("개별 파라미터 수동 조정"):
                c1, c2, c3, c4 = st.columns(4)
                use_ind = c1.checkbox("수동 적용", value=False, key=f"ind_{idx}")
                ind_inv = c1.checkbox("명암 반전", value=True, key=f"inv_{idx}", disabled=not use_ind)
                ind_t = c2.slider("임계값 (T)", 5, 250, 75, 1, key=f"t_{idx}", disabled=not use_ind)
                ind_r = c3.slider("Opening 반경 (R)", 0, 6, 1, 1, key=f"r_{idx}", disabled=not use_ind)
                ind_min = c4.slider("최소 면적 (px)", 5, 300, 20, 5, key=f"m_{idx}", disabled=not use_ind)

            if use_ind:
                local_manual = {
                    "invert": ind_inv, "med_k": 3, "use_bg": True, "bg_sigma": 40.0,
                    "hpf_w": 0.45, "thresh": ind_t, "open_r": ind_r,
                    "min_area": ind_min, "split": True
                }
                res = process_image(img_pil, manual=local_manual)
            else:
                res = process_image(img_pil, manual=manual_cfg)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("검출된 객체 수", f"{res['count']} 개")
            m2.metric("적용 임계값", f"T = {res['thresh']}")
            m3.metric("Opening 반경", f"R = {res['open_r']}")
            m4.metric("명암 반전", "적용됨" if res["is_inverted"] else "원본 유지")

            fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))

            axes[0].imshow(res["original_gray"], cmap="gray")
            axes[0].set_title("1. Grayscale Input")
            axes[0].axis("off")

            axes[1].imshow(res["corrected"], cmap="gray")
            axes[1].set_title("2. Preprocessed")
            axes[1].axis("off")

            axes[2].imshow(res["edge_map"], cmap="inferno")
            axes[2].set_title("3. HPF Edge")
            axes[2].axis("off")

            axes[3].imshow(res["binary_final"], cmap="gray")
            axes[3].set_title(f"4. Binary (R={res['open_r']}, T={res['thresh']})")
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
            "해상도": f"{img_pil.width} x {img_pil.height}",
            "검출된 객체 수": res["count"],
            "적용 임계값": res["thresh"],
            "Opening 반경": res["open_r"]
        })

    st.divider()
    st.subheader("전체 카운팅 결과 요약")
    st.dataframe(summary_data, use_container_width=True)
