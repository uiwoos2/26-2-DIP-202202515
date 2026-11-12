import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.ndimage as nd
from PIL import Image

st.set_page_config(page_title="Object Counting", layout="wide")
st.title("Object Counting")

# 1. 기본 유틸리티 함수
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

def fill_holes_limited(binary, max_hole_px=300):
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

def remove_border_regions(binary, max_area_ratio=0.04):
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

# 2. HPF 에지 추출 (Sobel + Laplacian)
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

# 3. 겹친 객체 및 교차 객체 분할 함수
def split_touching_objects(binary, edge_map, unit_area, clump_ratio=1.45, min_ratio=0.35):
    struct = nd.generate_binary_structure(2, 2)
    lbl, num = nd.label(binary, structure=struct)
    if num == 0 or unit_area <= 10:
        return binary, lbl, num

    sizes = np.bincount(lbl.ravel())
    slices = nd.find_objects(lbl)

    final_lbl = np.zeros_like(lbl, dtype=np.int32)
    final_bin = np.zeros_like(binary, dtype=bool)
    curr_id = 0

    for cid in range(1, num + 1):
        c_area = sizes[cid]
        if c_area < unit_area * min_ratio:
            continue

        sl = slices[cid - 1]
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        sub_mask = (lbl[y0:y1, x0:x1] == cid)

        if c_area < unit_area * clump_ratio:
            curr_id += 1
            final_lbl[y0:y1, x0:x1][sub_mask] = curr_id
            final_bin[y0:y1, x0:x1][sub_mask] = True
            continue

        k_num = int(np.clip(round(c_area / unit_area), 2, 120))
        ys, xs = np.nonzero(sub_mask)
        coords = np.column_stack((ys.astype(np.float64), xs.astype(np.float64)))

        if len(coords) < k_num * 5:
            curr_id += 1
            final_lbl[y0:y1, x0:x1][sub_mask] = curr_id
            final_bin[y0:y1, x0:x1][sub_mask] = True
            continue

        sub_edt = nd.distance_transform_edt(sub_mask)
        sub_edge = edge_map[y0:y1, x0:x1].astype(np.float64) / 255.0
        weights = sub_edt[ys, xs] * (1.0 - 0.55 * sub_edge[ys, xs]) + 0.1

        centers = np.zeros((k_num, 2), dtype=np.float64)
        centers[0] = coords[int(np.argmax(weights))]
        min_d2 = np.sum((coords - centers[0]) ** 2, axis=1)

        for k in range(1, k_num):
            idx = int(np.argmax(min_d2 * (weights ** 0.5)))
            centers[k] = coords[idx]
            min_d2 = np.minimum(min_d2, np.sum((coords - centers[k]) ** 2, axis=1))

        labels_k = np.zeros(len(coords), dtype=np.int32)
        for _ in range(12):
            d2 = np.sum((coords[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels_k = np.argmin(d2, axis=1)
            for k in range(k_num):
                mk = (labels_k == k)
                if np.any(mk):
                    wk = weights[mk]
                    centers[k, 0] = np.sum(coords[mk, 0] * wk) / np.sum(wk)
                    centers[k, 1] = np.sum(coords[mk, 1] * wk) / np.sum(wk)

        sub_map = np.zeros(sub_mask.shape, dtype=np.int32)
        for k in range(k_num):
            mk = (labels_k == k)
            if np.any(mk):
                curr_id += 1
                sub_map[ys[mk], xs[mk]] = curr_id

        max_nb = nd.maximum_filter(sub_map, size=3)
        min_nb = nd.minimum_filter(np.where(sub_map > 0, sub_map, curr_id + 10), size=3)
        ridge = (sub_map > 0) & (max_nb != min_nb)

        sub_b = sub_map > 0
        sub_b[ridge] = False

        final_lbl[y0:y1, x0:x1][sub_map > 0] = sub_map[sub_map > 0]
        final_bin[y0:y1, x0:x1][sub_b] = True

    return final_bin, final_lbl, curr_id

# 4. 밀집 원형 객체(동전 등) 전용 거리-에지 분할 함수
def segment_packed_circles(f_raw, f_denoised, border_med, manual=None):
    f_diff = np.clip(border_med - f_denoised, 0, None)
    p99 = np.percentile(f_diff, 99.5)
    f_norm = np.clip(f_diff / (p99 + 1e-6) * 255.0, 0, 255).astype(np.uint8)
    edge_map, valley_map = get_hpf_edge(f_raw)

    # 밝은 은색 동전도 누락되지 않도록 흰 배경(border_med) 기준 임계값 설정
    t_val = 28 if manual is None else manual["thresh"]
    bw = f_norm > t_val
    bw = remove_border_regions(bw, max_area_ratio=0.05)
    bw = fill_holes_limited(bw, max_hole_px=35)

    open_r = 1 if manual is None else manual["open_r"]
    if open_r > 0:
        bw = nd.binary_opening(bw, structure=disk_kernel(open_r))

    # 동전 내부 무늬는 스무딩하고 동전 외곽 테두리 에지와 거리 변환을 결합해 동전 중심점 검출
    edt = nd.distance_transform_edt(bw)
    r_est = float(np.percentile(edt[edt > 0], 82)) if np.any(edt > 0) else 15.0
    r_est = float(np.clip(r_est, 8.0, 32.0))

    f_smooth_dome = nd.gaussian_filter(f_norm.astype(np.float64), sigma=r_est * 0.35)
    edge_smooth = nd.gaussian_filter(edge_map.astype(np.float64) / 255.0, sigma=1.5)
    dome_score = (edt / (r_est + 1e-6)) * 0.65 + (f_smooth_dome / 255.0) * 0.35 - 0.30 * edge_smooth
    dome_score = nd.gaussian_filter(dome_score, sigma=max(1.2, r_est * 0.2))
    dome_score[~bw] = 0.0

    fp_size = max(7, int(round(r_est * 1.25)))
    local_max = (dome_score == nd.maximum_filter(dome_score, size=fp_size)) & (edt > r_est * 0.38)
    local_max = nd.binary_dilation(local_max, structure=disk_kernel(max(1, int(r_est * 0.18))))

    struct = nd.generate_binary_structure(2, 2)
    markers, num_seeds = nd.label(local_max, structure=struct)

    if num_seeds > 1:
        markers_ws = markers.astype(np.int32)
        markers_ws[~bw] = num_seeds + 1
        cost = (255.0 - np.clip(dome_score / (dome_score.max() + 1e-6) * 200.0, 0, 200) + edge_smooth * 55.0).astype(np.uint8)
        ws = nd.watershed_ift(cost, markers_ws)
        ws_fg = np.where((ws >= 1) & (ws <= num_seeds), ws, 0)

        max_nb = nd.maximum_filter(ws_fg, size=3)
        min_nb = nd.minimum_filter(np.where(ws_fg > 0, ws_fg, num_seeds + 5), size=3)
        ridge = (ws_fg > 0) & (max_nb != min_nb)
        bw_final = bw & (~ridge)

        min_coin_area = max(25, int(np.pi * (r_est ** 2) * 0.22))
        lbl_f, num_f = nd.label(bw_final, structure=struct)
        sizes_f = np.bincount(lbl_f.ravel())
        keep = sizes_f >= min_coin_area
        keep[0] = False
        bw_final = keep[lbl_f]
        lbl_f, num_f = nd.label(bw_final, structure=struct)
        return f_norm, edge_map, bw_final, lbl_f, num_f, t_val, open_r, True

    lbl_f, num_f = nd.label(bw, structure=struct)
    return f_norm, edge_map, bw, lbl_f, num_f, t_val, open_r, True

# 5. 메인 영상처리 함수
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
    global_med = float(np.median(f_denoised))
    otsu_g = get_otsu_threshold(f_denoised.astype(np.uint8))

    # 수동 설정이 활성화된 경우
    if manual is not None:
        is_inv = manual["invert"]
        f_base = (255.0 - f_denoised) if is_inv else f_denoised.copy()
        if manual["use_bg"]:
            f_small = f_base[::4, ::4]
            bg_s = nd.percentile_filter(f_small, percentile=45, size=21)
            bg_s = nd.gaussian_filter(bg_s, sigma=4.0)
            bg_full = np.array(Image.fromarray(bg_s.astype(np.float32)).resize((w, h), Image.Resampling.BILINEAR), dtype=np.float64)
            f_corr = np.clip(f_base - bg_full, 0, None)
        else:
            f_corr = f_base

        p99 = np.percentile(f_corr, 99.5)
        f_norm = np.clip(f_corr / (p99 + 1e-6) * 255.0, 0, 255).astype(np.uint8)
        edge_map, valley_map = get_hpf_edge(f_norm)

        f_cut = np.clip(f_norm.astype(np.float64) - manual["hpf_w"] * valley_map.astype(np.float64), 0, 255)
        t_val = manual["thresh"]
        bw = f_cut > t_val
        bw = fill_holes_limited(bw, max_hole_px=250)

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
            if manual["split"]:
                valid_s = sizes[1:][sizes[1:] >= manual["min_area"]]
                if len(valid_s) > 0:
                    unit_a = float(np.percentile(valid_s, 35))
                    bw, lbl, num = split_touching_objects(bw, edge_map, unit_a, clump_ratio=1.45, min_ratio=0.3)
            else:
                lbl, num = nd.label(bw, structure=struct)

        centroids = nd.center_of_mass(lbl > 0, lbl, range(1, num + 1)) if num > 0 else []
        return {
            "original_gray": np.array(gray_resized),
            "corrected": f_norm,
            "edge_map": edge_map,
            "binary_final": bw,
            "labeled": lbl,
            "count": num,
            "centroids": centroids,
            "thresh": t_val,
            "open_r": open_r,
            "is_inverted": is_inv
        }

    # --- 이하 자동 처리 모드 ---
    # Case 1: 밝은 배경 위 검은 가방 속 흰색 숫자 (가방 내부가 매우 어둡고 균일한 경우만 해당)
    dark_region = remove_border_regions(f_denoised < (otsu_g * 0.90), max_area_ratio=0.15)
    lbl_d, num_d = nd.label(dark_region)
    max_dark_ratio = (np.max(np.bincount(lbl_d.ravel())[1:]) / float(h * w)) if num_d > 0 else 0.0

    if border_med > 140 and 0.16 <= max_dark_ratio <= 0.72:
        bag_id = int(np.argmax(np.bincount(lbl_d.ravel())[1:])) + 1
        bag_mask = (lbl_d == bag_id)
        bag_med_intensity = float(np.median(f_denoised[bag_mask]))
        bag_std_intensity = float(np.std(f_denoised[bag_mask]))

        # 동전 무리(밝기 중앙값 ~125, 표준편차 >35)와 구분하기 위해 실제 검은 천(중앙값 < 60)인지 확인
        if bag_med_intensity < 60.0 and bag_std_intensity < 32.0:
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

    # Case 2: 순백색 배경 위에 동전들이 고밀도로 빽빽하게 밀집한 경우 (image_94f742 / IMG_9637)
    if border_med > 225 and max_dark_ratio > 0.30:
        f_norm, edge_map, bw, lbl, num, t_val, open_r, is_inv = segment_packed_circles(f_raw, f_denoised, border_med)
        centroids = nd.center_of_mass(lbl > 0, lbl, range(1, num + 1)) if num > 0 else []
        return {
            "original_gray": np.array(gray_resized), "corrected": f_norm, "edge_map": edge_map,
            "binary_final": bw, "labeled": lbl, "count": num, "centroids": centroids,
            "thresh": t_val, "open_r": open_r, "is_inverted": is_inv
        }

    # 대구경 표면 배경 추정 (객체 내부가 파이지 않도록 1/4 축소 후 23x23 = 92px 대구경 윈도우 사용)
    f_smooth5 = signal.medfilt2d(f_raw, kernel_size=5)
    f_small = f_smooth5[::4, ::4]
    # 어두운 줄눈을 먼저 닫아준 뒤 중위 표면 추정
    f_closed_s = nd.grey_closing(f_small, size=(5, 5))
    bg_surf_s = nd.percentile_filter(f_closed_s, percentile=52, size=23)
    bg_surf_s = nd.gaussian_filter(bg_surf_s, sigma=4.0)
    bg_surface = np.array(Image.fromarray(bg_surf_s.astype(np.float32)).resize((w, h), Image.Resampling.BILINEAR), dtype=np.float64)

    bg_upper_s = nd.percentile_filter(f_small, percentile=88, size=23)
    bg_upper_s = nd.gaussian_filter(bg_upper_s, sigma=4.0)
    bg_upper = np.array(Image.fromarray(bg_upper_s.astype(np.float32)).resize((w, h), Image.Resampling.BILINEAR), dtype=np.float64)

    # 밝은 방향 편차(f - bg)와 어두운 방향 편차(bg - f) 비교
    diff_bright = np.clip(f_denoised - bg_surface, 0, None)
    diff_dark = np.clip(bg_upper - f_denoised, 0, None)

    # Case 3: 어두운 배경 위 밝은 객체 (면봉, 쌀알, 어두운 바닥 위 흰 종이 조각 image_954d99)
    if border_med < 125 and np.percentile(diff_bright, 98) > np.percentile(diff_dark, 98) * 0.8:
        p99 = np.percentile(diff_bright, 99.5)
        f_norm = np.clip(diff_bright / (p99 + 1e-6) * 255.0, 0, 255).astype(np.uint8)
        edge_map, valley_map = get_hpf_edge(f_norm)

        otsu_t = get_otsu_threshold(f_norm)
        t_val = max(38, int(otsu_t * 0.82))

        # 객체 내부가 절대 파이지 않도록 이진화 후 구멍을 먼저 메움
        bw = f_norm > t_val
        bw = fill_holes_limited(bw, max_hole_px=350)

        # 나란히 붙은 면봉 등의 경계 골짜기 미세 절단
        edt = nd.distance_transform_edt(bw)
        r_med = float(np.percentile(edt[edt > 0], 80)) if np.any(edt > 0) else 4.0
        cut_line = (valley_map > np.percentile(valley_map, 88)) & (edt > 0) & (edt <= max(2.2, r_med * 0.45))
        bw = bw & (~cut_line)

        open_r = 1
        bw = nd.binary_opening(bw, structure=disk_kernel(open_r))
        bw = fill_holes_limited(bw, max_hole_px=350)

        struct = nd.generate_binary_structure(2, 2)
        lbl, num = nd.label(bw, structure=struct)
        if num > 0:
            sizes = np.bincount(lbl.ravel())[1:]
            valid_s = sizes[sizes >= 20]
            if len(valid_s) > 0:
                # 하위 30% 백분위수로 단일 객체 1개의 기준 면적을 산출하여 교차된 면봉(2배 면적)만 정확히 분할
                # (흰 종이 조각 image_954d99는 면적이 균일하므로 쪼개지지 않음)
                unit_area = float(np.percentile(valid_s, 30))
                bw, lbl, num = split_touching_objects(bw, edge_map, unit_area, clump_ratio=1.55, min_ratio=0.32)

        centroids = nd.center_of_mass(lbl > 0, lbl, range(1, num + 1)) if num > 0 else []
        return {
            "original_gray": np.array(gray_resized), "corrected": f_norm, "edge_map": edge_map,
            "binary_final": bw, "labeled": lbl, "count": num, "centroids": centroids,
            "thresh": t_val, "open_r": open_r, "is_inverted": False
        }

    # 어두운 객체 신호와 밝은 객체 신호 중 어느 쪽이 실제 타겟인지 판별 (image_954d42 vs 낙엽 사진 구분)
    dark_blobs = remove_border_regions(diff_dark > 28.0, max_area_ratio=0.04)
    _, n_dark_cand = nd.label(nd.binary_opening(dark_blobs, structure=disk_kernel(1)))

    bright_blobs = remove_border_regions(diff_bright > 28.0, max_area_ratio=0.04)
    _, n_bright_cand = nd.label(nd.binary_opening(bright_blobs, structure=disk_kernel(1)))

    # Case 4: 보도블럭·아스팔트 위 밝은 낙엽들 (밝은 후보가 우세하고 배경이 순백색이 아닌 경우)
    if border_med < 185 and n_bright_cand > n_dark_cand * 1.2:
        # 아스팔트 미세 자갈 노이즈는 스무딩하고 대구경 배경 차감으로 큰 은행잎(image_954a34) 내부를 100% 보존
        f_leaf_smooth = nd.gaussian_filter(f_smooth5, sigma=1.1)
        f_leaf_diff = np.clip(f_leaf_smooth - bg_surface, 0, None)

        # 배경 노이즈 바닥(하위 85% 화소)의 표준편차를 직접 측정하여 자갈 폭증(354/432/928개) 방지
        bg_floor = f_leaf_diff[f_leaf_diff <= np.percentile(f_leaf_diff, 85)]
        noise_std = float(np.std(bg_floor)) + 1.0
        raw_thresh = max(16.5, float(np.mean(bg_floor)) + 4.8 * noise_std)

        max_scale = max(float(np.percentile(f_leaf_diff, 99.8)), 40.0)
        f_norm = np.clip(f_leaf_diff / max_scale * 255.0, 0, 255).astype(np.uint8)
        edge_map, _ = get_hpf_edge(f_norm)

        t_val = int(np.clip(raw_thresh / max_scale * 255.0, 38, 120))
        bw = f_norm > t_val
        bw = fill_holes_limited(bw, max_hole_px=250)

        open_r = 1
        bw = nd.binary_opening(bw, structure=disk_kernel(open_r))

        struct = nd.generate_binary_structure(2, 2)
        lbl, num = nd.label(bw, structure=struct)
        if num > 0:
            sizes = np.bincount(lbl.ravel())
            valid_fg = sizes[1:][sizes[1:] >= 12]
            med_a = float(np.median(valid_fg)) if len(valid_fg) > 0 else 15.0
            min_a = max(12, int(med_a * 0.22))
            keep = (sizes >= min_a) & (sizes <= (bw.size * 0.025))
            keep[0] = False
            bw = keep[lbl]
            lbl, num = nd.label(bw, structure=struct)

        centroids = nd.center_of_mass(bw, lbl, range(1, num + 1)) if num > 0 else []
        return {
            "original_gray": np.array(gray_resized), "corrected": f_norm, "edge_map": edge_map,
            "binary_final": bw, "labeled": lbl, "count": num, "centroids": centroids,
            "thresh": t_val, "open_r": open_r, "is_inverted": False
        }

    # Case 5: 밝은/회색 배경 위 어두운 객체 (블록, 키보드 자판, 회색 바닥 위 부품 image_954d42 등)
    p99 = np.percentile(diff_dark, 99.5)
    f_norm_lin = np.clip(diff_dark / (p99 + 1e-6) * 255.0, 0, 255)
    f_norm = np.clip(((f_norm_lin / 255.0) ** 0.85) * 255.0, 0, 255).astype(np.uint8)
    edge_map, valley_map = get_hpf_edge(f_norm)

    otsu_t = get_otsu_threshold(f_norm)
    t_val = int(np.clip(otsu_t * 0.88, 48, 145))

    # 키보드 배열인지 판별
    bw_test = remove_border_regions(f_norm > t_val)
    lbl_t, num_t = nd.label(bw_test)
    rect_fills = []
    if num_t >= 25:
        sizes_t = np.bincount(lbl_t.ravel())[1:]
        for idx, sl in enumerate(nd.find_objects(lbl_t)):
            if sl is not None and sizes_t[idx] > 40:
                ba = (sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start)
                rect_fills.append(sizes_t[idx] / float(max(1, ba)))

    if len(rect_fills) >= 25 and np.median(rect_fills) > 0.74:
        # 키보드 격자 분리
        f_cut = np.clip(f_norm.astype(np.float64) - 0.65 * valley_map.astype(np.float64), 0, 255)
        bw = remove_border_regions(f_cut > t_val, max_area_ratio=0.03)
        open_r = 2
        bw = nd.binary_opening(bw, structure=disk_kernel(open_r))
        bw = fill_holes_limited(bw, max_hole_px=80)
        struct = nd.generate_binary_structure(2, 2)
        lbl, num = nd.label(bw, structure=struct)
        if num > 0:
            sizes = np.bincount(lbl.ravel())
            med_a = float(np.median(sizes[1:]))
            keep = (sizes >= max(20, med_a * 0.25)) & (sizes <= med_a * 7.5)
            keep[0] = False
            bw = keep[lbl]
            lbl, num = nd.label(bw, structure=struct)
    else:
        # 일반 입체 블록 및 소형 부품 (image_954d42, 겹친 블록 image_95cce0, 20개 블록 image_95c93b)
        bw = remove_border_regions(f_norm > t_val, max_area_ratio=0.05)

        # image_954d42처럼 회색 바닥 위에 어두운 부품들과 함께 같은 크기의 흰색 부품이 섞여 있는 경우 병합
        if 125 <= border_med <= 200 and border_mad < 12.0:
            bright_parts = remove_border_regions(diff_bright > 38.0, max_area_ratio=0.01)
            bright_parts = nd.binary_opening(bright_parts, structure=disk_kernel(1))
            bw = bw | bright_parts

        bw = fill_holes_limited(bw, max_hole_px=180)
        edt = nd.distance_transform_edt(bw)
        peaks = (edt == nd.maximum_filter(edt, size=7)) & (edt > 2.0)
        r_unit = float(np.median(edt[peaks])) if np.any(peaks) else 6.0
        r_unit = float(np.clip(r_unit, 2.5, 35.0))

        cut_px = (valley_map > np.percentile(valley_map, 85)) & (edt > 0) & (edt <= r_unit * 0.6)
        bw_carved = bw & (~cut_px)

        open_r = 1 if r_unit < 7.0 else int(np.clip(round(r_unit * 0.22), 1, 3))
        bw_opened = nd.binary_opening(bw_carved, structure=disk_kernel(open_r))
        bw_opened = fill_holes_limited(bw_opened, max_hole_px=180)

        struct = nd.generate_binary_structure(2, 2)
        lbl, num = nd.label(bw_opened, structure=struct)
        if num > 0:
            sizes = np.bincount(lbl.ravel())[1:]
            min_prelim = max(10, int(np.pi * (r_unit ** 2) * 0.20))
            valid_s = sizes[sizes >= min_prelim]
            if len(valid_s) > 0:
                unit_area = float(np.percentile(valid_s, 32))
                bw, lbl, num = split_touching_objects(
                    bw_opened, edge_map, unit_area, clump_ratio=1.48, min_ratio=0.40
                )

    centroids = nd.center_of_mass(lbl > 0, lbl, range(1, num + 1)) if num > 0 else []
    return {
        "original_gray": np.array(gray_resized), "corrected": f_norm, "edge_map": edge_map,
        "binary_final": bw, "labeled": lbl, "count": num, "centroids": centroids,
        "thresh": t_val, "open_r": open_r, "is_inverted": True
    }

# 6. 사이드바 UI (자동 / 수동 선택)
st.sidebar.header("파라미터 설정")
mode = st.sidebar.radio("작동 모드", ["자동 설정", "수동 설정"], index=0)

manual_cfg = None
if mode == "수동 설정":
    invert_opt = st.sidebar.checkbox("명암 반전 (밝은 배경 / 어두운 객체)", value=True)
    med_k_opt = st.sidebar.slider("Median Filter 커널 크기", 1, 11, 5, step=2)
    use_bg_opt = st.sidebar.checkbox("배경 조명 제거", value=True)
    hpf_w_opt = st.sidebar.slider("HPF 에지 차감 강도", 0.0, 1.2, 0.5, step=0.05)
    thresh_opt = st.sidebar.slider("이진화 임계값 (Threshold)", 5, 250, 75, step=1)
    open_r_opt = st.sidebar.slider("Opening 커널 반경 (R)", 0, 6, 1, step=1)
    min_area_opt = st.sidebar.slider("최소 객체 면적 (px)", 5, 300, 20, step=5)
    split_opt = st.sidebar.checkbox("겹친 객체 자동 분할", value=True)

    manual_cfg = {
        "invert": invert_opt,
        "med_k": med_k_opt,
        "use_bg": use_bg_opt,
        "hpf_w": hpf_w_opt,
        "thresh": thresh_opt,
        "open_r": open_r_opt,
        "min_area": min_area_opt,
        "split": split_opt
    }

# 7. 메인 화면: 이미지 업로드 및 결과 출력
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
            with st.expander("현재 이미지 수동 파라미터 조정"):
                c1, c2, c3, c4 = st.columns(4)
                use_ind = c1.checkbox("개별 수동 적용", value=False, key=f"ind_{idx}")
                ind_inv = c1.checkbox("명암 반전", value=True, key=f"inv_{idx}", disabled=not use_ind)
                ind_t = c2.slider("임계값 (T)", 5, 250, 75, 1, key=f"t_{idx}", disabled=not use_ind)
                ind_r = c3.slider("Opening 반경 (R)", 0, 6, 1, 1, key=f"r_{idx}", disabled=not use_ind)
                ind_min = c4.slider("최소 면적 (px)", 5, 300, 20, 5, key=f"m_{idx}", disabled=not use_ind)

            if use_ind:
                local_manual = {
                    "invert": ind_inv, "med_k": 3, "use_bg": True,
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
