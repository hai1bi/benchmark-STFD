"""
audit_stfd.py
=============
Audit bộ STFD (Screenshot Text Forgery Dataset) trực tiếp từ file zip đã mã hóa AES.

Chạy:
    python audit_stfd.py [--zip data/stfd/STFD_ICASSP2023.zip] [--limit N]

Đầu ra:
    stfd_stats.csv         mỗi dòng một ảnh (cột theo yêu cầu + cột phụ ở cuối)
    data/stfd_summary.json số liệu tổng hợp dùng để viết stfd_audit.md

Mật khẩu giải nén do tác giả công bố tại README của github.com/ZeqinYu/STFL-Net.
Không giải nén ra đĩa: mọi thứ đọc từ zip vào bộ nhớ.
"""
import argparse
import collections
import hashlib
import io
import json
import multiprocessing as mp
import os
import sys
import zlib

import cv2
import numpy as np
import pandas as pd
import pyzipper
from PIL import Image

PASSWORD = b"STFD_ICASSP2023"
ROOT = "STFD_ICASSP2023"
CATEGORIES = ["1_Copy-move", "2_Splicing", "3_Removal", "4_Insertion", "5_Replacement"]
Image.MAX_IMAGE_PIXELS = None

_zf = None


def _init(zip_path):
    global _zf
    _zf = pyzipper.AESZipFile(zip_path)
    _zf.setpassword(PASSWORD)


def sniff(b):
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return "PNG"
    if b[:3] == b"\xff\xd8\xff":
        return "JPEG"
    return "OTHER"


def grid8_naive(gray):
    """[THẤT BẠI ĐỐI CHỨNG - giữ lại để tái tạo] max/median của |gradient| trung bình theo pha (mod 8).
    Không phân biệt được PNG với JPEG q75 (xem grid_control), nên KHÔNG dùng để kết luận."""
    out = []
    for axis in (1, 0):
        d = np.abs(np.diff(gray, axis=axis)).mean(axis=1 - axis)
        n = (len(d) // 8) * 8
        prof = d[:n].reshape(-1, 8).mean(axis=0)
        med = np.median(prof)
        out.append(float(prof.max() / med) if med > 0 else np.nan)
    return max(out)


def grid8_small(gray, thr=6):
    """Dấu vết lưới JPEG 8x8 chỉ xét cặp pixel kề nhau chênh lệch nhỏ (vùng phẳng).
    Với mỗi pha (mod 8) tính tỉ lệ bước nhỏ khác 0; trả max/median của 8 pha (max của hướng ngang/dọc).
    ~1.0 = không có lưới. Đã đối chứng: PNG ~1.07, JPEG q95 ~1.21, q85 ~1.41, q75 ~1.47 (median, 20 ảnh).
    Là chỉ báo gián tiếp, KHÔNG có ngưỡng đã hiệu chuẩn trên ảnh chắc chắn chưa từng qua JPEG."""
    out = []
    for axis in (1, 0):
        df = np.abs(np.diff(gray, axis=axis))
        cnt = ((df > 0) & (df <= thr)).sum(axis=1 - axis).astype(np.float64)
        tot = (df <= thr).sum(axis=1 - axis).astype(np.float64)
        n = (len(cnt) // 8) * 8
        frac = cnt[:n].reshape(-1, 8).sum(0) / np.maximum(tot[:n].reshape(-1, 8).sum(0), 1)
        med = np.median(frac)
        out.append(float(frac.max() / med) if med > 0 else np.nan)
    return max(out)


def analyse(args):
    cat, name = args
    base = f"{ROOT}/{cat}"
    tb = _zf.read(f"{base}/tamper/{name}")
    mb = _zf.read(f"{base}/masks/{name}")
    rec = {
        "image_id": os.path.splitext(name)[0],
        "path": f"{cat}/tamper/{name}",
        "mask_path": f"{cat}/masks/{name}",
        "tamper_type": cat.split("_", 1)[1],
        "file_bytes": len(tb),
        "md5_content": hashlib.md5(tb).hexdigest(),
        "mask_md5": hashlib.md5(mb).hexdigest(),
    }
    rec["name_is_md5_of_content"] = rec["md5_content"] == rec["image_id"]
    rec["img_format"] = sniff(tb)
    rec["mask_format"] = sniff(mb)

    im = Image.open(io.BytesIO(tb))
    rec["img_mode"] = im.mode
    rec["width"], rec["height"] = im.size
    rec["png_text_keys"] = ",".join(sorted(im.info.keys())) if im.format == "PNG" else ""
    if im.format == "JPEG":
        q = im.quantization
        rec["jpeg_qtables"] = json.dumps({str(k): list(v) for k, v in q.items()})
    else:
        rec["jpeg_qtables"] = ""
    rgb = np.asarray(im.convert("RGB"))
    gray = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    rec["brightness"] = float(gray.mean())
    rec["theme_est"] = "Light" if rec["brightness"] >= 128 else "Dark"
    # Chỉ số lưới dùng cùng định nghĩa xám với grid_control (PIL 'L', số nguyên) để so sánh được với đối chứng
    g8 = np.asarray(im.convert("L")).astype(np.float32)
    rec["grid8_naive"] = grid8_naive(g8)
    rec["grid8_small"] = grid8_small(g8)
    sub = rgb[::4, ::4].reshape(-1, 3)
    rec["n_colors_sub"] = int(len(np.unique(sub, axis=0)))

    mk = Image.open(io.BytesIO(mb))
    rec["mask_mode"] = mk.mode
    rec["mask_w"], rec["mask_h"] = mk.size
    m = np.asarray(mk.convert("L"))
    u = np.unique(m)
    rec["mask_values"] = ",".join(map(str, u[:8])) + ("..." if len(u) > 8 else "")
    rec["mask_n_unique"] = int(len(u))
    rec["mask_size_match"] = (rec["mask_w"], rec["mask_h"]) == (rec["width"], rec["height"])

    mb_bin = (m > 127).astype(np.uint8)
    area = int(mb_bin.sum())
    rec["mask_area_px"] = area
    rec["mask_area_ratio"] = area / mb_bin.size
    rec["is_clean"] = area == 0
    rec["mask_full"] = area == mb_bin.size
    if area > 0:
        ys, xs = np.where(mb_bin)
        bbox = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
        rec["mask_fill_ratio"] = area / bbox
        n, _, st, _ = cv2.connectedComponentsWithStats(mb_bin, connectivity=8)
        comp = st[1:]
        rec["n_components"] = int(n - 1)
        rec["comp_fill_weighted"] = float(comp[:, cv2.CC_STAT_AREA].sum() /
                                          (comp[:, cv2.CC_STAT_WIDTH] * comp[:, cv2.CC_STAT_HEIGHT]).sum())
        big = comp[comp[:, cv2.CC_STAT_AREA].argmax()]
        rec["largest_comp_fill"] = float(big[cv2.CC_STAT_AREA] / (big[cv2.CC_STAT_WIDTH] * big[cv2.CC_STAT_HEIGHT]))
    else:
        rec["mask_fill_ratio"] = np.nan
        rec["n_components"] = 0
        rec["comp_fill_weighted"] = np.nan
        rec["largest_comp_fill"] = np.nan
    return rec


def grid_control(zip_path, names, k=20, qualities=(95, 85, 75)):
    """Đối chứng: nén thêm JPEG vài ảnh rồi đo lại cả hai chỉ số -> kiểm tra chỉ số có phát hiện được lưới không."""
    zf = pyzipper.AESZipFile(zip_path)
    zf.setpassword(PASSWORD)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(names), size=min(k, len(names)), replace=False)
    vals = {m: {"png": []} for m in ("naive", "small")}
    for i in pick:
        cat, name = names[i]
        im = Image.open(io.BytesIO(zf.read(f"{ROOT}/{cat}/tamper/{name}"))).convert("RGB")
        g0 = np.asarray(im.convert("L")).astype(np.float32)
        vals["naive"]["png"].append(grid8_naive(g0))
        vals["small"]["png"].append(grid8_small(g0))
        for q in qualities:
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=q)
            g1 = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("L")).astype(np.float32)
            vals["naive"].setdefault(f"q{q}", []).append(grid8_naive(g1))
            vals["small"].setdefault(f"q{q}", []).append(grid8_small(g1))
    return {"n": len(pick),
            **{m: {c: {"median": float(np.median(v)), "min": float(np.min(v)), "max": float(np.max(v))}
                   for c, v in d.items()} for m, d in vals.items()}}


def hist_bins(ratio):
    r = np.asarray(ratio)
    return {
        "<0.1%": int((r < 0.001).sum()),
        "0.1-1%": int(((r >= 0.001) & (r < 0.01)).sum()),
        "1-5%": int(((r >= 0.01) & (r < 0.05)).sum()),
        ">5%": int((r >= 0.05).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/stfd/STFD_ICASSP2023.zip")
    ap.add_argument("--limit", type=int, default=0, help="chỉ chạy N ảnh đầu (debug)")
    ap.add_argument("--procs", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    a = ap.parse_args()

    zf = pyzipper.AESZipFile(a.zip)
    zf.setpassword(PASSWORD)
    names = zf.namelist()
    summary = {"zip_bytes": os.path.getsize(a.zip), "zip_entries": len(names)}

    # ---- cấu trúc & ghép cặp ----
    pairs, unpaired = [], []
    for cat in CATEGORIES:
        t = {os.path.basename(n) for n in names if n.startswith(f"{ROOT}/{cat}/tamper/") and not n.endswith("/")}
        m = {os.path.basename(n) for n in names if n.startswith(f"{ROOT}/{cat}/masks/") and not n.endswith("/")}
        unpaired += [(cat, "tamper_only", x) for x in t - m] + [(cat, "mask_only", x) for x in m - t]
        pairs += [(cat, x) for x in sorted(t & m)]
    summary["n_pairs"] = len(pairs)
    summary["unpaired"] = unpaired
    top = collections.Counter(n.split("/")[1] if n.count("/") >= 1 else n for n in names)
    summary["top_level_entries"] = dict(top)
    summary["other_files"] = [n for n in names if not n.endswith("/") and not n.lower().endswith(".png")]
    nested = [n for n in names if n.startswith(f"{ROOT}/{ROOT}/")]
    summary["nested_dup_dir_entries"] = nested
    for n in summary["other_files"]:
        try:
            summary.setdefault("other_file_text", {})[n] = zf.read(n).decode("utf-8", "replace")[:6000]
        except Exception as e:  # noqa
            summary.setdefault("other_file_text", {})[n] = f"ERR {e}"
    zf.close()

    if a.limit:
        pairs = pairs[: a.limit]
    print(f"{len(pairs)} cặp; procs={a.procs}", flush=True)

    with mp.Pool(a.procs, initializer=_init, initargs=(a.zip,)) as pool:
        recs = []
        for i, r in enumerate(pool.imap(analyse, pairs, chunksize=8), 1):
            recs.append(r)
            if i % 250 == 0:
                print(f"  {i}/{len(pairs)}", flush=True)
    df = pd.DataFrame(recs)

    # ---- CSV theo cột yêu cầu, cột phụ ở cuối ----
    req = ["image_id", "path", "is_clean", "img_format", "width", "height", "mask_area_px",
           "mask_area_ratio", "mask_fill_ratio", "theme_est", "tamper_type"]
    extra = [c for c in df.columns if c not in req]
    df[req + extra].to_csv("stfd_stats.csv", index=False)

    # ---- tổng hợp ----
    S = summary
    S["per_category"] = df.groupby("tamper_type").size().to_dict()
    S["img_format"] = df.img_format.value_counts().to_dict()
    S["mask_format"] = df.mask_format.value_counts().to_dict()
    S["img_mode"] = df.img_mode.value_counts().to_dict()
    S["png_text_keys"] = df.png_text_keys.value_counts().to_dict()
    S["jpeg_by_content"] = int((df.img_format == "JPEG").sum())
    if S["jpeg_by_content"]:
        q = df[df.img_format == "JPEG"].jpeg_qtables
        S["jpeg_qtable_clusters"] = q.value_counts().head(10).to_dict()

    # kích thước
    df["res"] = df.width.astype(str) + "x" + df.height.astype(str)
    S["n_resolutions"] = int(df.res.nunique())
    S["top_resolutions"] = df.res.value_counts().head(25).to_dict()
    S["width_minmax"] = [int(df.width.min()), int(df.width.max())]
    S["height_minmax"] = [int(df.height.min()), int(df.height.max())]
    S["orientation"] = {"portrait": int((df.height > df.width).sum()),
                        "landscape": int((df.width > df.height).sum()),
                        "square": int((df.width == df.height).sum())}

    # mask
    S["mask_mode"] = df.mask_mode.value_counts().to_dict()
    S["mask_values"] = df.mask_values.value_counts().to_dict()
    S["mask_size_match_all"] = bool(df.mask_size_match.all())
    S["mask_size_mismatch_n"] = int((~df.mask_size_match).sum())
    S["mask_empty_n"] = int(df.is_clean.sum())
    S["mask_full_n"] = int(df.mask_full.sum())
    r = df.mask_area_ratio
    S["area_ratio"] = {"min": float(r.min()), "max": float(r.max()), "median": float(r.median()),
                       "mean": float(r.mean()), "p5": float(r.quantile(.05)), "p95": float(r.quantile(.95))}
    S["area_hist"] = hist_bins(r)
    S["area_hist_by_cat"] = {c: hist_bins(g.mask_area_ratio) for c, g in df.groupby("tamper_type")}
    S["area_ratio_by_cat"] = {c: {"min": float(g.mask_area_ratio.min()), "median": float(g.mask_area_ratio.median()),
                                  "max": float(g.mask_area_ratio.max())} for c, g in df.groupby("tamper_type")}

    # fill ratio
    def fdist(s):
        s = s.dropna()
        q = s.quantile([.05, .25, .5, .75, .95])
        bins = {"<0.2": int((s < .2).sum()), "0.2-0.5": int(((s >= .2) & (s < .5)).sum()),
                "0.5-0.85": int(((s >= .5) & (s < .85)).sum()), "0.85-0.95": int(((s >= .85) & (s < .95)).sum()),
                ">=0.95": int((s >= .95).sum())}
        return {"n": int(len(s)), "min": float(s.min()), "p05": float(q[.05]), "p25": float(q[.25]),
                "median": float(q[.5]), "p75": float(q[.75]), "p95": float(q[.95]), "max": float(s.max()),
                "bins": bins}
    S["fill_aggregate_bbox"] = fdist(df.mask_fill_ratio)
    S["fill_weighted_components"] = fdist(df.comp_fill_weighted)
    S["fill_largest_component"] = fdist(df.largest_comp_fill)
    S["fill_aggregate_by_cat"] = {c: fdist(g.mask_fill_ratio) for c, g in df.groupby("tamper_type")}
    S["fill_weighted_by_cat"] = {c: fdist(g.comp_fill_weighted) for c, g in df.groupby("tamper_type")}
    S["n_components"] = {"median": float(df.n_components.median()), "max": int(df.n_components.max()),
                         "single_component": int((df.n_components == 1).sum())}

    # theme
    S["theme"] = df.theme_est.value_counts().to_dict()
    S["theme_by_cat"] = df.groupby(["tamper_type", "theme_est"]).size().unstack(fill_value=0).to_dict("index")
    S["brightness_ambiguous_100_156"] = int(df.brightness.between(100, 156).sum())
    S["brightness_quantiles"] = {str(q): float(df.brightness.quantile(q)) for q in (0, .05, .25, .5, .75, .95, 1)}

    # nén nhiều lần (gián tiếp)
    for col in ("grid8_naive", "grid8_small"):
        g = df[col].dropna()
        S[col] = {"min": float(g.min()), "p05": float(g.quantile(.05)), "median": float(g.median()),
                  "p95": float(g.quantile(.95)), "max": float(g.max()), "n": int(len(g)),
                  **{f"n_gt_{t}": int((g > t).sum()) for t in (1.1, 1.2, 1.4)}}
    S["grid8_small_by_cat_median"] = df.groupby("tamper_type").grid8_small.median().to_dict()
    S["n_colors_sub"] = {"min": int(df.n_colors_sub.min()), "median": float(df.n_colors_sub.median()),
                         "max": int(df.n_colors_sub.max())}
    S["grid_control"] = grid_control(a.zip, pairs)

    # bất thường
    S["name_is_md5_of_content"] = int(df.name_is_md5_of_content.sum())
    dup = df.groupby("md5_content").size()
    S["dup_image_bytes_groups"] = int((dup > 1).sum())
    S["dup_image_bytes_extra_copies"] = int((dup[dup > 1] - 1).sum())
    dupx = df[df.md5_content.isin(dup[dup > 1].index)].groupby("md5_content").tamper_type.apply(
        lambda s: sorted(set(s)))
    S["dup_across_categories"] = int((dupx.map(len) > 1).sum())
    mdup = df.groupby("mask_md5").size()
    S["dup_mask_groups"] = int((mdup > 1).sum())
    S["file_bytes"] = {"min": int(df.file_bytes.min()), "median": float(df.file_bytes.median()),
                       "max": int(df.file_bytes.max())}
    S["id_reused_across_categories"] = int((df.groupby("image_id").size() > 1).sum())

    S["nonbinary_mask_files"] = df[df.mask_values != "0,255"][["path", "mask_mode", "mask_values"]].to_dict("records")
    S["mask_i16_files"] = df[df.mask_mode == "I;16"].path.tolist()
    S["odd_width_files"] = df[df.width % 2 == 1][["path", "width", "height"]].to_dict("records")
    S["rgba_by_cat"] = df[df.img_mode == "RGBA"].groupby("tamper_type").size().to_dict()
    # heuristic của forensichub_dataset.py: solidity của thành phần lớn nhất > 0.85 => "BoxPatch"
    S["repo_boxpatch_heuristic"] = {
        "overall": int((df.largest_comp_fill > 0.85).sum()),
        "by_cat": df.groupby("tamper_type").apply(lambda g: float((g.largest_comp_fill > 0.85).mean()),
                                                  include_groups=False).to_dict()}

    os.makedirs("data", exist_ok=True)
    with open("data/stfd_summary.json", "w", encoding="utf-8") as f:
        json.dump(S, f, indent=2, ensure_ascii=False, default=str)
    print("done: stfd_stats.csv, data/stfd_summary.json")


if __name__ == "__main__":
    mp.freeze_support()
    main()
