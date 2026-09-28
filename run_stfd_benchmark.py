"""
run_stfd_benchmark.py
=====================
Chạy inference thật của TruFor / CAFTB-Net / ADCD-Net trên mẫu STFD và ghi CSV từng (ảnh, mô hình).

Giao thức (theo ke-hoach-do-an.md, đã thống nhất):
  * Lưới đánh giá chung: ảnh hạ về <= 2 MP bằng cv2.INTER_AREA (giữ tỉ lệ); mask hạ bằng INTER_AREA rồi ngưỡng 0.5
    (giữ được nét mảnh tốt hơn INTER_NEAREST). Mọi bản đồ dự đoán ở đúng lưới này.
  * TruFor: cả ảnh. CAFTB-Net, ADCD-Net: cửa sổ trượt 512x512 (bước --stride) ở độ phân giải của lưới đánh giá.
  * Thời gian: có torch.cuda.synchronize, đã khởi động nóng 1 ảnh (loại khỏi thống kê). Với ADCD-Net, infer_ms gồm cả
    OCR (ocr_ms ghi riêng).
  * Không đảo cực bản đồ. Không dùng nhãn sạch/bẩn nào ngoài mask.
  * Chạy lần lượt theo từng mô hình (nạp -> chạy hết -> giải phóng) để vừa 4 GB VRAM; ghi CSV tăng dần, chạy lại được.

Chạy:
  python run_stfd_benchmark.py --models trufor,caftb,adcd [--limit 20] [--out stfd_benchmark_results.csv]
"""
import argparse
import csv
import gc
import json
import os
import time
import traceback

import cv2
import numpy as np
import torch
from PIL import Image

from stfd_metrics import pixel_metrics

MAX_PIXELS = 2_000_000
HEAT_DIR = "data/heatmaps"
COLS = ["image_id", "dataset", "model", "theme", "tamper_type", "width", "height", "eval_w", "eval_h", "scale",
        "infer_ms", "ocr_ms", "peak_vram_mb", "is_clean", "P_05", "R_05", "F1_05", "IoU_05", "PR_AUC", "AP",
        "ROC_AUC", "oracle_F1", "tau_star", "gt_area_px", "gt_area_ratio", "gt_area_ratio_native", "pred_pixels_05",
        "pred_area_ratio_img", "area_ratio_05", "fp_rate_outside_05", "prob_mean", "prob_p99", "error"]


def load_pair(rec):
    img = np.asarray(Image.open(rec["img_path"]).convert("RGB"))
    m = (np.asarray(Image.open(rec["mask_path"]).convert("L")) > 127)
    return img, m


def to_eval_grid(img, mask, max_pixels=MAX_PIXELS):
    h, w = img.shape[:2]
    if h * w <= max_pixels:
        return img, mask, 1.0
    s = (max_pixels / (h * w)) ** 0.5
    nw, nh = int(round(w * s)), int(round(h * s))
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    mk = cv2.resize(mask.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA) >= 0.5
    return im, mk, s


def interleaved(records):
    """Xen kẽ theo loại thao tác để kết quả từng phần vẫn cân bằng nếu bị ngắt."""
    by = {}
    for r in records:
        by.setdefault(r["tamper_type"], []).append(r)
    out = []
    for i in range(max(len(v) for v in by.values())):
        for k in sorted(by):
            if i < len(by[k]):
                out.append(by[k][i])
    return out


def build(name, device, args):
    from stfd_models import ADCDNetDet, CAFTBNet, TruFor
    if name == "trufor":
        return TruFor(device, mode=args.trufor_mode)
    if name == "caftb":
        return CAFTBNet(device, "data/weights/caftb/caftb-9.pth", stride=args.stride)
    if name == "adcd":
        return ADCDNetDet(device, stride=args.stride)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="data/stfd_sample.json")
    ap.add_argument("--out", default="stfd_benchmark_results.csv")
    ap.add_argument("--models", default="trufor,caftb,adcd")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--trufor-mode", default="fp32", choices=["fp32", "fp16", "tile"])
    ap.add_argument("--max-pixels", type=int, default=MAX_PIXELS)
    a = ap.parse_args()

    if os.name == "nt":
        # Không cho hệ thống tự ngủ trong lúc chạy (chỉ có hiệu lực trong tiến trình này; tự hết khi thoát).
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recs = interleaved(json.load(open(a.sample, encoding="utf-8")))
    if a.limit:
        recs = recs[: a.limit]

    done = set()
    if os.path.exists(a.out):
        with open(a.out, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if not r.get("error"):
                    done.add((r["image_id"], r["model"]))
    new_file = not os.path.exists(a.out)
    fh = open(a.out, "a", newline="", encoding="utf-8")
    wr = csv.DictWriter(fh, fieldnames=COLS)
    if new_file:
        wr.writeheader()

    for mname in a.models.split(","):
        det = None
        todo = [r for r in recs if (r["img_id"], {"trufor": "TruFor", "caftb": "CAFTB-Net", "adcd": "ADCD-Net"}[mname]) not in done]
        print(f"[{mname}] {len(todo)}/{len(recs)} ảnh cần chạy", flush=True)
        if not todo:
            continue
        det = build(mname, device, a)
        os.makedirs(f"{HEAT_DIR}/{det.name}", exist_ok=True)
        # khởi động nóng (loại khỏi thống kê)
        img0, m0 = load_pair(todo[0])
        im0, _, _ = to_eval_grid(img0, m0, a.max_pixels)
        det.predict(im0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.time()
        for i, rec in enumerate(todo):
            row = {c: "" for c in COLS}
            row.update(image_id=rec["img_id"], dataset=rec["dataset"], model=det.name, theme=rec["theme"],
                       tamper_type=rec["tamper_type"], width=rec["width"], height=rec["height"])
            try:
                img, m = load_pair(rec)
                native_ratio = float(m.mean())
                im, mk, s = to_eval_grid(img, m, a.max_pixels)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                prob = det.predict(im)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) * 1000
                peak = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else 0.0
                assert prob.shape == mk.shape, (prob.shape, mk.shape)
                met = pixel_metrics(prob, mk)
                row.update(met)
                row.update(eval_w=im.shape[1], eval_h=im.shape[0], scale=s, infer_ms=dt,
                           ocr_ms=getattr(det, "ocr_ms", 0.0) if det.name == "ADCD-Net" else 0.0,
                           peak_vram_mb=peak, gt_area_ratio_native=native_ratio)
                # heatmap thu nhỏ (uint8) để xem lại/vẽ hình
                k = 384 / max(prob.shape)
                hm = cv2.resize((np.clip(prob, 0, 1) * 255).astype(np.uint8), None, fx=k, fy=k,
                                interpolation=cv2.INTER_AREA)
                cv2.imwrite(f"{HEAT_DIR}/{det.name}/{rec['img_id']}.png", hm)
            except Exception as e:                                   # ghi lỗi từng ảnh, không dừng cả lượt
                row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                traceback.print_exc()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            wr.writerow(row)
            fh.flush()
            if (i + 1) % 10 == 0 or i == len(todo) - 1:
                el = time.time() - t_start
                print(f"  [{det.name}] {i + 1}/{len(todo)}  {el / (i + 1):.1f}s/ảnh  còn ~{el / (i + 1) * (len(todo) - i - 1) / 60:.0f} phút",
                      flush=True)
        del det
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    fh.close()
    print("xong ->", a.out)


if __name__ == "__main__":
    main()
