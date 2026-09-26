"""
forensichub_dataset.py
======================
Quét thư mục DocTamper đã giải nén (LMDB → jpg/png), tự động phân loại
metadata cho từng ảnh (theme, tamper_type, is_clean, gt_area_ratio) rồi
xuất JSON index theo chuẩn ForensicHub / IMDLBenCo.

Metadata tự suy luận:
  - theme        : Light (brightness >= 128) | Dark (< 128)
  - tamper_type   : Authentic (mask trắng) | BoxPatch (solidity > 0.85) |
                    TightContour (solidity <= 0.85)
  - gt_area_ratio : diện tích mask / tổng pixel (trước resize)
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path


# ------------------------------------------------------------------
# Hàm phân loại metadata cho 1 ảnh
# ------------------------------------------------------------------
def _classify_image(img_path: Path, mask_path: Path):
    """Trả về dict chứa theme, tamper_type, gt_area_ratio, is_clean."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    theme = "Light" if gray.mean() >= 128 else "Dark"

    # Đọc mask
    if mask_path.exists():
        msk = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    else:
        msk = None

    if msk is None:
        return {
            "theme": theme,
            "tamper_type": "Authentic",
            "gt_area_ratio": 0.0,
            "is_clean": True,
        }

    msk_bin = (msk > 127).astype(np.uint8)
    total_pixels = msk_bin.shape[0] * msk_bin.shape[1]
    tampered_pixels = int(msk_bin.sum())

    if tampered_pixels == 0:
        return {
            "theme": theme,
            "tamper_type": "Authentic",
            "gt_area_ratio": 0.0,
            "is_clean": True,
        }

    gt_area_ratio = tampered_pixels / total_pixels

    # Phân biệt kiểu can thiệp qua hình dạng contour
    contours, _ = cv2.findContours(msk_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cnt_area = cv2.contourArea(largest)
        x, y, w, h = cv2.boundingRect(largest)
        bbox_area = w * h
        solidity = cnt_area / bbox_area if bbox_area > 0 else 0
        tamper_type = "BoxPatch" if solidity > 0.85 else "TightContour"
    else:
        tamper_type = "BoxPatch"  # fallback

    return {
        "theme": theme,
        "tamper_type": tamper_type,
        "gt_area_ratio": gt_area_ratio,
        "is_clean": False,
    }


# ------------------------------------------------------------------
# Quét toàn bộ thư mục và xuất JSON
# ------------------------------------------------------------------
def prepare_forensichub_json(extracted_dir, output_json="doctamper_forensichub.json"):
    """
    Duyệt *_img.jpg, ghép mask, tự suy metadata, ghi JSON.
    Trả về đường dẫn output.
    """
    extracted_path = Path(extracted_dir)
    if not extracted_path.exists():
        print(f"ERROR: Thu muc khong ton tai: {extracted_dir}")
        return None

    print(f"Dang quet {extracted_dir} ...")
    img_files = sorted(extracted_path.glob("*_img.jpg"))[:100]
    total = len(img_files)
    records = []

    for i, img_path in enumerate(img_files):
        idx_str = img_path.name.split("_")[0]
        mask_path = img_path.with_name(f"{idx_str}_mask.png")

        meta = _classify_image(img_path, mask_path)

        record = {
            "image_path": str(img_path).replace("\\", "/"),
            "mask_path": str(mask_path).replace("\\", "/") if mask_path.exists() else None,
            "label": 0 if meta["is_clean"] else 1,
            "img_id": idx_str,
            "theme": meta["theme"],
            "tamper_type": meta["tamper_type"],
            "gt_area_ratio": meta["gt_area_ratio"],
            "is_clean": meta["is_clean"],
        }
        records.append(record)

        if (i + 1) % 5000 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] done")

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # Thống kê nhanh
    n_clean = sum(1 for r in records if r["is_clean"])
    n_tampered = total - n_clean
    themes = {}
    ttypes = {}
    for r in records:
        themes[r["theme"]] = themes.get(r["theme"], 0) + 1
        ttypes[r["tamper_type"]] = ttypes.get(r["tamper_type"], 0) + 1

    print(f"\nDa luu {total} mau vao {output_json}")
    print(f"  Clean: {n_clean}  |  Tampered: {n_tampered}")
    print(f"  Theme:       {themes}")
    print(f"  Tamper type: {ttypes}")
    return output_json


# ------------------------------------------------------------------
# Load JSON cho benchmark.py
# ------------------------------------------------------------------
def load_dataset_for_benchmark(json_path):
    """
    Doc JSON, tra ve list[dict] voi cac truong can thiet cho benchmark.py:
      img_id, dataset, img_path, mask_path, theme, tamper_type
    """
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    return [
        {
            "img_id": rec["img_id"],
            "dataset": "DocTamper",
            "img_path": rec["image_path"],
            "mask_path": rec["mask_path"],
            "theme": rec["theme"],
            "tamper_type": rec["tamper_type"],
        }
        for rec in records
    ]


# ------------------------------------------------------------------
if __name__ == "__main__":
    data_dir = "./data/doctamper/DocTamperV1-TestingSet_Extracted"
    json_file = "./data/doctamper_forensichub.json"

    prepare_forensichub_json(data_dir, json_file)

    data_list = load_dataset_for_benchmark(json_file)
    print(f"\nSan sang {len(data_list)} mau cho benchmark.")
