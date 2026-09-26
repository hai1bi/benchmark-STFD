"""
benchmark.py  --  STFD Zero-Shot Benchmark
===========================================
Đánh giá ba mô hình phát hiện giả mạo tài liệu theo năm nhóm tiêu chí:

  Nhóm 1 — Hiệu năng cơ bản: P/R/F1/IoU@tau=0.5, PR-AUC, FPR mức ảnh
  Nhóm 2 — Chế độ hỏng:       area_ratio, pred_pixels (ảnh sạch),
                               Oracle-F1, tau*, calibration_gap
  Nhóm 3 — Phân tầng:         theo gt_area_ratio, tamper_type, theme
  Nhóm 4 — Đồng thuận:        tương quan thất bại giữa mô hình
  Nhóm 5 — Vận hành:          infer_time_sec, peak_vram_mb

Lưu ý công bằng:
  1. Cố định INPUT_SIZE cho cả ba mô hình.
  2. Ghi kết quả từng ảnh ra CSV; mọi phân tích nhóm 3-4 chạy từ CSV.
  3. Đây là zero-shot: không mô hình nào huấn luyện trên STFD.
"""

import os
import time
import torch
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import precision_recall_curve, auc

# ============================================================
# 0. CẤU HÌNH & WRAPPER MÔ HÌNH
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = (512, 512)  # Cố định đầu vào — Lưu ý 1

# TODO: Thay bằng code gọi model thực tế
def run_trufor(img_tensor):
    return np.random.rand(512, 512)

def run_caftb_net(img_tensor):
    return np.random.rand(512, 512)

def run_adcd_net(img_tensor):
    return np.random.rand(512, 512)

MODELS = {
    "TruFor":     run_trufor,
    "CAFTB-Net":  run_caftb_net,
    "ADCD-Net":   run_adcd_net,
}


# ============================================================
# 1. TÍNH METRIC CHO MỘT CẶP (pred, gt)  — Nhóm 1 & 2
# ============================================================
def calculate_metrics(pred_prob, gt_mask):
    """
    Parameters
    ----------
    pred_prob : ndarray [H, W], giá trị liên tục [0, 1]
    gt_mask   : ndarray [H, W], nhị phân {0, 1}

    Returns
    -------
    dict  với đầy đủ cột cho CSV (Nhóm 1 + 2 + 5-phần metric).
    """
    H, W = gt_mask.shape
    total_pixels = H * W
    pred_flat = pred_prob.flatten()
    gt_flat   = gt_mask.flatten()

    gt_area     = int(gt_flat.sum())
    is_clean    = (gt_area == 0)

    # --- Ảnh sạch (Authentic) -------------------------------------------
    if is_clean:
        pred_bin = (pred_flat >= 0.5).astype(np.uint8)
        fp_pixels = int(pred_bin.sum())
        return {
            "is_clean":         True,
            # Nhóm 1 — không áp dụng
            "P_05": np.nan, "R_05": np.nan, "F1_05": np.nan, "IoU_05": np.nan,
            "PR_AUC": np.nan,
            "FP_image":         int(fp_pixels > 0),      # Nhóm 1: FPR mức ảnh
            # Nhóm 2 — chế độ hỏng trên ảnh sạch
            "area_ratio_05":    np.nan,                   # mẫu số = 0 → báo pixel
            "pred_pixels_05":   fp_pixels,                # số pixel gắn cờ nhầm
            "gt_area_ratio":    0.0,
            # Nhóm 2 — Oracle
            "tau_star": np.nan, "oracle_F1": np.nan, "calibration_gap": np.nan,
        }

    # --- Ảnh bị can thiệp (Tampered) ------------------------------------
    gt_area_ratio = gt_area / total_pixels

    # PR-AUC  (Nhóm 1)
    precisions, recalls, thresholds = precision_recall_curve(gt_flat, pred_flat)
    pr_auc = auc(recalls, precisions)

    # Oracle F1 & tau*  (Nhóm 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_arr = np.where(
            (precisions + recalls) > 0,
            2 * precisions * recalls / (precisions + recalls),
            0.0,
        )
    best_idx  = int(np.argmax(f1_arr))
    oracle_f1 = float(f1_arr[best_idx])
    tau_star  = float(thresholds[best_idx]) if best_idx < len(thresholds) else 1.0

    # Metrics @ tau = 0.5  (Nhóm 1)
    pred_bin_05 = (pred_flat >= 0.5).astype(np.uint8)
    tp = int(np.logical_and(pred_bin_05 == 1, gt_flat == 1).sum())
    fp = int(np.logical_and(pred_bin_05 == 1, gt_flat == 0).sum())
    fn = int(np.logical_and(pred_bin_05 == 0, gt_flat == 1).sum())

    p_05   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r_05   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_05  = 2 * p_05 * r_05 / (p_05 + r_05) if (p_05 + r_05) > 0 else 0.0
    iou_05 = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    # Area ratio  (Nhóm 2)
    pred_area_05   = int(pred_bin_05.sum())
    area_ratio_05  = pred_area_05 / gt_area   # 0=im lặng, ~1=vừa, >>1=bão hoà

    calibration_gap = oracle_f1 - f1_05

    return {
        "is_clean":         False,
        # Nhóm 1
        "P_05": p_05, "R_05": r_05, "F1_05": f1_05, "IoU_05": iou_05,
        "PR_AUC":           pr_auc,
        "FP_image":         np.nan,           # chỉ dùng cho ảnh sạch
        # Nhóm 2
        "area_ratio_05":    area_ratio_05,
        "pred_pixels_05":   pred_area_05,
        "gt_area_ratio":    gt_area_ratio,
        "tau_star":         tau_star,
        "oracle_F1":        oracle_f1,
        "calibration_gap":  calibration_gap,
    }


# ============================================================
# 2. VÒNG LẶP ĐÁNH GIÁ  — tạo CSV (Nhóm 5 + ghi per-image)
# ============================================================
def evaluate_dataset(dataset_list, output_csv="benchmark_results.csv"):
    """
    dataset_list : list[dict] với các khoá bắt buộc:
        img_id, img_path, mask_path, dataset, theme, tamper_type
    output_csv   : đường dẫn file kết quả từng ảnh × mô hình.
    """
    results = []
    total = len(dataset_list)

    print(f"[Zero-shot] Bat dau danh gia {total} mau, "
          f"{len(MODELS)} mo hinh, INPUT_SIZE={INPUT_SIZE}")

    for idx, item in enumerate(dataset_list):
        # Đọc ảnh --------------------------------------------------
        img = cv2.imread(item["img_path"])
        if img is None:
            print(f"  WARN: khong doc duoc {item['img_path']}, bo qua.")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, INPUT_SIZE)

        # Đọc mask -------------------------------------------------
        if item.get("mask_path") and os.path.exists(item["mask_path"]):
            mask = cv2.imread(item["mask_path"], cv2.IMREAD_GRAYSCALE)
            mask_resized = cv2.resize(mask, INPUT_SIZE,
                                      interpolation=cv2.INTER_NEAREST)
            mask_resized = (mask_resized > 127).astype(np.uint8)
        else:
            mask_resized = np.zeros(INPUT_SIZE, dtype=np.uint8)

        # Tensor cho model ------------------------------------------
        input_tensor = (
            torch.from_numpy(img_resized)
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            .to(DEVICE) / 255.0
        )

        for model_name, model_func in MODELS.items():
            # Nhóm 5: đo thời gian & VRAM
            if DEVICE.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            start = time.time()

            with torch.no_grad():
                pred_prob = model_func(input_tensor)

            infer_time = time.time() - start
            peak_vram = (torch.cuda.max_memory_allocated() / (1024**2)
                         if DEVICE.type == "cuda" else 0.0)

            # Nhóm 1+2: tính metric
            metrics = calculate_metrics(pred_prob, mask_resized)

            # Đóng gói dòng CSV (Lưu ý 2: ghi theo từng ảnh)
            row = {
                "image_id":       item["img_id"],
                "dataset":        item.get("dataset", "STFD"),
                "model":          model_name,
                "theme":          item.get("theme", "Unknown"),
                "tamper_type":    item.get("tamper_type", "Unknown"),
                "infer_time_sec": infer_time,
                "peak_vram_mb":   peak_vram,
            }
            row.update(metrics)
            results.append(row)

        if (idx + 1) % 500 == 0 or (idx + 1) == total:
            print(f"  Da xu ly {idx+1}/{total} ...")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nDa luu {len(df)} dong ket qua vao: {output_csv}")
    return df


# ============================================================
# 3. PHÂN TÍCH TỪ CSV  — Nhóm 1-2-3-4-5
# ============================================================
def analyze_results(csv_path):
    """Đọc CSV kết quả per-image, in báo cáo đầy đủ 5 nhóm."""
    df = pd.read_csv(csv_path)
    models = df["model"].unique()

    SEP = "=" * 60

    # ------ NHÓM 1: Hiệu năng cơ bản ---------------------------
    print(f"\n{SEP}")
    print("NHOM 1 — HIEU NANG CO BAN")
    print(SEP)

    df_tam = df[df["is_clean"] == False].copy()
    if len(df_tam) > 0:
        grp1 = df_tam.groupby("model")[
            ["P_05", "R_05", "F1_05", "IoU_05", "PR_AUC"]
        ].mean()
        print("\n>>> P / R / F1 / IoU @ tau=0.5  va  PR-AUC  (chi anh tampered):")
        print(grp1.to_string(float_format="%.4f"))

    df_clean = df[df["is_clean"] == True].copy()
    if len(df_clean) > 0:
        fpr_tbl = df_clean.groupby("model")["FP_image"].agg(["sum", "count"])
        fpr_tbl.columns = ["false_positive_images", "total_clean"]
        fpr_tbl["FPR_image"] = fpr_tbl["false_positive_images"] / fpr_tbl["total_clean"]
        print("\n>>> FPR muc anh (tren anh sach):")
        print(fpr_tbl.to_string(float_format="%.4f"))
    else:
        print("\n>>> CANH BAO: Khong co anh sach trong dataset. "
              "FPR muc anh KHONG tinh duoc.")

    # ------ NHÓM 2: Chế độ hỏng ---------------------------------
    print(f"\n{SEP}")
    print("NHOM 2 — CHE DO HONG (FAILURE MODE)")
    print(SEP)

    if len(df_tam) > 0:
        grp2 = df_tam.groupby("model")[
            ["area_ratio_05", "tau_star", "oracle_F1", "calibration_gap"]
        ].mean()
        print("\n>>> area_ratio trung binh, tau*, Oracle-F1, calibration_gap:")
        print(grp2.to_string(float_format="%.4f"))

        print("\n  Giai thich area_ratio:")
        print("    ~0   = im lang, bo sot          (kieu TruFor)")
        print("    ~1   = khoanh vua")
        print("    >>1  = bao hoa, khoanh qua rong (kieu CAFTB-Net)")
        print("\n  Giai thich tau*:")
        print("    tau* thap (vd 0.15) = mo hinh xuat gia tri thap, can ha nguong")
        print("    tau* cao  (vd 0.90) = mo hinh xuat gia tri qua cao, can nang nguong")

    if len(df_clean) > 0:
        print("\n>>> Tren anh sach — so pixel bi gan co nham @ tau=0.5:")
        clean_pred = df_clean.groupby("model")["pred_pixels_05"].agg(["mean", "median", "max"])
        print(clean_pred.to_string(float_format="%.0f"))

    # ------ NHÓM 3: Phân tầng -----------------------------------
    print(f"\n{SEP}")
    print("NHOM 3 — PHAN TANG (STRATIFICATION)")
    print(SEP)

    if len(df_tam) > 0:
        # 3.1 Theo gt_area_ratio
        bins   = [0, 0.001, 0.01, 0.05, 1.0]
        labels = ["<0.1%", "0.1-1%", "1-5%", ">5%"]
        df_tam["area_bin"] = pd.cut(df_tam["gt_area_ratio"],
                                    bins=bins, labels=labels)

        print("\n>>> 3.1  F1@0.5  theo kich thuoc vung gia mao:")
        s31 = df_tam.groupby(["model", "area_bin"], observed=True)["F1_05"].mean().unstack()
        print(s31.to_string(float_format="%.4f"))

        print("\n>>> 3.1b area_ratio theo kich thuoc vung gia mao:")
        s31b = df_tam.groupby(["model", "area_bin"], observed=True)["area_ratio_05"].mean().unstack()
        print(s31b.to_string(float_format="%.2f"))

        # 3.2 Theo tamper_type
        if df_tam["tamper_type"].nunique() > 1:
            print("\n>>> 3.2  PR-AUC  theo loai thao tac:")
            s32 = df_tam.groupby(["model", "tamper_type"])["PR_AUC"].mean().unstack()
            print(s32.to_string(float_format="%.4f"))
        else:
            print(f"\n>>> 3.2  Chi co 1 tamper_type: '{df_tam['tamper_type'].iloc[0]}' "
                  "— khong can phan tang.")

        # 3.3 Theo theme (Light / Dark)
        if df_tam["theme"].nunique() > 1:
            print("\n>>> 3.3  FPR muc anh (neu co) va F1@0.5 theo nen Light/Dark:")
            s33 = df_tam.groupby(["model", "theme"])[["F1_05", "PR_AUC"]].mean().unstack()
            print(s33.to_string(float_format="%.4f"))

            if len(df_clean) > 0 and df_clean["theme"].nunique() > 1:
                print("\n     FPR muc anh theo theme:")
                s33b = df_clean.groupby(["model", "theme"])["FP_image"].mean().unstack()
                print(s33b.to_string(float_format="%.4f"))
        else:
            print("\n>>> 3.3  Chi co 1 theme — khong phan tang.")

    # ------ NHÓM 4: Đồng thuận ----------------------------------
    print(f"\n{SEP}")
    print("NHOM 4 — DONG THUAN (CONSENSUS)")
    print(SEP)

    if len(df_tam) > 0:
        df_tam["is_success"] = (df_tam["IoU_05"] > 0.3).astype(int)
        pivot = df_tam.pivot_table(
            index="image_id", columns="model",
            values="is_success", aggfunc="first"
        ).dropna()

        n = len(pivot)
        if n > 0:
            all_ok   = int(pivot.all(axis=1).sum())
            all_fail = int((pivot.sum(axis=1) == 0).sum())
            partial  = n - all_ok - all_fail

            print(f"\nTong anh tampered: {n}")
            print(f"  Ca {len(models)} mo hinh CUNG phat hien (IoU>0.3): "
                  f"{all_ok}  ({all_ok/n*100:.1f}%)")
            print(f"  Ca {len(models)} mo hinh CUNG that bai:            "
                  f"{all_fail}  ({all_fail/n*100:.1f}%)")
            print(f"  Mot so thanh cong, mot so that bai:     "
                  f"{partial}  ({partial/n*100:.1f}%)")

            print(f"\n  → Ca ba that bai NHIEU = tinh chat cua mien du lieu")
            print(f"  → That bai roi rac     = yeu diem rieng kien truc")

            print("\nMa tran tuong quan thanh cong (Pearson):")
            print(pivot.corr().to_string(float_format="%.4f"))

    # ------ NHÓM 5: Vận hành ------------------------------------
    print(f"\n{SEP}")
    print("NHOM 5 — VAN HANH")
    print(SEP)

    grp5 = df.groupby("model")[["infer_time_sec", "peak_vram_mb"]].agg(
        ["mean", "std", "max"]
    )
    print(f"\nINPUT_SIZE co dinh: {INPUT_SIZE}")
    print(">>> Thoi gian suy luan (sec) va VRAM dinh (MB):")
    print(grp5.to_string(float_format="%.4f"))

    print(f"\n{SEP}")
    print("LUU Y: Day la danh gia ZERO-SHOT.")
    print("Khong mo hinh nao duoc huan luyen tren STFD.")
    print("Bang nay do kha nang TONG QUAT HOA, khong do chat luong tuyet doi.")
    print(SEP)


# ============================================================
# CHẠY THỬ (DUMMY)
# ============================================================
if __name__ == "__main__":
    if not os.path.exists("dummy.jpg"):
        cv2.imwrite("dummy.jpg",
                     np.zeros((512, 512, 3), dtype=np.uint8))
        cv2.imwrite("dummy_mask.jpg",
                     np.zeros((512, 512), dtype=np.uint8))

    dummy_dataset = [
        {"img_id": "doc_01", "img_path": "dummy.jpg",
         "mask_path": "dummy_mask.jpg",
         "theme": "Light", "tamper_type": "BoxPatch"},
        {"img_id": "doc_02", "img_path": "dummy.jpg",
         "mask_path": "dummy_mask.jpg",
         "theme": "Dark", "tamper_type": "TightContour"},
        {"img_id": "clean_01", "img_path": "dummy.jpg",
         "mask_path": None,
         "theme": "Light", "tamper_type": "Authentic"},
    ]
    # df = evaluate_dataset(dummy_dataset, "stfd_benchmark_raw.csv")
    # analyze_results("stfd_benchmark_raw.csv")