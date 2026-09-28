"""
stfd_metrics.py
===============
Metric pixel cho một cặp (bản đồ dự đoán, mask) trên lưới đánh giá chung.

  - Chỉ số ở tau = 0.5 (P, R, F1, IoU, pred_pixels, fp_rate_outside): tính chính xác trên số thực.
  - ROC-AUC, PR-AUC (tích phân hình thang, cùng quy ước sklearn.metrics.auc(recall, precision) như benchmark.py),
    AP: tính từ histogram 65 536 mức của xác suất (sai số lượng tử hóa <= 1.6e-5 theo xác suất). Có selftest so
    với sklearn.
  - Oracle-F1 và tau*: quét 50 ngưỡng cách đều np.linspace(0, 1, 50) (theo ke-hoach-do-an.md), tính chính xác.
    CHỈ LÀ CHẨN ĐOÁN/CẬN TRÊN (dùng ground truth để chọn ngưỡng), không phải kết quả.
  - Ảnh sạch (mask rỗng): các chỉ số cần dương tính là NaN; vẫn ghi pred_pixels_05.
"""
import numpy as np

N_BINS = 65536
ORACLE_TAUS = np.linspace(0.0, 1.0, 50)


def _curves_from_hist(pos_h, neg_h):
    """Từ histogram dương/âm (chỉ số bin tăng = xác suất tăng), trả ROC-AUC, PR-AUC (hình thang), AP."""
    P, Nn = pos_h.sum(), neg_h.sum()
    tp = np.cumsum(pos_h[::-1]).astype(np.float64)          # ngưỡng giảm dần: dự đoán dương nếu bin >= b
    fp = np.cumsum(neg_h[::-1]).astype(np.float64)
    tpr = np.concatenate([[0.0], tp / P])
    fpr = np.concatenate([[0.0], fp / Nn]) if Nn > 0 else np.zeros(len(tpr))
    roc = float(np.trapezoid(tpr, fpr)) if Nn > 0 else np.nan
    denom = tp + fp
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(denom > 0, tp / denom, 1.0)
    rec = tp / P
    # bỏ điểm trùng recall liên tiếp đầu chuỗi để khớp cách sklearn cắt các ngưỡng vô nghĩa
    rec_full = np.concatenate([[0.0], rec])
    prec_full = np.concatenate([[1.0], prec])
    pr_auc = float(np.trapezoid(prec_full, rec_full))
    ap = float(np.sum(np.diff(rec_full) * prec_full[1:]))
    return roc, pr_auc, ap


def pixel_metrics(prob, gt):
    """
    prob: float ndarray [H, W] trong [0, 1]; gt: bool ndarray [H, W].
    Trả dict các cột (khoá cùng quy ước với CSV cũ: P_05, R_05, F1_05, IoU_05, PR_AUC, ...).
    """
    prob = np.asarray(prob, np.float32)
    gt = np.asarray(gt, bool)
    n = gt.size
    g = int(gt.sum())
    pred = prob >= 0.5
    pp = int(pred.sum())
    out = {"gt_area_px": g, "gt_area_ratio": g / n, "pred_pixels_05": pp, "pred_area_ratio_img": pp / n,
           "prob_mean": float(prob.mean()), "prob_p99": float(np.quantile(prob[:: max(1, n // 200000)], 0.99)),
           "is_clean": g == 0}
    nan = np.nan
    if g == 0:
        out.update(dict(P_05=nan, R_05=nan, F1_05=nan, IoU_05=nan, PR_AUC=nan, AP=nan, ROC_AUC=nan,
                        oracle_F1=nan, tau_star=nan, area_ratio_05=nan, fp_rate_outside_05=pp / n))
        return out
    tp = int((pred & gt).sum())
    fp, fn = pp - tp, g - tp
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    iou = tp / (tp + fp + fn)
    out.update(P_05=p, R_05=r, F1_05=f1, IoU_05=iou, area_ratio_05=pp / g,
               fp_rate_outside_05=(fp / (n - g)) if n > g else nan)
    # đường cong từ histogram
    q = np.minimum((prob * (N_BINS - 1) + 0.5).astype(np.int64), N_BINS - 1)
    pos_h = np.bincount(q[gt], minlength=N_BINS)
    neg_h = np.bincount(q[~gt], minlength=N_BINS)
    roc, pr_auc, ap = _curves_from_hist(pos_h, neg_h)
    out.update(PR_AUC=pr_auc, AP=ap, ROC_AUC=roc)
    # Oracle-F1 trên 50 ngưỡng (chẩn đoán)
    best, best_t = -1.0, nan
    for t in ORACLE_TAUS:
        pr_t = prob >= t
        pp_t = int(pr_t.sum())
        tp_t = int((pr_t & gt).sum())
        f1_t = 2 * tp_t / (pp_t + g) if (pp_t + g) else 0.0       # F1 = 2TP/(2TP+FP+FN) = 2TP/(pred+gt)
        if f1_t > best:
            best, best_t = f1_t, float(t)
    out.update(oracle_F1=best, tau_star=best_t)
    return out


def selftest(seed=0):
    """So với sklearn trên dữ liệu tổng hợp có cấu trúc + trường hợp suy biến. Trả về sai lệch lớn nhất."""
    from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
    rng = np.random.default_rng(seed)
    worst = {"roc": 0.0, "pr": 0.0, "ap": 0.0, "f1": 0.0, "iou": 0.0}
    for k in range(6):
        h, w = 120, 160
        gt = np.zeros((h, w), bool)
        y0, x0 = rng.integers(10, 60), rng.integers(10, 80)
        gt[y0:y0 + rng.integers(5, 40), x0:x0 + rng.integers(5, 60)] = True
        sig = 0.15 * k                                                # k=0 -> nhiễu thuần; tăng dần tín hiệu
        prob = np.clip(rng.normal(0.3, 0.2, (h, w)) + sig * gt, 0, 1).astype(np.float32)
        m = pixel_metrics(prob, gt)
        y, s = gt.ravel().astype(int), prob.ravel()
        pr, rc, _ = precision_recall_curve(y, s)
        worst["roc"] = max(worst["roc"], abs(m["ROC_AUC"] - roc_auc_score(y, s)))
        worst["pr"] = max(worst["pr"], abs(m["PR_AUC"] - auc(rc, pr)))
        worst["ap"] = max(worst["ap"], abs(m["AP"] - average_precision_score(y, s)))
        pb = (s >= 0.5)
        tp = int((pb & (y == 1)).sum())
        f1 = 2 * tp / (pb.sum() + y.sum())
        iou = tp / (pb.sum() + y.sum() - tp)
        worst["f1"] = max(worst["f1"], abs(m["F1_05"] - f1))
        worst["iou"] = max(worst["iou"], abs(m["IoU_05"] - iou))
        # Oracle: cận trên của F1@0.5 khi 0.5 nằm trong lưới ngưỡng? (không bắt buộc) -> chỉ kiểm oracle >= F1 tại ngưỡng gần 0.5
        near = ORACLE_TAUS[np.argmin(np.abs(ORACLE_TAUS - 0.5))]
        pt = s >= near
        tpn = int((pt & (y == 1)).sum())
        assert m["oracle_F1"] + 1e-12 >= 2 * tpn / (pt.sum() + y.sum())
    # mask rỗng -> NaN
    e = pixel_metrics(rng.random((20, 20)).astype(np.float32), np.zeros((20, 20), bool))
    assert e["is_clean"] and np.isnan(e["F1_05"]) and e["pred_pixels_05"] > 0
    # dự đoán hoàn hảo
    gt = np.zeros((50, 50), bool); gt[10:20, 10:30] = True
    m = pixel_metrics(gt.astype(np.float32), gt)
    assert abs(m["F1_05"] - 1) < 1e-12 and abs(m["ROC_AUC"] - 1) < 1e-6 and abs(m["PR_AUC"] - 1) < 1e-6
    return worst


if __name__ == "__main__":
    w = selftest()
    print("sai lệch lớn nhất so với sklearn:", {k: f"{v:.2e}" for k, v in w.items()})
