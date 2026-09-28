"""
analyze_benchmark.py
====================
Phân tích kết quả benchmark từng ảnh (mục A-G) và kiểm tra tính hợp lệ của chính file kết quả.

Chạy:  python analyze_benchmark.py [--csv forensichub_benchmark_results.csv] [--out-dir figures]

Đầu ra: benchmark_tables.json (mọi con số), figures/*.png. benchmark_analysis.md viết tay từ JSON này.

Thích ứng schema (schema thực tế trong CSV khác schema mô tả trong yêu cầu):
  iou_at_05 <- IoU_05        f1_at_05 <- F1_05         pr_auc <- PR_AUC       oracle_f1 <- oracle_F1
  pred_area_px <- pred_pixels_05
  gt_area_px   <- round(gt_area_ratio * 512*512)  (mask đã resize về INPUT_SIZE=512x512 khi tính metric)
  roc_auc      : KHÔNG CÓ trong CSV (không lưu bản đồ dự đoán) -> không tính được
  inference_ms <- infer_time_sec*1000
"""
import argparse
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
N_PIX = 512 * 512
MODELS = ["TruFor", "CAFTB-Net", "ADCD-Net"]
COL = {"TruFor": "#2a78d6", "CAFTB-Net": "#eb6834", "ADCD-Net": "#1baf7a"}   # slot 1-3, đã validate all-pairs
MARK = {"TruFor": "o", "CAFTB-Net": "s", "ADCD-Net": "^"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SUCCESS_IOU = 0.3
B = 10000
rng = np.random.default_rng(0)


# ---------------------------------------------------------------- tiện ích
def wilson(k, n, z=1.959964):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def boot_ci(vals, stat=np.mean, b=B):
    """Percentile bootstrap 95% (resample ảnh). vals: 1-D array."""
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan, 0)
    idx = rng.integers(0, len(v), size=(b, len(v)))
    s = np.apply_along_axis(stat, 1, v[idx]) if stat is not np.mean else v[idx].mean(1)
    return (float(stat(v)), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5)), int(len(v)))


def fmt(t, nd=6):
    m, lo, hi, n = t
    if n == 0:
        return "n/a"
    return f"{m:.{nd}f} [{lo:.{nd}f}; {hi:.{nd}f}]"


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


# ---------------------------------------------------------------- nạp & thích ứng
def load(path):
    raw = pd.read_csv(path)
    d = pd.DataFrame({
        "image_id": raw.image_id, "dataset": raw.dataset, "model": raw.model,
        "theme": raw.theme, "tamper_type": raw.tamper_type, "is_clean": raw.is_clean.astype(bool),
        "iou_at_05": raw.IoU_05, "f1_at_05": raw.F1_05, "pr_auc": raw.PR_AUC,
        "roc_auc": np.nan, "oracle_f1": raw.oracle_F1, "tau_star": raw.tau_star,
        "pred_area_px": raw.pred_pixels_05, "gt_area_ratio": raw.gt_area_ratio,
        "gt_area_px": np.round(raw.gt_area_ratio * N_PIX).astype(int),
        "area_ratio_csv": raw.area_ratio_05, "fp_image": raw.FP_image,
        "inference_ms": raw.infer_time_sec * 1000, "peak_vram_mb": raw.peak_vram_mb,
        "gap_csv": raw.calibration_gap,
    })
    return raw, d


def wide(d, col):
    return d.pivot(index="image_id", columns="model", values=col)[MODELS]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="forensichub_benchmark_results.csv")
    ap.add_argument("--out-dir", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    raw, d = load(a.csv)
    R = {"schema_columns": list(raw.columns), "n_rows": len(d), "datasets": d.dataset.value_counts().to_dict()}
    T = d[~d.is_clean].copy()
    C = d[d.is_clean].copy()
    R["n_images"] = int(d.image_id.nunique())
    R["n_tampered"] = int(T.image_id.nunique())
    R["n_clean"] = int(C.image_id.nunique())
    R["all_models_same_images"] = len({frozenset(g.image_id) for _, g in d.groupby("model")}) == 1

    # kiểm tra nhất quán các cột dẫn xuất
    chk = T.assign(ar=T.pred_area_px / T.gt_area_px)
    R["consistency"] = {
        "max_abs_diff_area_ratio_vs_csv": float((chk.ar - chk.area_ratio_csv).abs().max()),
        "max_abs_diff_gap_vs_csv": float((T.oracle_f1 - T.f1_at_05 - T.gap_csv).abs().max()),
        "gt_area_px_integer_error_max": float((raw[~raw.is_clean].gt_area_ratio * N_PIX -
                                                np.round(raw[~raw.is_clean].gt_area_ratio * N_PIX)).abs().max()),
        "gt_area_ratio_same_across_models": bool((T.groupby("image_id").gt_area_ratio.nunique() == 1).all()),
    }

    # ================= A. hiệu năng cơ bản =================
    iou, f1, pra = wide(T, "iou_at_05"), wide(T, "f1_at_05"), wide(T, "pr_auc")
    nT = len(iou)
    idx = rng.integers(0, nT, size=(B, nT))          # resample ảnh dùng chung cho 3 mô hình (ghép cặp)
    A = {}
    for m in MODELS:
        row = {}
        for name, W in (("IoU@0.5", iou), ("F1@0.5", f1), ("PR-AUC", pra)):
            v = W[m].to_numpy()
            s = v[idx].mean(1)
            row[name] = (float(v.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5)), nT)
        succ = (iou[m] > SUCCESS_IOU).to_numpy().astype(float)
        s = succ[idx].mean(1)
        row["success_rate(IoU>0.3)"] = (float(succ.mean()), float(np.percentile(s, 2.5)),
                                        float(np.percentile(s, 97.5)), nT)
        row["ROC-AUC"] = None
        A[m] = row
    R["A_table"] = A
    R["A_success_counts"] = {m: int((iou[m] > SUCCESS_IOU).sum()) for m in MODELS}
    # McNemar (exact, hai phía) trên phân loại thành công IoU>0.3
    mc = {}
    for i, m1 in enumerate(MODELS):
        for m2 in MODELS[i + 1:]:
            s1, s2 = iou[m1] > SUCCESS_IOU, iou[m2] > SUCCESS_IOU
            b = int((s1 & ~s2).sum()); c = int((~s1 & s2).sum())
            p = float(stats.binomtest(b, b + c, 0.5).pvalue) if b + c > 0 else None
            mc[f"{m1} vs {m2}"] = {"both_succ": int((s1 & s2).sum()), "only_first": b, "only_second": c,
                                   "both_fail": int((~s1 & ~s2).sum()), "p_exact": p}
    R["A_mcnemar"] = mc
    # so sánh ghép cặp IoU liên tục (chẩn đoán): Wilcoxon
    R["A_wilcoxon_iou"] = {}
    for i, m1 in enumerate(MODELS):
        for m2 in MODELS[i + 1:]:
            R["A_wilcoxon_iou"][f"{m1} vs {m2}"] = float(stats.wilcoxon(iou[m1], iou[m2]).pvalue)

    # ================= B. chế độ hỏng =================
    B_ = {}
    for m in MODELS:
        t = T[T.model == m]
        ar = (t.pred_area_px / t.gt_area_px).to_numpy()
        n = len(ar)
        k = {"IM LẶNG (<0.5)": int((ar < 0.5).sum()),
             "KHỚP (0.5–2.0)": int(((ar >= 0.5) & (ar <= 2.0)).sum()),
             "BÃO HOÀ (>2.0)": int((ar > 2.0).sum())}
        B_[m] = {"n": n, "counts": k, "wilson95": {kk: wilson(v, n) for kk, v in k.items()},
                 "area_ratio_min": float(ar.min()), "area_ratio_median": float(np.median(ar)),
                 "area_ratio_max": float(ar.max())}
        c = C[C.model == m]
        fl = int((c.pred_area_px > 0).sum())
        B_[m]["clean"] = {"n": len(c), "flagged_images": fl, "FPR_image": fl / len(c),
                          "FPR_wilson95": wilson(fl, len(c)),
                          "pred_px_min": int(c.pred_area_px.min()), "pred_px_median": float(c.pred_area_px.median()),
                          "pred_px_max": int(c.pred_area_px.max()),
                          "pred_px_median_pct_of_image": float(c.pred_area_px.median() / N_PIX * 100)}
    R["B"] = B_

    # ================= C. hiệu chỉnh (Oracle-F1 CHỈ LÀ CHẨN ĐOÁN) =================
    C_ = {}
    for m in MODELS:
        t = T[T.model == m]
        gap = (t.oracle_f1 - t.f1_at_05).to_numpy()
        ts = t.tau_star.to_numpy()
        hist, _ = np.histogram(ts, bins=np.linspace(0, 1, 11))
        C_[m] = {"gap_median": boot_ci(gap, np.median), "oracle_f1_median_DIAGNOSTIC": boot_ci(t.oracle_f1, np.median),
                 "f1_at_05_median": boot_ci(t.f1_at_05, np.median),
                 "tau_quantiles": {str(q): float(np.quantile(ts, q)) for q in (0, .05, .25, .5, .75, .95, 1)},
                 "tau_hist_10bins": hist.tolist(), "tau_lt_0.05": int((ts < 0.05).sum()),
                 "tau_gt_0.95": int((ts > 0.95).sum()), "n": len(ts)}
    R["C"] = C_

    # ================= D. phân tầng =================
    def strat(mask_fn, labels, key):
        out = {}
        for lab in labels:
            sub = T[mask_fn(T, lab)]
            ids = sub.image_id.unique()
            if len(ids) == 0:
                continue
            out[lab] = {"n_images": int(len(ids))}
            for m in MODELS:
                s = sub[sub.model == m]
                out[lab][m] = {"IoU@0.5": boot_ci(s.iou_at_05), "F1@0.5": boot_ci(s.f1_at_05),
                               "PR-AUC": boot_ci(s.pr_auc),
                               "success": int((s.iou_at_05 > SUCCESS_IOU).sum())}
        return out
    bins = [("<0.1%", 0, 0.001), ("0.1-1%", 0.001, 0.01), ("1-5%", 0.01, 0.05), (">5%", 0.05, 1.01)]
    R["D_area"] = strat(lambda t, lab: (t.gt_area_ratio >= dict((b[0], b[1]) for b in bins)[lab]) &
                        (t.gt_area_ratio < dict((b[0], b[2]) for b in bins)[lab]), [b[0] for b in bins], "area")
    R["D_tamper_type"] = strat(lambda t, lab: t.tamper_type == lab, sorted(T.tamper_type.unique()), "tt")
    R["D_theme"] = strat(lambda t, lab: t.theme == lab, sorted(T.theme.unique()), "theme")
    # xu hướng: tương quan hạng IoU ~ gt_area_ratio
    R["D_spearman_iou_vs_area"] = {}
    for m in MODELS:
        s = T[T.model == m]
        r, p = stats.spearmanr(s.gt_area_ratio, s.iou_at_05)
        R["D_spearman_iou_vs_area"][m] = {"rho": float(r), "p": float(p), "n": len(s)}
    # FPR theo theme trên ảnh sạch
    R["D_clean_by_theme"] = {th: int(C[C.theme == th].image_id.nunique()) for th in sorted(C.theme.unique())}

    # ================= E. đồng thuận =================
    S = (iou > SUCCESS_IOU).astype(int)
    cnt = S.sum(1)
    E = {"pattern_counts": {"cả 3 thành công": int((cnt == 3).sum()), "đúng 2": int((cnt == 2).sum()),
                            "đúng 1": int((cnt == 1).sum()), "cả 3 thất bại": int((cnt == 0).sum())}}
    with np.errstate(all="ignore"):
        phi = S.corr().to_numpy()
    E["phi_matrix"] = [[None if np.isnan(x) else float(x) for x in r] for r in phi]
    E["success_constant_columns"] = [m for m in MODELS if S[m].nunique() < 2]
    sp = iou.corr(method="spearman")
    E["spearman_iou_matrix"] = sp.to_numpy().tolist()
    # tương quan riêng phần (loại ảnh hưởng của gt_area) trên hạng
    ga = T[T.model == MODELS[0]].set_index("image_id").gt_area_ratio.reindex(iou.index)
    rk = lambda s: stats.rankdata(s)
    res = {}
    for m in MODELS:
        x, g = rk(iou[m]), rk(ga)
        sl = np.polyfit(g, x, 1)
        res[m] = x - np.polyval(sl, g)
    E["partial_spearman_iou_given_area"] = {
        f"{m1} vs {m2}": float(np.corrcoef(res[m1], res[m2])[0, 1])
        for i, m1 in enumerate(MODELS) for m2 in MODELS[i + 1:]}
    R["E"] = E

    # ================= F. vận hành =================
    R["F"] = {m: {"inference_ms_median": float(d[d.model == m].inference_ms.median()),
                  "inference_ms_p25_p75": [float(d[d.model == m].inference_ms.quantile(.25)),
                                            float(d[d.model == m].inference_ms.quantile(.75))],
                  "peak_vram_mb_median": float(d[d.model == m].peak_vram_mb.median()),
                  "peak_vram_mb_unique": sorted(d[d.model == m].peak_vram_mb.unique().tolist())} for m in MODELS}
    R["F_input_tensor_mib_512"] = 1 * 3 * 512 * 512 * 4 / 2**20

    # ================= G. hợp lệ =================
    G = {}
    G["iou_exact_zero_all_models"] = int(((iou == 0).all(axis=1)).sum())
    G["iou_exact_zero_any_model"] = int(((iou == 0).any(axis=1)).sum())
    G["iou_min_over_all"] = float(iou.min().min())
    G["roc_auc_available"] = False
    G["tau_edge"] = {m: {"lt_0.05": C_[m]["tau_lt_0.05"], "gt_0.95": C_[m]["tau_gt_0.95"], "n": C_[m]["n"]}
                     for m in MODELS}
    # --- kiểm tra "đầu ra có phải np.random.rand(512,512)>=0.5 không?" ---
    p_exp, sd_exp = 0.5 * N_PIX, np.sqrt(N_PIX * 0.25)
    rnd = {"expected_pred_px_binomial": p_exp, "expected_sd": float(sd_exp), "clean": {}}
    for m in MODELS:
        x = C[C.model == m].pred_area_px.to_numpy()
        rnd["clean"][m] = {"mean": float(x.mean()), "sd": float(x.std(ddof=1)),
                           "z_of_mean": float((x.mean() - p_exp) / (sd_exp / np.sqrt(len(x)))), "n": len(x)}
    cw = wide(C, "pred_area_px")
    rnd["clean_pred_px_pearson_between_models"] = cw.corr().to_numpy().tolist()
    # ảnh tampered: kỳ vọng với dự đoán ngẫu nhiên đều p=0.5: precision=g, recall=0.5
    g = T.groupby("image_id").gt_area_ratio.first().reindex(iou.index).to_numpy()
    exp_iou, exp_f1 = g / (1 + g), g / (0.5 + g)
    rnd["tampered"] = {}
    for m in MODELS:
        rnd["tampered"][m] = {
            "mean_obs_iou": float(iou[m].mean()), "mean_expected_iou_random": float(exp_iou.mean()),
            "mean_obs_f1": float(f1[m].mean()), "mean_expected_f1_random": float(exp_f1.mean()),
            "mean_obs_pr_auc": float(pra[m].mean()), "mean_gt_area_ratio(=PR-AUC của ngẫu nhiên)": float(g.mean()),
            "median_rel_err_iou_vs_random": float(np.median(np.abs(iou[m] - exp_iou) / exp_iou)),
            "median_rel_err_f1_vs_random": float(np.median(np.abs(f1[m] - exp_f1) / exp_f1)),
            "median_rel_err_prauc_vs_gt_ratio": float(np.median(np.abs(pra[m] - g) / g))}
    rnd["oracle_f1_median_all_models"] = {m: float(T[T.model == m].oracle_f1.median()) for m in MODELS}
    rnd["max_abs_iou_diff_between_models_median"] = float(
        np.median(np.abs(iou.max(axis=1) - iou.min(axis=1))))
    G["random_baseline_test"] = rnd
    R["G"] = G

    with open("benchmark_tables.json", "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=2, ensure_ascii=False, default=lambda o: None if o is None else str(o))

    # ================= HÌNH =================
    # 1. phân bố area_ratio theo mô hình
    fig, ax = plt.subplots(figsize=(8.2, 4.4), facecolor=SURF)
    style(ax)
    lo = 0.0
    allar = np.concatenate([(T[T.model == m].pred_area_px / T[T.model == m].gt_area_px).to_numpy() for m in MODELS])
    edges = np.logspace(np.log10(min(allar.min(), 0.1)), np.log10(max(allar.max(), 10)), 36)
    # ba phân bố gần như trùng nhau: vẽ độ dày giảm dần để cả ba cùng nhìn thấy
    for m, lw in zip(MODELS, (6.0, 3.6, 1.6)):
        t = T[T.model == m]
        ax.hist((t.pred_area_px / t.gt_area_px), bins=edges, histtype="step", lw=lw, color=COL[m], label=m)
    ax.text(0.03, 0.55, "Ba mô hình gần như trùng khít\n(độ dày nét khác nhau để cả ba cùng hiện)",
            transform=ax.transAxes, ha="left", fontsize=8, color=INK2)
    ax.set_xscale("log")
    for xv, lab in ((0.5, "0,5"), (2.0, "2,0")):
        ax.axvline(xv, color=INK2, lw=1, ls="--")
        ax.text(xv, ax.get_ylim()[1] * 0.96, f" {lab}", color=INK2, fontsize=8, va="top")
    ax.text(0.25, ax.get_ylim()[1] * 0.9, "IM LẶNG", color=INK2, fontsize=8, ha="center")
    ax.text(1.0, ax.get_ylim()[1] * 0.9, "KHỚP", color=INK2, fontsize=8, ha="center")
    ax.text(30, ax.get_ylim()[1] * 0.9, "BÃO HOÀ", color=INK2, fontsize=8, ha="center")
    ax.set_xlabel("area_ratio = pred_area_px / gt_area_px (thang log)")
    ax.set_ylabel("Số ảnh tampered")
    ax.set_title(f"Phân bố area_ratio theo mô hình (n = {nT} ảnh tampered)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout(); fig.savefig(f"{a.out_dir}/area_ratio_by_model.png", dpi=160); plt.close(fig)

    # 2. hiệu năng theo gt_area_ratio (IoU@0.5 trung bình + CI bootstrap)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURF, sharex=True)
    labs = [b[0] for b in bins if b[0] in R["D_area"]]
    xs = np.arange(len(labs))
    for ax, key, title in ((axs[0], "IoU@0.5", "IoU@0.5"), (axs[1], "F1@0.5", "F1@0.5")):
        style(ax)
        for j, m in enumerate(MODELS):
            mean = np.array([R["D_area"][l][m][key][0] for l in labs])
            lo_ = np.array([R["D_area"][l][m][key][1] for l in labs])
            hi_ = np.array([R["D_area"][l][m][key][2] for l in labs])
            x = xs + (j - 1) * 0.12
            ax.errorbar(x, mean, yerr=[mean - lo_, hi_ - mean], color=COL[m], marker=MARK[m], ms=6, lw=1.5,
                        capsize=3, label=m, mfc=COL[m], mec=SURF)
        ax.set_xticks(xs); ax.set_xticklabels([f"{l}\n(n={R['D_area'][l]['n_images']})" for l in labs])
        ax.set_xlabel("gt_area_ratio"); ax.set_ylabel(title + " (trung bình, CI bootstrap 95%)")
        ax.set_title(f"{title} theo diện tích vùng sửa", loc="left", fontsize=11)
    axs[0].legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout(); fig.savefig(f"{a.out_dir}/performance_by_gt_area.png", dpi=160); plt.close(fig)

    # 3. ma trận đồng thuận: số ảnh cả hai cùng thành công / cùng thất bại / bất đồng
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURF)
    ax = axs[0]; ax.set_facecolor(SURF)
    M = np.zeros((3, 3))
    for i, m1 in enumerate(MODELS):
        for j, m2 in enumerate(MODELS):
            M[i, j] = ((S[m1] == S[m2]).mean())
    im = ax.imshow(M, vmin=0, vmax=1, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                    color="white" if M[i, j] > 0.6 else INK, fontsize=10)
    ax.set_xticks(range(3)); ax.set_xticklabels(MODELS, fontsize=9, color=INK2)
    ax.set_yticks(range(3)); ax.set_yticklabels(MODELS, fontsize=9, color=INK2)
    ax.set_title("Tỉ lệ ảnh hai mô hình cùng kết quả\n(thành công/thất bại, IoU>0,3)", loc="left", fontsize=10, color=INK)
    ax = axs[1]; style(ax)
    labs2 = list(E["pattern_counts"].keys()); vals = list(E["pattern_counts"].values())
    ax.bar(range(4), vals, color=["#2a78d6"] * 4, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom", color=INK, fontsize=10)
    ax.set_xticks(range(4)); ax.set_xticklabels(labs2, fontsize=8)
    ax.set_ylabel("Số ảnh tampered"); ax.set_title(f"Số mô hình thành công trên mỗi ảnh (n = {nT})", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{a.out_dir}/agreement_matrix.png", dpi=160); plt.close(fig)

    # 4. (chẩn đoán) quan sát vs kỳ vọng của bộ dự đoán ngẫu nhiên
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURF)
    ax = axs[0]; style(ax)
    for m in MODELS:
        ax.scatter(exp_iou, iou[m], s=18, color=COL[m], marker=MARK[m], alpha=.8, label=m, edgecolors=SURF, linewidths=.5)
    lim = [0, max(exp_iou.max(), iou.max().max()) * 1.05]
    ax.plot(lim, lim, color=INK2, lw=1, ls="--"); ax.text(lim[1] * .55, lim[1] * .5, "y = x", color=INK2, fontsize=8)
    ax.set_xlabel("IoU kỳ vọng nếu dự đoán = np.random.rand ≥ 0,5:  g/(1+g)")
    ax.set_ylabel("IoU@0.5 quan sát"); ax.set_title("Quan sát vs kỳ vọng của bộ dự đoán ngẫu nhiên", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    ax = axs[1]; style(ax)
    for m in MODELS:
        x = C[C.model == m].pred_area_px / N_PIX
        ax.hist(x, bins=np.linspace(0.495, 0.505, 21), histtype="step", lw=1.8, color=COL[m], label=m)
    ax.axvline(0.5, color=INK2, lw=1, ls="--")
    ax.set_xlabel("Tỉ lệ pixel bị gắn cờ trên ảnh sạch (τ = 0,5)"); ax.set_ylabel("Số ảnh sạch")
    ax.set_title(f"Ảnh sạch: pixel gắn cờ quanh 50% (n = {len(C.image_id.unique())})", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{a.out_dir}/validity_random_baseline.png", dpi=160); plt.close(fig)
    print("done: benchmark_tables.json, figures/*.png")


if __name__ == "__main__":
    main()
