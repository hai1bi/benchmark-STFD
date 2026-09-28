"""
analyze_stfd.py
===============
Phân tích kết quả benchmark THẬT trên STFD (schema của run_stfd_benchmark.py): mục A-G của yêu cầu + đối chiếu
3 giả thuyết. Tái dùng các hàm thống kê/định dạng của analyze_benchmark.py.

Chạy:  python analyze_stfd.py --csv stfd_benchmark_results.csv [--out-json benchmark_tables_stfd.json] [--out-dir figures_stfd]

Khác với analyze_benchmark.py (schema cũ): có ROC-AUC/AP, không có ảnh sạch (STFD), phân tầng theo 5 loại thao tác
thật, theo độ phân giải, và theo HÌNH DẠNG mask (độ lấp đầy hộp bao của các thành phần liên thông, lấy từ
stfd_stats.csv của bước audit) để kiểm chứng trực tiếp giả thuyết "hộp phát hiện được, sát nét chữ thì không".
Oracle-F1 CHỈ là chẩn đoán/cận trên, không bao giờ được báo cáo như kết quả.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import analyze_benchmark as ab
from analyze_benchmark import GRID, INK, INK2, MARK, MODELS, SURF, boot_ci, style, wilson

COL = ab.COL
SUCCESS = 0.3
B = 10000
N_PIX_NOTE = "lưới đánh giá <= 2 MP (thay đổi theo ảnh)"
AREA_BINS = [("<0.1%", 0, 0.001), ("0.1-1%", 0.001, 0.01), ("1-5%", 0.01, 0.05), (">5%", 0.05, 1.01)]
FILL_BINS = [("tight (<0.5)", 0, 0.5), ("mid (0.5-0.85)", 0.5, 0.85), ("box-like (>=0.85)", 0.85, 1.01)]
METRICS = [("IoU@0.5", "IoU_05"), ("F1@0.5", "F1_05"), ("PR-AUC", "PR_AUC"), ("ROC-AUC", "ROC_AUC")]


def load(csv_path, stats_csv="stfd_stats.csv"):
    raw = pd.read_csv(csv_path)
    R = {"n_rows_raw": len(raw), "n_error_rows": int(raw.error.fillna("").astype(str).str.len().gt(0).sum())}
    R["error_examples"] = raw[raw.error.fillna("").astype(str).str.len() > 0].error.head(5).tolist()
    d = raw[raw.error.fillna("").astype(str).str.len() == 0].copy()
    # chỉ giữ ảnh có mặt ở đủ 3 mô hình (so sánh ghép cặp)
    cnt = d.groupby("image_id").model.nunique()
    common = cnt[cnt == len(MODELS)].index
    R["n_images_all_models"] = int(len(common))
    R["n_images_dropped_incomplete"] = int(d.image_id.nunique() - len(common))
    d = d[d.image_id.isin(common)].copy()
    st = pd.read_csv(stats_csv)
    st["image_id_key"] = st.tamper_type + "_" + st.image_id.astype(str)
    st = st.set_index("image_id_key")[["comp_fill_weighted", "largest_comp_fill", "n_components"]]
    d = d.join(st, on="image_id")
    d["res"] = d.width.astype(int).astype(str) + "x" + d.height.astype(int).astype(str)
    return raw, d, R


def wide(d, col):
    return d.pivot(index="image_id", columns="model", values=col)[MODELS]


def strat_table(T, mask_fn, labels):
    out = {}
    for lab in labels:
        sub = T[mask_fn(T, lab)]
        ids = sub.image_id.unique()
        if len(ids) == 0:
            continue
        out[lab] = {"n_images": int(len(ids))}
        for m in MODELS:
            s = sub[sub.model == m]
            out[lab][m] = {k: boot_ci(s[c]) for k, c in METRICS}
            out[lab][m]["success"] = int((s.IoU_05 > SUCCESS).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="stfd_benchmark_results.csv")
    ap.add_argument("--out-json", default="benchmark_tables_stfd.json")
    ap.add_argument("--out-dir", default="figures_stfd")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    raw, d, R = load(a.csv)
    R["datasets"] = raw.dataset.value_counts().to_dict()
    T = d[~d.is_clean.astype(bool)].copy()
    C = d[d.is_clean.astype(bool)].copy()
    R["n_tampered"] = int(T.image_id.nunique())
    R["n_clean"] = int(C.image_id.nunique())
    nT = R["n_tampered"]
    rng = ab.rng

    iou, f1, pra, roc = (wide(T, c) for c in ("IoU_05", "F1_05", "PR_AUC", "ROC_AUC"))
    ids = iou.index
    idx = rng.integers(0, nT, size=(B, nT))

    # ================= A =================
    A = {}
    for m in MODELS:
        row = {}
        for name, W in (("IoU@0.5", iou), ("F1@0.5", f1), ("PR-AUC", pra), ("ROC-AUC", roc)):
            v = W[m].to_numpy()
            s = v[idx].mean(1)
            row[name] = (float(v.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5)), nT)
        succ = (iou[m] > SUCCESS).to_numpy()
        row["success_count(IoU>0.3)"] = int(succ.sum())
        row["success_rate_wilson95"] = (float(succ.mean()), *wilson(int(succ.sum()), nT))
        A[m] = row
    R["A_table"] = A
    mc, pd_diff = {}, {}
    for i, m1 in enumerate(MODELS):
        for m2 in MODELS[i + 1:]:
            s1, s2 = iou[m1] > SUCCESS, iou[m2] > SUCCESS
            b = int((s1 & ~s2).sum()); c = int((~s1 & s2).sum())
            mc[f"{m1} vs {m2}"] = {"both_succ": int((s1 & s2).sum()), "only_first": b, "only_second": c,
                                   "both_fail": int((~s1 & ~s2).sum()),
                                   "p_exact": float(stats.binomtest(b, b + c, 0.5).pvalue) if b + c else None}
            dd = {}
            for name, W in (("IoU@0.5", iou), ("F1@0.5", f1), ("PR-AUC", pra), ("ROC-AUC", roc)):
                v = (W[m1] - W[m2]).to_numpy()
                s = v[idx].mean(1)
                lo, hi = float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))
                dd[name] = {"mean_diff": float(v.mean()), "ci95": [lo, hi], "significant": bool(lo > 0 or hi < 0)}
            pd_diff[f"{m1} - {m2}"] = dd
    R["A_mcnemar"] = mc
    R["A_paired_bootstrap_diff"] = pd_diff

    # ================= B =================
    Bd = {}
    for m in MODELS:
        t = T[T.model == m]
        ar = t.area_ratio_05.to_numpy()
        k = {"IM LẶNG (<0.5)": int((ar < 0.5).sum()), "KHỚP (0.5–2.0)": int(((ar >= 0.5) & (ar <= 2.0)).sum()),
             "BÃO HOÀ (>2.0)": int((ar > 2.0).sum())}
        fpo = t.fp_rate_outside_05.to_numpy()
        Bd[m] = {"n": len(ar), "counts": k, "wilson95": {kk: wilson(v, len(ar)) for kk, v in k.items()},
                 "area_ratio": {"min": float(ar.min()), "median": float(np.median(ar)), "max": float(ar.max())},
                 "pred_area_zero": int((t.pred_pixels_05 == 0).sum()),
                 "fp_rate_outside_median": boot_ci(fpo, np.median), "fp_rate_outside_mean": boot_ci(fpo),
                 "fp_rate_outside_gt1pct": int((fpo > 0.01).sum()),
                 "pred_area_ratio_img_median": float(t.pred_area_ratio_img.median())}
    R["B"] = Bd
    R["B_clean_note"] = ("STFD không có ảnh sạch: FPR mức ảnh KHÔNG tính được. fp_rate_outside_05 (tỉ lệ pixel ngoài "
                         "vùng sửa bị gắn cờ trên ảnh bị sửa) chỉ là chẩn đoán thay thế, không phải FPR mức ảnh."
                         if len(C) == 0 else "có ảnh sạch")

    # ================= C (Oracle: chẩn đoán) =================
    Cd = {}
    for m in MODELS:
        t = T[T.model == m]
        gap = (t.oracle_F1 - t.F1_05).to_numpy()
        ts = t.tau_star.to_numpy()
        hist, _ = np.histogram(ts, bins=np.linspace(0, 1.0000001, 11))
        oracle_med = float(np.median(t.oracle_F1))
        gap_med = float(np.median(gap))
        verdict = ("lệch ngưỡng (cứu được bằng hiệu chỉnh)" if (gap_med > 0.05 and oracle_med >= 0.3)
                   else "sụp biểu diễn (oracle cũng thấp)" if oracle_med < 0.1 else "trung gian / không rõ")
        Cd[m] = {"gap_median": boot_ci(gap, np.median), "oracle_f1_median_DIAGNOSTIC": boot_ci(t.oracle_F1, np.median),
                 "f1_at_05_median": boot_ci(t.F1_05, np.median),
                 "tau_quantiles": {str(q): float(np.quantile(ts, q)) for q in (0, .05, .25, .5, .75, .95, 1)},
                 "tau_hist_10bins": hist.tolist(), "tau_eq_0": int((ts == 0).sum()), "tau_ge_0.98": int((ts >= 0.98).sum()),
                 "n": len(ts), "verdict_rule": "gap>0.05 & oracle>=0.3 -> lệch ngưỡng; oracle<0.1 -> sụp; else trung gian (quy ước)",
                 "verdict": verdict}
    R["C"] = Cd

    # ================= D =================
    R["D_area"] = strat_table(T, lambda t, l: (t.gt_area_ratio >= dict((b[0], b[1]) for b in AREA_BINS)[l]) &
                              (t.gt_area_ratio < dict((b[0], b[2]) for b in AREA_BINS)[l]), [b[0] for b in AREA_BINS])
    R["D_tamper_type"] = strat_table(T, lambda t, l: t.tamper_type == l, sorted(T.tamper_type.unique()))
    R["D_theme"] = strat_table(T, lambda t, l: t.theme == l, sorted(T.theme.unique()))
    topres = T.drop_duplicates("image_id").res.value_counts()
    keep = [r for r in topres.index[:3]]
    T["res_grp"] = np.where(T.res.isin(keep), T.res, "khác")
    R["D_resolution"] = strat_table(T, lambda t, l: t.res_grp == l, keep + ["khác"])
    R["D_fill_weighted"] = strat_table(T, lambda t, l: (t.comp_fill_weighted >= dict((b[0], b[1]) for b in FILL_BINS)[l]) &
                                       (t.comp_fill_weighted < dict((b[0], b[2]) for b in FILL_BINS)[l]), [b[0] for b in FILL_BINS])
    R["D_fill_largest"] = strat_table(T, lambda t, l: (t.largest_comp_fill >= dict((b[0], b[1]) for b in FILL_BINS)[l]) &
                                      (t.largest_comp_fill < dict((b[0], b[2]) for b in FILL_BINS)[l]), [b[0] for b in FILL_BINS])
    # fill theo từng loại thao tác (hộp/tight có lẫn với loại thao tác)
    R["D_fill_by_type_counts"] = pd.crosstab(T.drop_duplicates("image_id").tamper_type,
                                             pd.cut(T.drop_duplicates("image_id").comp_fill_weighted,
                                                    [0, 0.5, 0.85, 1.01], right=False,
                                                    labels=["tight", "mid", "box-like"])).to_dict("index")
    sp = {}
    for m in MODELS:
        s = T[T.model == m]
        r1 = stats.spearmanr(s.gt_area_ratio, s.IoU_05)
        r2 = stats.spearmanr(s.comp_fill_weighted, s.IoU_05)
        r3 = stats.spearmanr(s.comp_fill_weighted, s.gt_area_ratio)
        sp[m] = {"iou_vs_area": [float(r1[0]), float(r1[1])], "iou_vs_fill": [float(r2[0]), float(r2[1])],
                 "fill_vs_area(confound)": [float(r3[0]), float(r3[1])], "n": len(s)}
    R["D_spearman"] = sp

    # ================= E =================
    E = {}
    for label, thr in (("iou>0.3 (theo đặc tả)", 0.3), ("iou>0.1 (bổ sung, ngoài đặc tả)", 0.1)):
        S = (iou > thr).astype(int)
        cnt = S.sum(1)
        with np.errstate(all="ignore"):
            phi = S.corr().to_numpy()
        E[label] = {"pattern_counts": {"cả 3 thành công": int((cnt == 3).sum()), "đúng 2": int((cnt == 2).sum()),
                                       "đúng 1": int((cnt == 1).sum()), "cả 3 thất bại": int((cnt == 0).sum())},
                    "phi_matrix": [[None if np.isnan(x) else float(x) for x in r] for r in phi],
                    "constant_columns": [m for m in MODELS if S[m].nunique() < 2]}
    E["spearman_iou_matrix"] = iou.corr(method="spearman").to_numpy().tolist()
    # tương quan riêng phần của IoU liên tục giữa các mô hình sau khi loại diện tích (trên hạng)
    ga = T[T.model == MODELS[0]].set_index("image_id").gt_area_ratio.reindex(ids)
    res = {}
    for m in MODELS:
        x, g = stats.rankdata(iou[m]), stats.rankdata(ga)
        res[m] = x - np.polyval(np.polyfit(g, x, 1), g)
    E["partial_spearman_given_area"] = {f"{m1} vs {m2}": float(np.corrcoef(res[m1], res[m2])[0, 1])
                                        for i, m1 in enumerate(MODELS) for m2 in MODELS[i + 1:]}
    R["E"] = E

    # ================= F =================
    R["F"] = {m: {"infer_ms_median": float(d[d.model == m].infer_ms.median()),
                  "infer_ms_p25_p75": [float(d[d.model == m].infer_ms.quantile(.25)), float(d[d.model == m].infer_ms.quantile(.75))],
                  "ocr_ms_median": float(d[d.model == m].ocr_ms.median()),
                  "peak_vram_mb_median": float(d[d.model == m].peak_vram_mb.median()),
                  "peak_vram_mb_max": float(d[d.model == m].peak_vram_mb.max())} for m in MODELS}
    R["F_eval_grid"] = {"scale_min": float(d.scale.min()), "scale_median": float(d.scale.median()),
                        "images_resized": int((d.drop_duplicates('image_id').scale < 1).sum()),
                        "eval_pixels_median": float((d.eval_w * d.eval_h).median())}

    # ================= G =================
    G = {"iou_exact_zero_all_models": int(((iou == 0).all(axis=1)).sum()),
         "iou_exact_zero_any_model": int(((iou == 0).any(axis=1)).sum()),
         "iou_exact_zero_per_model": {m: int((iou[m] == 0).sum()) for m in MODELS}}
    G["roc_vs_pr"] = {m: {"roc_auc_median": float(roc[m].median()), "pr_auc_median": float(pra[m].median()),
                          "base_rate_median(gt_area_ratio)": float(T[T.model == m].gt_area_ratio.median()),
                          "roc_auc_mean": float(roc[m].mean()), "pr_auc_mean": float(pra[m].mean())} for m in MODELS}
    g = T[T.model == MODELS[0]].set_index("image_id").gt_area_ratio.reindex(ids).to_numpy()
    rnd = {}
    for m in MODELS:
        exp_iou, exp_f1 = g / (1 + g), g / (0.5 + g)
        rnd[m] = {"median_rel_err_iou_vs_random": float(np.median(np.abs(iou[m] - exp_iou) / exp_iou)),
                  "median_rel_err_f1_vs_random": float(np.median(np.abs(f1[m] - exp_f1) / exp_f1)),
                  "median_rel_err_prauc_vs_base_rate": float(np.median(np.abs(pra[m] - g) / g)),
                  "roc_auc_median": float(roc[m].median())}
        # "giống ngẫu nhiên" nếu IoU/F1 trùng kỳ vọng (<5%) VÀ ROC-AUC ~ 0.5
        rnd[m]["random_like"] = bool(rnd[m]["median_rel_err_iou_vs_random"] < 0.05 and abs(rnd[m]["roc_auc_median"] - 0.5) < 0.05)
    G["random_baseline_test"] = rnd
    G["any_random_like"] = any(v["random_like"] for v in rnd.values())
    G["between_model_iou_max_min_median_absdiff"] = float(np.median(np.abs(iou.max(axis=1) - iou.min(axis=1))))
    G["tau_edges"] = {m: {"eq_0": Cd[m]["tau_eq_0"], "ge_0.98": Cd[m]["tau_ge_0.98"], "n": Cd[m]["n"]} for m in MODELS}
    G["nan_counts"] = {c: int(T[c].isna().sum()) for c in ("IoU_05", "F1_05", "PR_AUC", "ROC_AUC", "oracle_F1")}
    R["G"] = G

    with open(a.out_json, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=2, ensure_ascii=False, default=lambda o: None if o is None else str(o))

    # ================= hình =================
    od = a.out_dir
    # 1. phân bố area_ratio
    fig, ax = plt.subplots(figsize=(8.2, 4.4), facecolor=SURF)
    style(ax)
    allar = np.concatenate([T[T.model == m].area_ratio_05.to_numpy() for m in MODELS])
    allar = allar[allar > 0]
    lo = min(allar.min(), 0.05)
    edges = np.logspace(np.log10(lo), np.log10(max(allar.max(), 10)), 40)
    for m, lw in zip(MODELS, (5.0, 3.0, 1.6)):
        v = T[T.model == m].area_ratio_05.to_numpy()
        ax.hist(np.clip(v, edges[0], None), bins=edges, histtype="step", lw=lw, color=COL[m], label=f"{m} (im lặng: {(v < .5).sum()}/{len(v)})")
    ax.set_xscale("log")
    top = ax.get_ylim()[1]
    for xv, lab in ((0.5, "0,5"), (2.0, "2,0")):
        ax.axvline(xv, color=INK2, lw=1, ls="--")
        ax.text(xv, top * 0.97, f" {lab}", color=INK2, fontsize=8, va="top")
    for xx, t_ in ((edges[0] * 1.5, "IM LẶNG"), (1.0, "KHỚP"), (edges[-1] / 3, "BÃO HOÀ")):
        ax.text(xx, top * 0.9, t_, color=INK2, fontsize=8, ha="center")
    ax.set_xlabel("area_ratio = pred_pixels@0.5 / gt_area_px (thang log)")
    ax.set_ylabel("Số ảnh bị sửa")
    ax.set_title(f"Phân bố area_ratio theo mô hình (n = {nT} ảnh STFD)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.tight_layout(); fig.savefig(f"{od}/area_ratio_by_model.png", dpi=160); plt.close(fig)

    # 2. hiệu năng theo gt_area_ratio
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURF, sharex=True)
    labs = [b[0] for b in AREA_BINS if b[0] in R["D_area"]]
    xs = np.arange(len(labs))
    for ax, key in ((axs[0], "IoU@0.5"), (axs[1], "F1@0.5")):
        style(ax)
        for j, m in enumerate(MODELS):
            mean = np.array([R["D_area"][l][m][key][0] for l in labs])
            lo_ = np.array([R["D_area"][l][m][key][1] for l in labs])
            hi_ = np.array([R["D_area"][l][m][key][2] for l in labs])
            ax.errorbar(xs + (j - 1) * 0.12, mean, yerr=[mean - lo_, hi_ - mean], color=COL[m], marker=MARK[m], ms=6,
                        lw=1.5, capsize=3, label=m, mfc=COL[m], mec=SURF)
        ax.set_xticks(xs); ax.set_xticklabels([f"{l}\n(n={R['D_area'][l]['n_images']})" for l in labs])
        ax.set_xlabel("gt_area_ratio"); ax.set_ylabel(key + " (trung bình, CI bootstrap 95%)")
        ax.set_title(f"{key} theo diện tích vùng sửa", loc="left", fontsize=11)
    axs[0].legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout(); fig.savefig(f"{od}/performance_by_gt_area.png", dpi=160); plt.close(fig)

    # 3. đồng thuận
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURF)
    key3 = "iou>0.3 (theo đặc tả)"
    S = (iou > SUCCESS).astype(int)
    ax = axs[0]; ax.set_facecolor(SURF)
    M = np.array([[(S[m1] == S[m2]).mean() for m2 in MODELS] for m1 in MODELS])
    ax.imshow(M, vmin=0, vmax=1, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", color="white" if M[i, j] > .6 else INK, fontsize=10)
    ax.set_xticks(range(3)); ax.set_xticklabels(MODELS, fontsize=9, color=INK2)
    ax.set_yticks(range(3)); ax.set_yticklabels(MODELS, fontsize=9, color=INK2)
    ax.set_title("Tỉ lệ ảnh hai mô hình cùng kết quả\n(thành công/thất bại, IoU>0,3)", loc="left", fontsize=10, color=INK)
    ax = axs[1]; style(ax)
    pc = E[key3]["pattern_counts"]
    ax.bar(range(4), list(pc.values()), color=["#2a78d6"] * 4, width=0.6)
    for i, v in enumerate(pc.values()):
        ax.text(i, v, str(v), ha="center", va="bottom", color=INK, fontsize=10)
    ax.set_xticks(range(4)); ax.set_xticklabels(list(pc.keys()), fontsize=8)
    ax.set_ylabel("Số ảnh bị sửa"); ax.set_title(f"Số mô hình thành công trên mỗi ảnh (n = {nT})", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{od}/agreement_matrix.png", dpi=160); plt.close(fig)

    # 4. theo loại thao tác và theo hình dạng mask
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURF)
    for ax, key, title, labs4 in ((axs[0], "D_tamper_type", "theo loại thao tác", sorted(R["D_tamper_type"].keys())),
                                  (axs[1], "D_fill_weighted", "theo hình dạng mask (độ lấp đầy hộp bao)", [b[0] for b in FILL_BINS if b[0] in R["D_fill_weighted"]])):
        style(ax)
        xs4 = np.arange(len(labs4))
        for j, m in enumerate(MODELS):
            mean = np.array([R[key][l][m]["IoU@0.5"][0] for l in labs4])
            lo_ = np.array([R[key][l][m]["IoU@0.5"][1] for l in labs4])
            hi_ = np.array([R[key][l][m]["IoU@0.5"][2] for l in labs4])
            ax.errorbar(xs4 + (j - 1) * 0.12, mean, yerr=[mean - lo_, hi_ - mean], color=COL[m], marker=MARK[m], ms=6, lw=0,
                        elinewidth=1.5, capsize=3, label=m, mfc=COL[m], mec=SURF)
        ax.set_xticks(xs4); ax.set_xticklabels([f"{l}\n(n={R[key][l]['n_images']})" for l in labs4], fontsize=8)
        ax.set_ylabel("IoU@0.5 (trung bình, CI bootstrap 95%)"); ax.set_title(f"IoU@0.5 {title}", loc="left", fontsize=10)
    axs[0].legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout(); fig.savefig(f"{od}/performance_by_type_and_shape.png", dpi=160); plt.close(fig)
    print("done:", a.out_json, od)


if __name__ == "__main__":
    main()
