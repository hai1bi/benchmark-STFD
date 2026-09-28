"""
make_stfd_sample.py
===================
Chọn mẫu phân tầng từ STFD (mặc định 100 ảnh mỗi loại thao tác, seed 0) dựa trên stfd_stats.csv,
trích ảnh + mask từ zip (mật khẩu công khai của tác giả) vào data/stfd_sample/, ghi data/stfd_sample.json.

Chạy:  python make_stfd_sample.py [--per-type 100] [--seed 0]
"""
import argparse
import json
import os

import pandas as pd
import pyzipper

ZIP = "data/stfd/STFD_ICASSP2023.zip"
ROOT = "STFD_ICASSP2023"
PASSWORD = b"STFD_ICASSP2023"
OUT = "data/stfd_sample"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    df = pd.read_csv("stfd_stats.csv")
    parts = [g.sample(min(a.per_type, len(g)), random_state=a.seed) for _, g in df.groupby("tamper_type")]
    sample = pd.concat(parts).sort_values(["tamper_type", "image_id"]).reset_index(drop=True)

    os.makedirs(f"{OUT}/images", exist_ok=True)
    os.makedirs(f"{OUT}/masks", exist_ok=True)
    zf = pyzipper.AESZipFile(ZIP)
    zf.setpassword(PASSWORD)
    records = []
    for r in sample.itertuples():
        cat = r.path.split("/")[0]                       # ví dụ "3_Removal"
        name = r.path.split("/")[-1]
        key = f"{r.tamper_type}_{r.image_id}"
        ip, mp = f"{OUT}/images/{key}.png", f"{OUT}/masks/{key}.png"
        if not os.path.exists(ip):
            with open(ip, "wb") as f:
                f.write(zf.read(f"{ROOT}/{cat}/tamper/{name}"))
        if not os.path.exists(mp):
            with open(mp, "wb") as f:
                f.write(zf.read(f"{ROOT}/{cat}/masks/{name}"))
        records.append({"img_id": key, "dataset": "STFD", "img_path": ip, "mask_path": mp,
                        "tamper_type": r.tamper_type, "theme": r.theme_est,
                        "width": int(r.width), "height": int(r.height)})
    with open("data/stfd_sample.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=1, ensure_ascii=False)
    print(f"{len(records)} ảnh -> data/stfd_sample.json")
    print(sample.groupby("tamper_type").size().to_dict())
    print(sample.assign(res=sample.width.astype(str) + "x" + sample.height.astype(str)).res.value_counts().head(6).to_dict())
    print(sample.theme_est.value_counts().to_dict())


if __name__ == "__main__":
    main()
