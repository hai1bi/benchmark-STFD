"""
run_benchmark_with_forensichub.py
=================================
1. Nạp dataset từ JSON (đã phân loại theme, tamper_type)
2. Chạy evaluate_dataset → CSV per-image
3. Chạy analyze_results  → báo cáo 5 nhóm
"""
import os
from benchmark import evaluate_dataset, analyze_results
from forensichub_dataset import load_dataset_for_benchmark

OUTPUT_CSV = "forensichub_benchmark_results.csv"

def main():
    json_file = "./data/doctamper_forensichub.json"
    if not os.path.exists(json_file):
        print(f"ERROR: Khong tim thay {json_file}.")
        print("Hay chay:  python forensichub_dataset.py")
        return

    print("Dang nap dataset tu ForensicHub JSON...")
    dataset_list = load_dataset_for_benchmark(json_file)

    # --- Chon so mau de chay ---
    # Doi thanh dataset_list (khong slice) de chay toan bo 30k
    sample_list = dataset_list[:100]

    print(f"Da nap {len(dataset_list)} mau. "
          f"Chay tren {len(sample_list)} mau dau tien.")

    # Buoc 1: Danh gia → CSV
    evaluate_dataset(sample_list, output_csv=OUTPUT_CSV)

    # Buoc 2: Phan tich tu CSV
    analyze_results(OUTPUT_CSV)


if __name__ == "__main__":
    main()
