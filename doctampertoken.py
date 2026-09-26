import subprocess
from pathlib import Path

import os

def pull_doctamper_dataset(
    download_path="./data/doctamper",
    dataset_slug="dinmkeljiame/doctamper",
):
    """
    Tải DocTamper qua Kaggle CLI và trả về danh sách ảnh tìm thấy.

    Kaggle credentials must already be configured for the CLI (for example,
    with the credentials file downloaded from Kaggle account settings). A
    dataset does not contain your personal Kaggle API credentials.
    """
    print("Đang tải DocTamper từ Kaggle... (Dung lượng lớn, vui lòng đợi)")
    destination = Path(download_path).expanduser()
    destination.mkdir(parents=True, exist_ok=True)

    # Read the token and add it to environment
    env = os.environ.copy()
    token_path = Path.home() / ".kaggle" / "access_token"
    if token_path.is_file():
        with open(token_path, "r") as f:
            token = f.read().strip()
            env["KAGGLE_API_TOKEN"] = token
            # Optional: Also support legacy variable name if KAGGLE_API_TOKEN doesn't work for some Kaggle versions
            env["KAGGLE_KEY"] = token
            env["KAGGLE_USERNAME"] = "api_token"  # Dummy username just in case

    try:
        if not (destination / "DocTamperV1-TestingSet").exists():
            subprocess.run(
                [
                    "kaggle",
                    "datasets",
                    "download",
                    "-d",
                    dataset_slug,
                    "-p",
                    str(destination),
                    "--unzip",
                ],
                check=True,
                env=env
            )
        else:
            print("Dataset đã có sẵn, bỏ qua bước tải...")
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Không tìm thấy Kaggle CLI. Hãy cài đặt gói 'kaggle' và đảm bảo "
            "lệnh 'kaggle' có trong PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Tải dataset thất bại. Hãy kiểm tra slug dataset và cấu hình "
            "Kaggle API credentials."
        ) from exc

    # Handle LMDB Extraction for TestingSet
    import lmdb

    testing_lmdb_path = destination / "DocTamperV1-TestingSet"
    extracted_dir = destination / "DocTamperV1-TestingSet_Extracted"
    
    dataset_list = []
    
    if testing_lmdb_path.is_dir():
        if not extracted_dir.exists():
            print("Đang giải nén ảnh từ LMDB ra thư mục (chỉ làm lần đầu)...")
            extracted_dir.mkdir(parents=True, exist_ok=True)
            
            env = lmdb.open(str(testing_lmdb_path), readonly=True, lock=False)
            with env.begin() as txn:
                num_samples_bytes = txn.get(b'num-samples')
                num_samples = int(num_samples_bytes.decode()) if num_samples_bytes else 30000
                
                for i in range(num_samples):
                    img_key = f'image-{i:09d}'.encode()
                    lbl_key = f'label-{i:09d}'.encode()
                    
                    img_data = txn.get(img_key)
                    lbl_data = txn.get(lbl_key)
                    
                    if not img_data:
                        continue
                    
                    img_path = extracted_dir / f"{i:09d}_img.jpg"
                    lbl_path = extracted_dir / f"{i:09d}_mask.png"
                    
                    # Ghi file ảnh
                    with open(img_path, 'wb') as f:
                        f.write(img_data)
                        
                    # Ghi file mask
                    if lbl_data:
                        with open(lbl_path, 'wb') as f:
                            f.write(lbl_data)
                            
                    if (i + 1) % 5000 == 0:
                        print(f"Đã giải nén {i + 1}/{num_samples} ảnh...")
        
        # Build dataset_list
        for img_path in extracted_dir.glob("*_img.jpg"):
            idx_str = img_path.name.split('_')[0]
            mask_path = extracted_dir / f"{idx_str}_mask.png"
            
            dataset_list.append({
                'img_id': idx_str,
                'dataset': 'DocTamper',
                'img_path': str(img_path),
                'mask_path': str(mask_path) if mask_path.is_file() else None,
                'theme': 'Unknown',
                'tamper_type': 'Unknown'
            })
            
    print(f"Đã nạp {len(dataset_list)} mẫu từ DocTamper.")
    return dataset_list