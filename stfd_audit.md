# Audit bộ dữ liệu STFD (Screenshot Text Forgery Dataset)

- Nguồn kiểm tra: `https://huggingface.co/datasets/Zegkim/STFD` (commit `9edebed`, cập nhật 2026-03-18), file `STFD_ICASSP2023.zip` (2 941 753 426 byte, khớp kích thước trên Hugging Face).
- Toàn bộ 3 932 cặp (ảnh, mask) đã được đọc và kiểm tra, không lấy mẫu, trừ các mục ghi rõ "mẫu".
- Tái tạo: xem mục 9. Số liệu tổng hợp ở `data/stfd_summary.json`, số liệu từng ảnh ở `stfd_stats.csv`.

## 1. Bảng tóm tắt

| # | Câu hỏi | Kết luận ngắn |
|---|---|---|
| 1 | Ảnh sạch | **KHÔNG có.** 3 932 cặp đều là ảnh bị can thiệp: 0 mask rỗng, không có thư mục hay danh sách ảnh sạch. Không tính được tỉ lệ báo động giả mức ảnh trên STFD. |
| 2 | Định dạng nén | **100% PNG** theo magic bytes (3 932/3 932), 0 JPEG, nên không có bảng lượng tử. Chỉ số gián tiếp cho thấy khoảng 20–38% ảnh có dấu vết lưới JPEG 8×8, tập trung ở vài độ phân giải. Chỉ số này chưa hiệu chuẩn. |
| 3 | Loại thao tác / hình dạng vùng sửa | Có nhãn 5 loại (theo tên thư mục). Hình dạng **không đồng nhất**: Insertion/Replacement/Removal chủ yếu ôm sát nét chữ hoặc vùng xóa; Splicing/Copy-move lẫn nhiều vùng dạng hộp (nút, nhãn UI). Mask nào cũng gồm nhiều thành phần rời (median 53). |
| 4 | Tỉ lệ diện tích sửa | min 0,0168%, median **0,407%**, max 37,48%. Histogram: `<0,1%` 245, `0,1–1%` 2 601, `1–5%` 953, `>5%` 133. |
| 5 | Mask | PNG; 3 929/3 932 nhị phân 0/255, 3 mask không nhị phân, 1 mask kiểu I;16. Kích thước khớp ảnh 3 932/3 932. Không có mask rỗng hay phủ toàn ảnh. |
| 6 | Tập biên lai chuyển tiền | **KHÔNG XÁC ĐỊNH ĐƯỢC.** Không có nhãn cảnh cho từng ảnh (tên file là hash, không có file metadata). |
| 7 | Siêu dữ liệu khác | Theme: không có nhãn; ước lượng Light 3 595 / Dark 337. 15 độ phân giải, 67,8% là 1080×2340. Không có nhãn app nguồn. Không có chia train/val/test. |
| 8 | Giấy phép, trích dẫn | **Mâu thuẫn:** metadata Hugging Face ghi CC-BY-4.0, nhưng README ghi "academic research purposes only". Bài báo: ICASSP 2023 (IEEE), có bản chính thức, không phải arXiv. |

## 2. Chi tiết

### 2.1 Cấu trúc và ảnh sạch (mục 1)

Danh mục zip (đọc bằng HTTP Range, không cần tải toàn bộ): 7 884 mục = 19 thư mục + 1 `Readme.md` + 7 864 file PNG.

| Thư mục | Ảnh tamper | Mask |
|---|---:|---:|
| 1_Copy-move | 758 | 758 |
| 2_Splicing | 830 | 830 |
| 3_Removal | 1 016 | 1 016 |
| 4_Insertion | 701 | 701 |
| 5_Replacement | 627 | 627 |
| **Tổng** | **3 932** | **3 932** |

- Tên file ảnh và mask khớp nhau hoàn toàn ở cả 5 thư mục (`unpaired = []`).
- `is_clean` (mask có 0 pixel trắng) = True cho **0** ảnh. `mask_full` = 0.
- Không có thư mục hay file nào ghi ảnh gốc/ảnh sạch. Cây thư mục trong `Readme.md` chỉ có `tamper/` và `masks/`.
- Repo GitHub `ZeqinYu/STFL-Net` chỉ có `code/`, `imgs/` (10 ảnh minh họa) và README: không có danh sách ảnh sạch.
- **Chưa kiểm chứng được:** bản của bài báo hay các bản trên Tianchi có ảnh sạch hay không (README nói một phần STFD từng dùng trong các cuộc thi Tianchi và ICDAR 2023 DTT).

### 2.2 Định dạng nén (mục 2)

- Đếm theo **nội dung file** (magic bytes), không theo đuôi: PNG 3 932, JPEG 0, khác 0. Mask cũng 100% PNG.
- README Hugging Face và `Readme.md` trong zip ghi "Format: PNG / JPEG", nhưng bản phát hành này **không có JPEG nào**.
- Chế độ màu ảnh: RGB 3 846, RGBA 86 (83 ảnh có alpha ≡ 255; 3 ảnh có alpha không đồng nhất).
- Không có bảng lượng tử để trích: `PIL.Image.quantization` chỉ tồn tại với JPEG; 0 ảnh JPEG nên cột `jpeg_qtables` trong CSV trống hoàn toàn. Không có "cụm bảng lượng tử" để báo cáo.
- Không có chunk metadata PNG (`png_text_keys` rỗng cho cả 3 932 ảnh).

**Dấu hiệu đã qua nén nhiều lần (chỉ báo gián tiếp):** PNG là định dạng không mất mát, nhưng ảnh có thể đã từng bị nén JPEG trước khi lưu PNG (ví dụ qua app hoặc mạng xã hội). Tôi dùng chỉ số `grid8_small` (tỉ lệ bước nhỏ khác 0 giữa các pixel kề nhau theo pha mod 8; xem `audit_stfd.py`):

| Phép đối chứng (20 ảnh, nén thêm JPEG) | median `grid8_small` |
|---|---:|
| PNG nguyên bản (min–max 1,01–1,56) | 1,07 |
| + JPEG q95 | 1,21 |
| + JPEG q85 | 1,40 |
| + JPEG q75 | 1,47 |

- Chỉ số **thứ nhất** (`grid8_naive`, max/median gradient theo pha) **thất bại đối chứng**: PNG 1,144 vs JPEG q75 1,144. Không dùng để kết luận. Vẫn giữ trong code và CSV để tái tạo.
- Trên toàn dataset (n = 3 932): median 1,129; p25 1,061; p75 1,335; p90 1,559. Có **1 495 ảnh (38,0%) > 1,2** và **773 ảnh (19,7%) > 1,4**, tức mức tương đương ảnh vừa bị nén q85 trở xuống.
- Chỉ số phụ thuộc mạnh vào độ phân giải (nghi liên quan thiết bị/app nguồn):

| Độ phân giải | n | median | % ảnh > 1,4 |
|---|---:|---:|---:|
| 1080×2340 | 2 665 | 1,11 | 14% |
| 1080×2400 | 504 | 1,32 | 38% |
| 750×1334 | 138 | 1,34 | 43% |
| 1080×2376 | 100 | 1,46 | 71% |
| 1080×2316 | 96 | 1,47 | 65% |
| các độ phân giải còn lại | 429 | 1,06–1,20 (riêng 1 ảnh 1079×1976: 1,37) | 0–16% |

- Kết luận thận trọng: **không xác định được số lần nén hay bảng lượng tử.** Có dấu hiệu gián tiếp rằng một phần ảnh (ước lượng 20–38%, phụ thuộc ngưỡng) từng qua nén JPEG trước khi lưu PNG. Ngưỡng chưa được hiệu chuẩn trên ảnh chắc chắn chưa từng nén (mẫu đối chứng PNG cũng có ảnh tới 1,56, tức có thể chính chúng đã có lịch sử nén).
- Phát biểu an toàn cho báo cáo: "STFD lưu dạng PNG (không mất mát); lịch sử nén của ảnh nguồn không rõ và có dấu vết JPEG ở một phần ảnh".

### 2.3 Loại thao tác và hình dạng vùng sửa (mục 3)

Phân bố nhãn (theo thư mục): Removal 1 016 (25,8%), Splicing 830 (21,1%), Copy-move 758 (19,3%), Insertion 701 (17,8%), Replacement 627 (15,9%). Không có nhãn nào khác ngoài tên thư mục.

**Chỉ số theo đúng yêu cầu (diện tích mask / diện tích hộp bao của toàn mask), cả 3 932 mask:**

| | min | p25 | median | p75 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|
| `mask_fill_ratio` | 0,0005 | 0,0065 | **0,0139** | 0,0331 | 0,0953 | 0,877 |

Khoảng: `<0,2`: 3 888; `0,2–0,5`: 38; `0,5–0,85`: 5; `0,85–0,95`: 1; `≥0,95`: 0.

**Chỉ số này không dùng được để trả lời "hộp hay ôm nét chữ".** Mask gồm nhiều vùng rời nhau (median **53 thành phần liên thông**, max 2 649, và **0/3 932 mask chỉ có 1 thành phần**), nên hộp bao của cả mask luôn rộng hơn nhiều so với các vùng thật. Vì vậy tôi tính thêm mức lấp đầy **theo từng thành phần liên thông**:

- `comp_fill_weighted` = Σ diện tích thành phần / Σ diện tích hộp bao từng thành phần.
- `largest_comp_fill` = diện tích / hộp bao của thành phần lớn nhất.

| | min | p25 | median | p75 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|
| `comp_fill_weighted` | 0,156 | 0,479 | **0,563** | 0,689 | 0,948 | 1,000 |
| `largest_comp_fill` | 0,066 | 0,534 | 0,637 | 0,876 | 1,000 | 1,000 |

Khoảng của `comp_fill_weighted`: `<0,2` 3; `0,2–0,5` 1 233; `0,5–0,85` 2 255; `0,85–0,95` 246; `≥0,95` 195 (5,0%).

**Theo loại thao tác:**

| Loại | n | median `comp_fill_weighted` | % mask có `comp_fill_weighted` ≥ 0,85 | % mask có `largest_comp_fill` > 0,85 |
|---|---:|---:|---:|---:|
| Insertion | 701 | 0,457 | 0,1% | 2,3% |
| Removal | 1 016 | 0,538 | 3,5% | 14,3% |
| Replacement | 627 | 0,583 | 1,1% | 5,1% |
| Copy-move | 758 | 0,614 | 16,8% | 41,8% |
| Splicing | 830 | 0,746 | 32,5% | 66,0% |

**Kiểm tra bằng mắt** (mẫu ngẫu nhiên có seed cố định: 3 ảnh mỗi loại, tô mask đỏ lên vùng quanh thành phần lớn nhất; đây là mẫu rất nhỏ, chỉ mang tính minh họa):
- Insertion, Replacement: mask bám sát nét chữ (kèm chút viền), không phải hộp.
- Removal: mask là vệt/khối bất quy tắc quanh chỗ chữ bị xóa và tô lại (inpainting).
- Splicing, Copy-move: lẫn cả vùng chữ bám nét lẫn vùng hình chữ nhật hoặc bo góc của thành phần UI (nút, nhãn, thanh).

**Kết luận:** không thể nói "STFD là vá hộp" hay "STFD là vá sát nét chữ" chung cho cả bộ. Ba loại Insertion/Replacement/Removal nghiêng về ôm nét/vùng xóa; Splicing và Copy-move có phần đáng kể (khoảng 1/3 đến 2/3 tùy chỉ số) là vùng dạng hộp. Với ngưỡng bạn nêu (≈1,0 = hộp, 0,2–0,5 = sát nét), các mask STFD nằm chủ yếu ở khoảng giữa 0,4–0,7 (mask bám nét chữ có giãn nhẹ, các nét liền nhau gộp thành cụm).

**Liên quan đến code hiện có:** heuristic `BoxPatch`/`TightContour` trong `forensichub_dataset.py` (solidity của thành phần lớn nhất > 0,85) nếu áp lên STFD sẽ gắn `BoxPatch` cho 1 058/3 932 ảnh (26,9%), riêng Splicing 66,0%. Heuristic này chỉ nhìn một thành phần, nên không phản ánh đúng mask nhiều thành phần của STFD. Nên dùng nhãn theo thư mục thay vì heuristic này.

### 2.4 Tỉ lệ diện tích vùng bị sửa (mục 4)

`mask_area_ratio` = (số pixel mask > 127) / (rộng × cao ảnh), toàn bộ 3 932 ảnh.

| min | p5 | median | mean | p95 | max |
|---:|---:|---:|---:|---:|---:|
| 0,0168% | 0,088% | **0,407%** | 1,083% | 4,00% | 37,48% |

| Khoảng | Số ảnh | Tỉ lệ |
|---|---:|---:|
| < 0,1% | 245 | 6,2% |
| 0,1–1% | 2 601 | 66,2% |
| 1–5% | 953 | 24,2% |
| > 5% | 133 | 3,4% |

Theo loại (`<0,1%` / `0,1–1%` / `1–5%` / `>5%`): Copy-move 22/343/347/46; Insertion 58/641/2/0; Removal 104/769/139/4; Replacement 57/566/4/0; Splicing 4/282/461/83. Splicing và Copy-move lớn hơn hẳn (median lần lượt 1,46% và 1,03%) so với Insertion/Removal/Replacement (khoảng 0,26–0,29%).

### 2.5 Mask (mục 5)

- Định dạng: PNG, 3 932/3 932. Chế độ lưu rất khác nhau: 1-bit 2 688, `L` 981, `RGB` 178, `P` 84, `I;16` 1. Loader phải chuyển về thang xám nhất quán.
- Giá trị: **3 929 mask chỉ có {0, 255}**, đúng như README ghi. Ba ngoại lệ có giá trị trung gian (nghi do lưu có mất mát hoặc khử răng cưa):
  - `3_Removal/tamper/8a4e8387df307512aaf478b9ae52bdc2.png` (mode L, giá trị 0,1,2,…)
  - `4_Insertion/tamper/ae95a78a6012c5fb7089277b4c05f6bf.png` (mode RGB, giá trị 0,8,20,23,…)
  - `5_Replacement/tamper/8cbadfde933e7c3b795235f7edd3e5a7.png` (mode L, giá trị 0,1,2,…)
- Một mask kiểu `I;16` (giá trị vẫn là 0/255): `3_Removal/tamper/c0d8681804f5cd8e574aaa2c54bd0f0a.png`.
- Kích thước mask khớp kích thước ảnh: **3 932/3 932**, 0 lệch.
- Mask rỗng: **0**. Mask phủ toàn ảnh: **0**. Số pixel mask được tính sau khi nhị phân hóa ở ngưỡng > 127, nên diện tích của 3 mask không nhị phân phụ thuộc ngưỡng đó.

### 2.6 Tập con biên lai chuyển tiền (mục 6)

**KHÔNG XÁC ĐỊNH ĐƯỢC.**
- README liệt kê 9 loại cảnh (Chat, Social Media, Mobile Payment, E-commerce, Online Banking, Maps & Transportation, Web Browsing, System Interfaces, Documents), nhưng **không có nhãn cảnh cho từng ảnh**: tên file là chuỗi hash, không có file CSV/JSON metadata trong zip hay trong repo GitHub.
- Mô tả "chat, biên lai chuyển tiền, trang tin" trong yêu cầu không khớp nguyên văn README (không có mục "trang tin" hay "biên lai" riêng).
- Trong khoảng 15 vùng cắt tôi đã xem bằng mắt, tôi thấy giao diện thương mại điện tử (giá, nút mua), mạng xã hội/trình duyệt. Đó chỉ là quan sát rời rạc, không dùng để ước lượng tỉ lệ.
- Để trả lời cần một trong ba: (a) nhãn cảnh từ tác giả; (b) OCR rồi lọc từ khóa (ảnh phần lớn là chữ Trung Quốc); (c) gán nhãn thủ công.

### 2.7 Siêu dữ liệu khác (mục 7)

**Theme.** Không có nhãn. Ước lượng theo cùng quy tắc của `forensichub_dataset.py` (độ sáng xám trung bình ≥ 128 là Light):

| | Light | Dark | % Dark |
|---|---:|---:|---:|
| Toàn bộ | 3 595 | 337 | 8,6% |
| Copy-move | 689 | 69 | 9,1% |
| Insertion | 606 | 95 | 13,6% |
| Removal | 980 | 36 | 3,5% |
| Replacement | 591 | 36 | 5,7% |
| Splicing | 729 | 101 | 12,2% |

Độ sáng trung bình: min 0,3; p5 72,9; median 221,2; p95 245,5; max 253,0. Có 141 ảnh nằm trong vùng mơ hồ 100–156. Ngưỡng này đo độ sáng ảnh, không chắc phản ánh chế độ tối của ứng dụng.

**Kích thước.** 15 độ phân giải khác nhau (chiều rộng 750–2 340, chiều cao 1 080–2 532). Dọc 3 838 ảnh, ngang 94 ảnh, không có ảnh vuông.

| Độ phân giải | Ảnh | | Độ phân giải | Ảnh |
|---|---:|---|---|---:|
| 1080×2340 | 2 665 | | 1125×2436 | 56 |
| 1080×2400 | 504 | | 1170×2532 | 53 |
| 750×1334 | 138 | | 1536×2048 | 34 |
| 1080×2376 | 100 | | 1668×2224 | 32 |
| 1080×2316 | 96 | | 2224×1668 | 9 |
| 1080×1920 | 91 | | 2048×1536 | 5 |
| 2340×1080 | 80 | | 1079×1976 | 1 |
| 828×1792 | 68 | | | |

**Ứng dụng nguồn.** Không có nhãn. README liệt kê 28 thiết bị và 4 hệ điều hành nhưng không gán cho từng ảnh; độ phân giải chỉ là chỉ báo yếu (nhiều thiết bị dùng chung độ phân giải).

**Chia train/val/test.** Không có: zip không chứa file chia, README không nhắc, repo GitHub không có danh sách chia (chỉ có `code/` và `imgs/`).

### 2.8 Giấy phép và trích dẫn (mục 8)

**Giấy phép: mâu thuẫn giữa các nguồn của chính tác giả.**
- Metadata Hugging Face (API và front matter của README): `license: cc-by-4.0`.
- Phần "License and Notice" trong README Hugging Face **và** trong `Readme.md` bên trong zip: "released for **academic research purposes only**"; đề nghị không phân phối lại ảnh nếu nghi có lộ thông tin.
- Repo GitHub `ZeqinYu/STFL-Net`: không có file LICENSE (API trả `license: null`).
- Cách đọc an toàn: dùng cho nghiên cứu học thuật, không phân phối lại. Nên hỏi tác giả (`kimjyu@foxmail.com`) nếu cần chắc chắn.

**Truy cập.** README Hugging Face yêu cầu gửi email từ địa chỉ học thuật để xin mật khẩu, nhưng README GitHub của tác giả **công bố mật khẩu công khai** (mục News 2026-03-06). Tôi đã dùng mật khẩu công khai này để đọc dữ liệu; nhóm tự quyết định có cần gửi email xin phép hay không.

**Bài báo gốc.** Yu, Li, Lin, Zeng, Zeng, "Learning to Locate the Text Forgery in Smartphone Screenshots", *ICASSP 2023 – IEEE International Conference on Acoustics, Speech and Signal Processing*, trang 1–5, IEEE, công bố 2023-06-04, DOI `10.1109/ICASSP49357.2023.10095070`. Đã xác nhận qua Crossref (tiêu đề, 5 tác giả, loại `proceedings-article`, nhà xuất bản IEEE). Liên kết IEEE Xplore trong README: `https://ieeexplore.ieee.org/abstract/document/10095070/`. Đây là bản chính thức đã qua bình duyệt; README Hugging Face/GitHub không dẫn arXiv, và tôi **chưa** tìm xem có bản arXiv hay không (không cần cho việc trích dẫn: nhóm trích dẫn bản IEEE). Tác giả yêu cầu trích dẫn bài báo này khi dùng STFD hoặc dữ liệu từ các cuộc thi Tianchi liên quan.

## 3. RỦI RO ảnh hưởng thiết kế thí nghiệm

1. **Không có ảnh sạch.** Không tính được FPR mức ảnh (cột `FP_image` trong `benchmark.py` sẽ toàn NaN) trên STFD. Muốn có FPR cần nguồn ảnh sạch bên ngoài và cùng miền (ảnh chụp màn hình thật), kèm rủi ro lệch miền so với STFD.
2. **Vùng sửa rất nhỏ.** Median 0,41% diện tích; 72% ảnh dưới 1%. Ảnh gốc 1080×2340 bị đưa về `INPUT_SIZE = (512, 512)` trong `benchmark.py`: co khoảng 2,1 lần theo chiều ngang và 4,6 lần theo chiều dọc, đồng thời làm biến dạng tỉ lệ. Với mask gồm nét chữ mảnh, việc co ảnh có thể làm mask mất hoặc nhòe; cần kiểm tra trước khi tin các chỉ số pixel (F1, IoU).
3. **"Hộp hay sát nét" phụ thuộc loại thao tác.** Không nên kết luận một câu cho cả bộ; nên phân tầng theo `tamper_type` (đã có nhãn). Heuristic `BoxPatch`/`TightContour` của repo không phù hợp mask nhiều thành phần của STFD (gắn Splicing 66% là `BoxPatch`).
4. **Lẫn giữa loại thao tác và nguồn thiết bị.** Removal có 30,5% ảnh ở 1080×2400 (310/1 016), trong khi Copy-move 6,9%, Insertion 5,8%, Replacement 9,3%, Splicing 5,2%. Dấu vết JPEG lại tập trung ở đúng vài độ phân giải (mục 2.2). Sai khác hiệu năng giữa các loại thao tác có thể là do nguồn/nén chứ không do thao tác.
5. **Cách phát biểu về "ảnh chưa nén".** STFD lưu PNG nhưng lịch sử nén không rõ, và có dấu vết JPEG gián tiếp ở khoảng 20–38% ảnh (chỉ báo chưa hiệu chuẩn). Không thể phát biểu "STFD là ảnh chưa từng nén".
6. **Giấy phép/truy cập mâu thuẫn** (mục 2.8). Bản zip 2,9 GB nằm trong `data/stfd/`; điều khoản cấm phân phối lại nên không được đẩy lên kho mã. Lưu ý: `.gitignore` của repo lưu ở mã hóa UTF-16 nên git **không đọc được** (đã kiểm tra, `data/` không bị ignore). Tôi đã thêm luật `data/` vào `.git/info/exclude` (chỉ áp dụng cục bộ). Nên chuyển `.gitignore` sang UTF-8.
7. **Theme Dark ít.** Chỉ 337 ảnh (8,6%); theo từng loại chỉ 36–101 ảnh, số liệu phân tầng theo theme sẽ không ổn định. Theme là ước lượng theo độ sáng, không phải nhãn.
8. **Vệ sinh dữ liệu.** 3 mask không nhị phân, 1 mask `I;16`, mask lưu ở 5 chế độ màu khác nhau, 86 ảnh RGBA (3 ảnh có alpha không đồng nhất). Loader phải chuyển RGB/nhị phân tường minh; nếu không, diện tích và kết quả có thể lệch âm thầm.
9. **Kết quả benchmark hiện có không phải của STFD.** `forensichub_benchmark_results.csv` ghi `dataset = DocTamper` (100 ảnh) và sinh từ các hàm mô hình giả (`np.random.rand`) trong `benchmark.py`; không được dùng làm số liệu STFD.
10. **Bản phát hành có thể khác dữ liệu trong bài báo.** Dataset được đưa lên Hugging Face vào 2026-03, còn bài báo năm 2023; số lượng trong bài báo chưa được đối chiếu với 3 932 cặp này.

## 4. Bất thường ngoài 8 câu hỏi

- README/`Readme.md` ghi định dạng "PNG / JPEG" nhưng chỉ có PNG.
- README ví dụ dùng tên file dạng MD5, nhưng **0/3 932** tên file bằng MD5 nội dung: tên không dùng để kiểm tra toàn vẹn được. Không có ảnh trùng byte (0 nhóm trùng), không trùng mask, không trùng id giữa các loại.
- Zip có thư mục rỗng lồng thừa: `STFD_ICASSP2023/STFD_ICASSP2023/1_Copy-move/masks/` (không chứa file).
- Ảnh `1_Copy-move/tamper/5991905ca6bb819bc4c5cff71010acea.png` có độ phân giải lẻ 1079×1976 (duy nhất, các ảnh khác là độ phân giải phổ biến). Chưa rõ nguyên nhân (có thể bị cắt/đổi kích thước).
- Điều khoản/giấy phép và cách xin mật khẩu không nhất quán (mục 2.8).

## 5. KHÔNG XÁC ĐỊNH ĐƯỢC

| Mục | Vì sao | Cần gì |
|---|---|---|
| 6 (tập biên lai) | Không có nhãn cảnh từng ảnh | Nhãn từ tác giả, hoặc OCR + từ khóa, hoặc gán nhãn thủ công |
| 7 (nhãn ứng dụng nguồn) | Không có nhãn | Nhãn từ tác giả |
| 7 (theme thật) | Chỉ ước lượng theo độ sáng | Nhãn từ tác giả, hoặc gán nhãn thủ công một mẫu để đo sai số của ngưỡng 128 |
| 2 (bảng lượng tử, số lần nén) | Toàn bộ là PNG, không lưu lịch sử nén | Thông tin xuất xứ từ tác giả; hoặc bộ ảnh đối chứng đã biết chắc lịch sử nén để hiệu chuẩn chỉ số `grid8_small` |
| 7 (chia train/val/test của bài báo) | Không có trong zip/README/GitHub | Toàn văn bài báo IEEE hoặc hỏi tác giả |
| 1 (ảnh sạch ở phiên bản khác) | Bản phát hành này không có; chưa kiểm tra bản Tianchi/ICDAR | Toàn văn bài báo hoặc hỏi tác giả |
| Số lượng trong bài báo so với 3 932 | Chưa đọc toàn văn bài báo | Toàn văn IEEE (qua tài khoản của trường) |
| Có bản arXiv không | Chưa tìm | Một lượt tra cứu arXiv (chỉ cần để biết, không trích dẫn) |

## 6. Code đã dùng

Toàn bộ nằm trong `audit_stfd.py` (đọc trực tiếp từ zip, không giải nén ra đĩa). Các đoạn chính:

```python
# Định dạng thật của ảnh (theo nội dung, không theo đuôi file)
def sniff(b):
    if b[:8] == b"\x89PNG\r\n\x1a\n": return "PNG"
    if b[:3] == b"\xff\xd8\xff":      return "JPEG"
    return "OTHER"

# Bảng lượng tử (chỉ tồn tại nếu là JPEG; STFD: 0 ảnh)
im = Image.open(io.BytesIO(tb));  q = im.quantization  # {0: [...64], 1: [...64]}

# Mask: diện tích, hộp bao, thành phần liên thông
mb_bin = (np.asarray(mask.convert("L")) > 127).astype(np.uint8)
area   = int(mb_bin.sum())
ys, xs = np.where(mb_bin)
fill   = area / ((ys.max()-ys.min()+1) * (xs.max()-xs.min()+1))      # mask_fill_ratio
n, _, st, _ = cv2.connectedComponentsWithStats(mb_bin, connectivity=8)
comp = st[1:]
fill_w = comp[:, cv2.CC_STAT_AREA].sum() / (comp[:, cv2.CC_STAT_WIDTH] * comp[:, cv2.CC_STAT_HEIGHT]).sum()

# Theme (cùng quy tắc forensichub_dataset.py)
theme = "Light" if np.asarray(im.convert("L")).mean() >= 128 else "Dark"
```

## 7. Cột trong `stfd_stats.csv`

11 cột theo yêu cầu, theo đúng thứ tự: `image_id, path, is_clean, img_format, width, height, mask_area_px, mask_area_ratio, mask_fill_ratio, theme_est, tamper_type`. Ghi chú:
- `path` tương đối so với gốc `STFD_ICASSP2023/` trong zip (ảnh ở `<path>`, mask ở cột phụ `mask_path`).
- `mask_fill_ratio` là mask / hộp bao **của toàn mask** (đúng yêu cầu) và ít có ý nghĩa vì mask nhiều thành phần; xem `comp_fill_weighted` và `largest_comp_fill` ở cột phụ.
- `theme_est` là ước lượng, không phải nhãn gốc. `is_clean` tính từ mask (diện tích = 0).
- Cột phụ: `mask_path, file_bytes, md5_content, mask_md5, name_is_md5_of_content, mask_format, img_mode, png_text_keys, jpeg_qtables, brightness, grid8_naive, grid8_small, n_colors_sub, mask_mode, mask_w, mask_h, mask_values, mask_n_unique, mask_size_match, mask_full, n_components, comp_fill_weighted, largest_comp_fill`.

## 8. Ghi chú về phương pháp

- Mọi con số trong tài liệu này đến từ một lần chạy `audit_stfd.py` trên toàn bộ 3 932 cặp, trừ: danh mục zip (đọc bằng HTTP Range), phép đối chứng chỉ số nén (20 ảnh ngẫu nhiên, seed 0) và kiểm tra bằng mắt (15 vùng cắt, seed cố định).
- Chỉ số nén đã sửa hai lỗi trong quá trình làm: chỉ số đầu thất bại đối chứng; lần chạy đầu dùng hai định nghĩa ảnh xám khác nhau giữa phần đối chứng và phần chính nên số liệu không so sánh được, đã sửa cho thống nhất rồi chạy lại. Các số nén trong tài liệu là của lần chạy cuối.

## 9. Tái tạo

```bash
pip install pyzipper pillow numpy pandas opencv-python
mkdir -p data/stfd
curl -L -o data/stfd/STFD_ICASSP2023.zip \
  https://huggingface.co/datasets/Zegkim/STFD/resolve/main/STFD_ICASSP2023.zip
python audit_stfd.py            # -> stfd_stats.csv, data/stfd_summary.json
```

Mật khẩu (công bố công khai trong README của `ZeqinYu/STFL-Net`) được nhúng trong `audit_stfd.py`. Mất khoảng 10 phút với 8 tiến trình (mặc định, chỉnh bằng `--procs`).
