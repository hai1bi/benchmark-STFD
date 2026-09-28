"""
jpeg_coeffs.py
==============
Đọc hệ số DCT đã lượng tử hóa từ một file JPEG baseline xám (1 thành phần), thay cho `jpeg2dct`
(chỉ có bản mã nguồn, không có gói Windows). ADCD-Net dùng `jpeg2dct.numpy.load(path, normalized=False)`
trên ảnh xám nén JPEG q100; ảnh do Pillow ghi ra là baseline Huffman nên giải mã entropy trực tiếp cho
hệ số CHÍNH XÁC (không phải xấp xỉ).

decode_gray_jpeg(bytes) -> (H, W, 64) int32 theo thứ tự tự nhiên (hàng*8 + cột) của từng khối 8x8,
cùng quy ước với `jpeg2dct` (libjpeg lưu khối theo thứ tự tự nhiên). Kiểm tra: xem test_jpeg_coeffs().
"""
import io

import numpy as np
from numba import njit

ZIGZAG = np.array([
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63], dtype=np.int64)


def _build_huff(counts, symbols):
    """Bảng tra chuẩn tắc: trả (mincode, maxcode, valptr, symbols) theo độ dài 1..16."""
    maxcode = np.full(18, -1, dtype=np.int64)
    mincode = np.zeros(18, dtype=np.int64)
    valptr = np.zeros(18, dtype=np.int64)
    code, k = 0, 0
    for length in range(1, 17):
        valptr[length] = k
        mincode[length] = code
        code += counts[length - 1]
        k += counts[length - 1]
        maxcode[length] = code - 1 if counts[length - 1] else -1
        code <<= 1
    return mincode, maxcode, valptr, np.array(symbols, dtype=np.int64)


@njit(cache=True)
def _decode(data, n_blocks, dc_min, dc_max, dc_ptr, dc_sym, ac_min, ac_max, ac_ptr, ac_sym, restart, zz):
    out = np.zeros((n_blocks, 64), dtype=np.int32)
    pos = 0
    bitbuf = 0
    bitcnt = 0
    pred = 0
    n = data.shape[0]
    for b in range(n_blocks):
        if restart > 0 and b > 0 and b % restart == 0:
            # bỏ bit dư, đọc marker RSTn (đã được loại ở bước tách), đặt lại DC
            bitbuf = 0
            bitcnt = 0
            pred = 0
        # ---- DC ----
        code = 0
        length = 0
        sym = -1
        while length < 16:
            if bitcnt == 0:
                if pos >= n:
                    return out
                bitbuf = data[pos]
                pos += 1
                bitcnt = 8
            bitcnt -= 1
            code = (code << 1) | ((bitbuf >> bitcnt) & 1)
            length += 1
            if dc_max[length] >= 0 and code <= dc_max[length] and code >= dc_min[length]:
                sym = dc_sym[dc_ptr[length] + code - dc_min[length]]
                break
        s = sym
        diff = 0
        if s > 0:
            v = 0
            for _ in range(s):
                if bitcnt == 0:
                    if pos >= n:
                        return out
                    bitbuf = data[pos]
                    pos += 1
                    bitcnt = 8
                bitcnt -= 1
                v = (v << 1) | ((bitbuf >> bitcnt) & 1)
            diff = v if v >= (1 << (s - 1)) else v - (1 << s) + 1
        pred += diff
        out[b, 0] = pred
        # ---- AC ----
        k = 1
        while k < 64:
            code = 0
            length = 0
            sym = -1
            while length < 16:
                if bitcnt == 0:
                    if pos >= n:
                        return out
                    bitbuf = data[pos]
                    pos += 1
                    bitcnt = 8
                bitcnt -= 1
                code = (code << 1) | ((bitbuf >> bitcnt) & 1)
                length += 1
                if ac_max[length] >= 0 and code <= ac_max[length] and code >= ac_min[length]:
                    sym = ac_sym[ac_ptr[length] + code - ac_min[length]]
                    break
            r = sym >> 4
            s = sym & 15
            if s == 0:
                if r == 15:
                    k += 16
                    continue
                break
            k += r
            v = 0
            for _ in range(s):
                if bitcnt == 0:
                    if pos >= n:
                        return out
                    bitbuf = data[pos]
                    pos += 1
                    bitcnt = 8
                bitcnt -= 1
                v = (v << 1) | ((bitbuf >> bitcnt) & 1)
            val = v if v >= (1 << (s - 1)) else v - (1 << s) + 1
            if k < 64:
                out[b, zz[k]] = val
            k += 1
    return out


def decode_gray_jpeg(buf: bytes):
    """Giải mã hệ số DCT (đã lượng tử) của JPEG baseline 1 thành phần. Trả (rows, cols, 64) int32 thứ tự tự nhiên."""
    assert buf[:2] == b"\xff\xd8", "không phải JPEG"
    p = 2
    dc_tabs, ac_tabs = {}, {}
    width = height = None
    restart = 0
    ncomp = None
    while p < len(buf):
        assert buf[p] == 0xFF, "marker hỏng"
        while buf[p + 1] == 0xFF:
            p += 1
        m = buf[p + 1]
        p += 2
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
            continue
        if m == 0xD9:
            break
        ln = int.from_bytes(buf[p:p + 2], "big")
        seg = buf[p + 2:p + ln]
        if m == 0xC0:                                   # SOF0 baseline
            height = int.from_bytes(seg[1:3], "big")
            width = int.from_bytes(seg[3:5], "big")
            ncomp = seg[5]
        elif m in (0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            raise NotImplementedError("JPEG không phải baseline")
        elif m == 0xC4:                                 # DHT
            q = 0
            while q < len(seg):
                tc_th = seg[q]
                counts = list(seg[q + 1:q + 17])
                total = sum(counts)
                syms = list(seg[q + 17:q + 17 + total])
                tab = _build_huff(counts, syms)
                (dc_tabs if (tc_th >> 4) == 0 else ac_tabs)[tc_th & 15] = tab
                q += 17 + total
        elif m == 0xDD:
            restart = int.from_bytes(seg[0:2], "big")
        elif m == 0xDA:                                 # SOS: dữ liệu entropy theo sau
            assert ncomp == 1, "chỉ hỗ trợ ảnh xám 1 thành phần"
            tdc_tac = seg[2]
            dc = dc_tabs[tdc_tac >> 4]
            ac = ac_tabs[tdc_tac & 15]
            q = p + ln
            raw = buf[q:]
            # tách phần entropy tới marker thật (FF khác 00/RSTn), bỏ nhồi byte FF00
            end = len(raw)
            i = 0
            out = bytearray()
            while i < len(raw):
                c = raw[i]
                if c == 0xFF:
                    nx = raw[i + 1] if i + 1 < len(raw) else 0
                    if nx == 0x00:
                        out.append(0xFF)
                        i += 2
                        continue
                    if 0xD0 <= nx <= 0xD7:
                        i += 2                           # RSTn: bỏ qua, hàm giải mã đặt lại tại ranh giới
                        continue
                    end = i
                    break
                out.append(c)
                i += 1
            data = np.frombuffer(bytes(out), dtype=np.uint8).astype(np.int64)
            rows, cols = (height + 7) // 8, (width + 7) // 8
            arr = _decode(data, rows * cols, dc[0], dc[1], dc[2], dc[3], ac[0], ac[1], ac[2], ac[3],
                          restart, ZIGZAG)
            return arr.reshape(rows, cols, 64)
        p += ln
    raise ValueError("không thấy SOS")


def blocks_to_plane(coefs):
    """(rows, cols, 64) -> mảng (8*rows, 8*cols) như `multi_jpeg` của ADCD-Net."""
    rows, cols, _ = coefs.shape
    return coefs.reshape(rows, cols, 8, 8).transpose(0, 2, 1, 3).reshape(rows * 8, cols * 8)


def test_jpeg_coeffs(n=5, seed=0):
    """Kiểm tra: FDCT(giải mã bằng Pillow) khớp hệ số đọc từ file JPEG q100 tới sai số làm tròn (|diff| <= 1)."""
    import scipy.fft
    from PIL import Image
    rng = np.random.default_rng(seed)
    worst = 0
    for _ in range(n):
        h, w = int(rng.integers(40, 200)), int(rng.integers(40, 200))
        base = rng.integers(0, 256, (h, w)).astype(np.float32)
        base = np.clip(np.kron(rng.integers(0, 256, (h // 8 + 1, w // 8 + 1)), np.ones((8, 8)))[:h, :w] * 0.7 +
                       base * 0.3, 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(base, "L").save(buf, "JPEG", quality=100)
        coefs = decode_gray_jpeg(buf.getvalue())
        dec = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("L")).astype(np.float64) - 128
        H, W = coefs.shape[0] * 8, coefs.shape[1] * 8
        pad = np.pad(dec, ((0, H - h), (0, W - w)), mode="edge")
        blocks = pad.reshape(H // 8, 8, W // 8, 8).transpose(0, 2, 1, 3)
        ref = scipy.fft.dctn(blocks, type=2, norm="ortho", axes=(2, 3)).reshape(H // 8, W // 8, 64)
        inner = (slice(0, h // 8), slice(0, w // 8))       # bỏ khối biên (Pillow đệm ảnh khác 'edge')
        worst = max(worst, float(np.abs(coefs[inner] - ref[inner]).max()))
    return worst


if __name__ == "__main__":
    print("max |hệ số đọc - FDCT giải mã| trên khối nội bộ:", test_jpeg_coeffs())
