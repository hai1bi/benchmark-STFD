"""
stfd_models.py
==============
Ba wrapper suy luận thật (TruFor, CAFTB-Net, ADCD-Net) với cùng giao diện:

    det = TruFor(...) / CAFTBNet(...) / ADCDNetDet(...)
    prob = det.predict(img_rgb_uint8)      # float32 [H, W] trong [0, 1], cùng kích thước ảnh vào; cao = giả mạo

Mã bên thứ ba nằm ở data/third_party/ (KHÔNG commit: TruFor chỉ dùng phi lợi nhuận). Trọng số ở data/weights/.
Không đảo cực bản đồ dự đoán ở bất kỳ mô hình nào.
"""
import contextlib
import io
import os
import sys
import time
import types

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
TP = os.path.join(HERE, "data", "third_party")
WT = os.path.join(HERE, "data", "weights")


@contextlib.contextmanager
def _pushd_path(path):
    """chdir + thêm vào sys.path trong khối with (mã bên thứ ba dùng đường dẫn tương đối)."""
    old = os.getcwd()
    os.chdir(path)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        os.chdir(old)
        if path in sys.path:
            sys.path.remove(path)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------- ghép ô cửa sổ trượt
def tile_predict(img, fn, tile=512, stride=256, batch=4):
    """
    Chạy `fn(list_tiles, list_coords) -> ndarray (B, tile, tile)` trên các ô tile x tile phủ ảnh
    (bước `stride`, ô cuối căn theo mép), trung bình vùng chồng. Ảnh nhỏ hơn ô được đệm phản chiếu.
    """
    H, W = img.shape[:2]
    ph, pw = max(0, tile - H), max(0, tile - W)
    ip = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT_101) if (ph or pw) else img
    Hp, Wp = ip.shape[:2]
    ys = list(range(0, Hp - tile + 1, stride))
    xs = list(range(0, Wp - tile + 1, stride))
    if ys[-1] != Hp - tile:
        ys.append(Hp - tile)
    if xs[-1] != Wp - tile:
        xs.append(Wp - tile)
    coords = [(y, x) for y in ys for x in xs]
    acc = np.zeros((Hp, Wp), np.float32)
    cnt = np.zeros((Hp, Wp), np.float32)
    for i in range(0, len(coords), batch):
        cb = coords[i:i + batch]
        tiles = [ip[y:y + tile, x:x + tile] for y, x in cb]
        out = fn(tiles, cb)
        for (y, x), o in zip(cb, out):
            acc[y:y + tile, x:x + tile] += o
            cnt[y:y + tile, x:x + tile] += 1
    return (acc / cnt)[:H, :W]


class Detector:
    name = "base"
    n_calls = 0

    def predict(self, img_rgb):
        raise NotImplementedError


# ---------------------------------------------------------------- TruFor
class TruFor(Detector):
    """
    Cả ảnh (không resize thêm; harness đã hạ về <=2 MP). Chế độ:
      'fp32'  : mặc định (đúng như test.py của tác giả)
      'fp16'  : autocast float16 (giảm VRAM)
      'tile'  : cửa sổ trượt tile x tile (giảm VRAM nhiều nhất; ghi rõ trong báo cáo nếu dùng)
    """
    name = "TruFor"

    def __init__(self, device, mode="fp32", tile=1024, stride=768):
        self.device, self.mode, self.tile, self.stride = device, mode, tile, stride
        root = os.path.join(TP, "TruFor", "TruFor_train_test")
        wfile = os.path.join(WT, "trufor", "weights", "trufor.pth.tar")
        with _pushd_path(root):
            from lib.config import config, update_config
            from lib.utils import get_model
            args = types.SimpleNamespace(experiment="trufor_ph3", gpu=0, opts=["TEST.MODEL_FILE", wfile])
            update_config(config, args)
            model = get_model(config)
            ck = torch.load(wfile, map_location="cpu", weights_only=False)
            model.load_state_dict(ck["state_dict"])
        self.model = model.to(device).eval()
        self.epoch = ck.get("epoch")

    @torch.no_grad()
    def _forward(self, rgb_uint8):
        x = torch.from_numpy(rgb_uint8.transpose(2, 0, 1).copy()).float().div(256.0).unsqueeze(0).to(self.device)
        if self.mode == "fp16":
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                pred, _, _, _ = self.model(x, save_np=False)
        else:
            pred, _, _, _ = self.model(x, save_np=False)
        return F.softmax(pred.float(), dim=1)[0, 1].cpu().numpy()

    def predict(self, img_rgb):
        if self.mode == "tile":
            return tile_predict(img_rgb, lambda ts, cs: np.stack([self._forward(t) for t in ts]),
                                tile=self.tile, stride=self.stride, batch=1)
        return self._forward(img_rgb)


# ---------------------------------------------------------------- CAFTB-Net
def _load_caftb_module():
    """Nạp caftb_net.py của ForensicHub mà không cài ForensicHub: dùng module giả cho registry/BaseModel."""
    import importlib.util
    fh = types.ModuleType("ForensicHub")
    reg = types.ModuleType("ForensicHub.registry")

    def register_model(x=None):
        if isinstance(x, type):
            return x
        return lambda c: c
    reg.register_model = register_model
    core = types.ModuleType("ForensicHub.core")
    bm = types.ModuleType("ForensicHub.core.base_model")
    bm.BaseModel = nn.Module
    for k, m in (("ForensicHub", fh), ("ForensicHub.registry", reg), ("ForensicHub.core", core),
                 ("ForensicHub.core.base_model", bm)):
        sys.modules[k] = m
    path = os.path.join(TP, "ForensicHub", "ForensicHub", "tasks", "document", "models", "caftb_net", "caftb_net.py")
    spec = importlib.util.spec_from_file_location("caftb_net_local", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CAFTBNet(Detector):
    """Cửa sổ trượt 512x512 ở độ phân giải gốc (đầu vào gốc của mô hình), chuẩn hóa ImageNet như DocTransform."""
    name = "CAFTB-Net"

    def __init__(self, device, ckpt, stride=256, batch=4, amp=False):
        import segmentation_models_pytorch as smp
        self.device, self.stride, self.batch, self.amp = device, stride, batch, amp
        mod = _load_caftb_module()
        orig = smp.from_pretrained
        # kiến trúc encoder SegFormer-B5 dựng từ config (không cần tải trọng số nền; checkpoint đầy đủ nạp sau)
        smp.from_pretrained = lambda *a, **k: smp.create_model(
            "Segformer", encoder_name="mit_b5", encoder_weights=None, classes=150, in_channels=3)
        try:
            model = mod.CAFTB_Net()
        finally:
            smp.from_pretrained = orig
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck.get("state_dict", ck)
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        miss, unexp = model.load_state_dict(sd, strict=False)
        self.load_report = {"missing": len(miss), "unexpected": len(unexp),
                            "missing_examples": list(miss)[:5], "unexpected_examples": list(unexp)[:5]}
        self.model = model.to(device).eval()
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    @torch.no_grad()
    def _run(self, tiles, coords):
        x = np.stack([((t.astype(np.float32) / 255.0) - self.mean) / self.std for t in tiles])
        x = torch.from_numpy(x.transpose(0, 3, 1, 2).copy()).to(self.device)
        m = torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3]), dtype=torch.long, device=self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
            out = self.model(x, m)["pred_mask"]
        return out.float()[:, 0].cpu().numpy()

    def predict(self, img_rgb):
        return tile_predict(img_rgb, self._run, tile=512, stride=self.stride, batch=self.batch)


# ---------------------------------------------------------------- ADCD-Net
class OCRMasker:
    """Mask vùng chữ bằng PaddleOCR TextDetection (PP-OCRv5_server_det), như seg_char.py của ADCD-Net."""

    def __init__(self):
        from paddleocr import TextDetection
        # enable_mkldnn=False: tránh lỗi oneDNN "ConvertPirAttribute2RuntimeAttribute" của Paddle 3.3 trên CPU Windows
        self.model = TextDetection(model_name="PP-OCRv5_server_det", enable_mkldnn=False)

    def __call__(self, img_rgb):
        bgr = np.ascontiguousarray(img_rgb[..., ::-1])
        res = self.model.predict(input=bgr, batch_size=1)[0]
        h, w = img_rgb.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        for poly, sc in zip(res["dt_polys"], res["dt_scores"]):
            if sc > 0.5:
                cv2.fillPoly(mask, [np.asarray(poly).astype(np.int32)], 1)
        return mask


class ADCDNetDet(Detector):
    """
    Cửa sổ trượt 512x512. Mỗi ô: ảnh xám nén JPEG q100 (giải mã lại thành RGB), hệ số DCT lượng tử đọc bằng
    jpeg_coeffs.decode_gray_jpeg (thay jpeg2dct), qt = bảng q100 (toàn 1), mask OCR cắt từ mask cả ảnh.
    Suy luận fp16 autocast như main.py của tác giả.
    """
    name = "ADCD-Net"

    def __init__(self, device, ocr=None, stride=256, batch=2):
        self.device, self.stride, self.batch = device, stride, batch
        root = os.path.join(TP, "ADCD-Net")
        wdir = os.path.join(WT, "adcd")
        with _pushd_path(root):
            import cfg
            cfg.docres_ckpt_path = os.path.join(wdir, "docres.pkl")
            cfg.ckpt = os.path.join(wdir, "ADCDNet.pth")
            from model.model import ADCDNet
            model = ADCDNet()
            ck = torch.load(cfg.ckpt, map_location="cpu", weights_only=True)
            sd = {k.replace("module.", ""): v for k, v in ck["model"].items()}
            miss, unexp = model.load_state_dict(sd, strict=False)
        self.load_report = {"missing": len(miss), "unexpected": len(unexp),
                            "missing_examples": list(miss)[:5], "unexpected_examples": list(unexp)[:5]}
        self.model = model.to(device).eval()
        self.ocr = ocr or OCRMasker()
        self.mean = np.array([0.485, 0.455, 0.406], np.float32)     # đúng giá trị trong ds.py (0.455)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)
        self.ocr_ms = 0.0
        self._ocr_mask = None

    @staticmethod
    def _jpeg_q100(tile):
        from jpeg_coeffs import blocks_to_plane, decode_gray_jpeg
        g = Image.fromarray(tile).convert("L")
        buf = io.BytesIO()
        g.save(buf, "JPEG", quality=100)
        raw = buf.getvalue()
        dct = blocks_to_plane(decode_gray_jpeg(raw))
        rgb = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
        return rgb, np.clip(np.abs(dct), 0, 20)

    @torch.no_grad()
    def _run(self, tiles, coords):
        imgs, dcts, ocrs = [], [], []
        for t, (y, x) in zip(tiles, coords):
            rgb, dct = self._jpeg_q100(t)
            imgs.append(((rgb.astype(np.float32) / 255.0) - self.mean) / self.std)
            dcts.append(dct.astype(np.int32))
            ocrs.append(self._ocr_mask_p[y:y + t.shape[0], x:x + t.shape[1]])
        img = torch.from_numpy(np.stack(imgs).transpose(0, 3, 1, 2).copy()).to(self.device)
        dct = torch.from_numpy(np.stack(dcts)).to(self.device)
        ocr = torch.from_numpy(np.stack(ocrs)[:, None].astype(np.int64)).to(self.device)
        mask = torch.zeros_like(ocr)
        qt = torch.ones((img.shape[0], 8, 8), dtype=torch.long, device=self.device)   # qts[100] (toàn 1)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            logits = self.model(img, dct, qt, mask, ocr, is_train=False)[0]
        return F.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()

    def predict(self, img_rgb):
        H, W = img_rgb.shape[:2]
        t0 = time.perf_counter()
        ocr = self.ocr(img_rgb)
        sync(self.device)
        self.ocr_ms = (time.perf_counter() - t0) * 1000
        ph, pw = max(0, 512 - H), max(0, 512 - W)
        self._ocr_mask_p = np.pad(ocr, ((0, ph), (0, pw))) if (ph or pw) else ocr
        return tile_predict(img_rgb, self._run, tile=512, stride=self.stride, batch=self.batch)
