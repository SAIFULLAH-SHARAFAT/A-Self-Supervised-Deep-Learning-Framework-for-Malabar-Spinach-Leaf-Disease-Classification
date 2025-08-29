import io, os, time, base64, traceback
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models.resnet import Bottleneck

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles

# =========================
# Configuration
# =========================
DEFAULT_PATHS = [
    r"C:\Users\saify\OneDrive\Desktop\Python\finetuned_cbam_best.pth",
    r"C:\Users\saify\OneDrive\Desktop\Python\finetuned_cbam_checkpoint.pth",
]
MODEL_PATH = os.getenv("MODEL_PATH") or next((p for p in DEFAULT_PATHS if os.path.exists(p)), DEFAULT_PATHS[0])

IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fallback classes (overridden by checkpoint meta if present)
CLASS_NAMES: List[str] = [c.strip() for c in os.getenv(
    "CLASSES",
    "Alternaria Leaf Disease,Straw Mite Leaf Disease,Healthy"
).split(",")]

USE_TTA = os.getenv("USE_TTA", "false").lower() in ("1", "true", "yes")
MODEL_TEMP = float(os.getenv("MODEL_TEMP", "1.0"))
RETURN_ALL_PROBS = os.getenv("RETURN_ALL_PROBS", "true").lower() in ("1", "true", "yes")

# =========================
# Transforms (mirror eval)
# =========================
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])
visual_transform = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE))])

# Optional Grad-CAM
_CAM_AVAILABLE = False
try:
    from pytorch_grad_cam import LayerCAM, EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    _CAM_AVAILABLE = True
except Exception:
    _CAM_AVAILABLE = False


# =========================
# CBAM (Conv2d-style CA keys: ca.fc1 / ca.fc2 ; SA key: sa.conv)
# =========================
class ChannelAttentionConv(nn.Module):
    """
    Channel attention that matches checkpoint keys:
      ca.fc1.weight [r, C, 1, 1], ca.fc2.weight [C, r, 1, 1]
    """
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        hidden = max(1, in_planes // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, hidden, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, in_planes, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward_one(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def forward(self, x):
        avg_out = self.forward_one(self.avg_pool(x))
        max_out = self.forward_one(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttentionConv(nn.Module):
    """
    Spatial attention that matches checkpoint key:
      sa.conv.weight
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAMCompatBottleneck(Bottleneck):
    """
    ResNet bottleneck where CBAM modules live as attributes:
      self.ca (ChannelAttentionConv)
      self.sa (SpatialAttentionConv)
    giving keys:
      ... .ca.fc1.weight, .ca.fc2.weight, .sa.conv.weight
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        planes = self.conv3.out_channels
        self.ca = ChannelAttentionConv(planes, ratio=16)
        self.sa = SpatialAttentionConv(kernel_size=7)

    def forward(self, x):
        identity = x

        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)

        out = out * self.ca(out)
        out = out * self.sa(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


def _make_layer(block, inplanes, planes, blocks, stride=1, dilate=False, norm_layer=None):
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d
    downsample = None
    previous_dilation = 1
    if dilate:
        dilation = previous_dilation * stride
        stride = 1
    else:
        dilation = 1
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
            norm_layer(planes * block.expansion),
        )
    layers = [block(inplanes, planes, stride, downsample, groups=1, base_width=64, dilation=dilation, norm_layer=norm_layer)]
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(inplanes, planes, groups=1, base_width=64, dilation=dilation, norm_layer=norm_layer))
    return nn.Sequential(*layers), inplanes


# =========================
# Utilities to read checkpoint FIRST
# =========================
def _load_raw_ckpt(path: str) -> Dict[str, Any]:
    if not (path and os.path.exists(path)):
        raise FileNotFoundError(f"MODEL_PATH not found: {path}")
    raw = torch.load(path, map_location=DEVICE)
    if not isinstance(raw, dict):
        raise RuntimeError("Checkpoint must be a dict with a state dict.")
    return raw

def _extract_state_and_meta(raw: Dict[str, Any]) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    meta = {}
    if "model" in raw and isinstance(raw["model"], dict):
        state = raw["model"]
        meta = raw.get("meta", {})
    elif "state_dict" in raw and isinstance(raw["state_dict"], dict):
        state = raw["state_dict"]
        meta = raw.get("meta", {})
    else:
        state = {k: v for k, v in raw.items() if isinstance(v, torch.Tensor)}
        meta = raw.get("meta", {})
    # Normalize common prefixes
    norm = {}
    for k, v in state.items():
        k = k.replace("module.", "")
        if k.startswith("encoder."):
            k = "backbone." + k[len("encoder."):]
        norm[k] = v
    return norm, meta

def _infer_classifier_shape(state: Dict[str, torch.Tensor], in_features: int, num_classes_fallback: int) -> Tuple[str, Dict[str, int]]:
    """
    Returns a 'type' and shape info:
      - "seq2": classifier.0.weight [H, in], classifier.3.weight [C, H]
      - "linear": classifier.weight [C, in]
    """
    info = {}
    if "classifier.0.weight" in state and "classifier.3.weight" in state:
        h, in_w = state["classifier.0.weight"].shape
        c, h2 = state["classifier.3.weight"].shape
        info = {"hidden": int(h), "in": int(in_w), "classes": int(c)}
        return "seq2", info
    if "classifier.weight" in state:
        c, in_w = state["classifier.weight"].shape
        info = {"in": int(in_w), "classes": int(c)}
        return "linear", info
    info = {"in": in_features, "classes": num_classes_fallback}
    return "linear", info


# =========================
# Build a model COMPATIBLE with the checkpoint keys
# (stem registered INSIDE backbone as backbone.0/1/2/3)
# =========================
class CBAMResNet50Compat(nn.Module):
    def __init__(self, num_classes: int, classifier_type: str = "linear", hidden: Optional[int] = None):
        super().__init__()
        inplanes = 64
        norm = nn.BatchNorm2d

        # Stem and stages directly inside a big Sequential so keys are backbone.0/1/...
        conv1 = nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        bn1   = norm(inplanes)
        relu  = nn.ReLU(inplace=True)
        maxp  = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        l1, c1 = _make_layer(CBAMCompatBottleneck, inplanes, 64, 3, stride=1, norm_layer=norm)
        l2, c2 = _make_layer(CBAMCompatBottleneck, c1, 128, 4, stride=2, norm_layer=norm)
        l3, c3 = _make_layer(CBAMCompatBottleneck, c2, 256, 6, stride=2, norm_layer=norm)
        l4, c4 = _make_layer(CBAMCompatBottleneck, c3, 512, 3, stride=2, norm_layer=norm)

        # Assemble backbone as sequential ONLY (no top-level conv1/bn1 attrs), so sd keys match 'backbone.0/1/...'
        self.backbone = nn.Sequential(
            conv1, bn1, relu, maxp,  # -> backbone.0/1/2/3
            l1,                      # -> backbone.4
            l2,                      # -> backbone.5
            l3,                      # -> backbone.6
            l4,                      # -> backbone.7
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        in_features = 512 * Bottleneck.expansion  # 2048

        if classifier_type == "seq2":
            if hidden is None:
                hidden = 1024
            self.classifier = nn.Sequential(
                nn.Linear(in_features, hidden, bias=True),  # classifier.0
                nn.ReLU(inplace=True),                      # classifier.1
                nn.Dropout(p=0.0),                          # classifier.2
                nn.Linear(hidden, num_classes, bias=True),  # classifier.3
            )
        else:
            self.classifier = nn.Linear(in_features, num_classes, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# =========================
# Instantiate model based on CKPT
# =========================
raw_ckpt = _load_raw_ckpt(MODEL_PATH)
STATE, META = _extract_state_and_meta(raw_ckpt)

# Allow checkpoint to define classes
if isinstance(META, dict) and "classes" in META and isinstance(META["classes"], (list, tuple)):
    CLASS_NAMES = list(META["classes"])

# Infer classifier layout from checkpoint
in_features_nominal = 2048
clf_type, clf_info = _infer_classifier_shape(STATE, in_features_nominal, num_classes_fallback=len(CLASS_NAMES))
num_classes = clf_info.get("classes", len(CLASS_NAMES))
hidden = clf_info.get("hidden")

# Build a compatible model (keys now match backbone.* and ca/sa conv shapes)
model = CBAMResNet50Compat(num_classes=num_classes, classifier_type=clf_type, hidden=hidden).to(DEVICE)
model.eval()

# Strict load should now match:
MISSING_KEYS, UNEXPECTED_KEYS = [], []
try:
    res = model.load_state_dict(STATE, strict=True)
    MISSING_KEYS, UNEXPECTED_KEYS = list(res.missing_keys), list(res.unexpected_keys)
except Exception as e:
    print("[FATAL] Model load failed:", e)
    raise

# If checkpoint classes differ from fallback, sync labels
if num_classes != len(CLASS_NAMES):
    if not (isinstance(META, dict) and "classes" in META):
        CLASS_NAMES = [f"class_{i}" for i in range(num_classes)]

# =========================
# Inference helpers
# =========================
def _forward_logits(x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        logits = model(x)
        if MODEL_TEMP != 1.0:
            logits = logits / max(MODEL_TEMP, 1e-6)
    return logits

def _predict_single(x: torch.Tensor) -> np.ndarray:
    probs = torch.softmax(_forward_logits(x), dim=1)
    return probs[0].detach().cpu().numpy()

def _tta_predict(x: torch.Tensor) -> np.ndarray:
    views = [x, torch.flip(x, dims=[-1])] if USE_TTA else [x]
    prob_list = []
    with torch.no_grad():
        for v in views:
            prob_list.append(torch.softmax(_forward_logits(v), dim=1))
    logp = torch.log(torch.stack(prob_list, dim=0) + 1e-12).mean(dim=0)
    probs = torch.exp(logp)
    probs = probs / probs.sum(dim=1, keepdim=True)
    return probs[0].detach().cpu().numpy()

def _find_last_conv_layer(m: nn.Module):
    last = None
    for module in m.modules():
        if isinstance(module, nn.Conv2d):
            last = module
    return last

def _make_heatmap(pil_img: Image.Image, tensor: torch.Tensor, class_idx: int, method: str = "layercam"):
    if not _CAM_AVAILABLE:
        return None
    disp = np.array(visual_transform(pil_img)).astype(np.float32) / 255.0
    target_layer = _find_last_conv_layer(model)
    if target_layer is None:
        return None
    Expl = LayerCAM if method == "layercam" else EigenCAM
    cam = Expl(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(int(class_idx))]
    try:
        grayscale = cam(input_tensor=tensor, targets=targets, eigen_smooth=True)[0, :]
        heat = show_cam_on_image(disp, grayscale, use_rgb=True)
        buf = io.BytesIO(); Image.fromarray(heat).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print("[WARN] CAM generation failed:", e)
        print(traceback.format_exc())
        return None
    finally:
        try:
            cam.activations_and_grads.release()
        except Exception:
            pass


# =========================
# FastAPI app
# =========================
app = FastAPI(title="Malabar Spinach Disease Detector — CBAM-ResNet (Compat)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

@app.get("/favicon.ico")
def favicon():
    png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
    return Response(base64.b64decode(png_base64), media_type="image/png")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "img_size": IMG_SIZE,
        "model_path": MODEL_PATH,
        "classes": CLASS_NAMES,
        "resolved_arch": "cbam_resnet50_compat",
        "classifier_type": clf_type,
        "hidden_dim": hidden,
        "tta": USE_TTA,
        "temp": MODEL_TEMP,
        "missing_keys": MISSING_KEYS,
        "unexpected_keys": UNEXPECTED_KEYS,
        "grad_cam_available": _CAM_AVAILABLE,
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    explain: Optional[bool] = Form(False),
    cam_method: Optional[str] = Form(None),
    tta: Optional[bool] = Form(None),
):
    try:
        t0 = time.time()
        content = await file.read()
        img = Image.open(io.BytesIO(content))
        img = ImageOps.exif_transpose(img).convert("RGB")

        x = transform(img).unsqueeze(0).to(DEVICE)
        use_tta = USE_TTA if tta is None else bool(str(tta).lower() in ("1", "true", "yes"))

        probs = _tta_predict(x) if use_tta else _predict_single(x)
        idx = int(np.argmax(probs))
        dt = (time.time() - t0) * 1000.0

        resp = {
            "label": CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}",
            "probability": float(probs[idx]),
            "probs": (
                [{"class": (CLASS_NAMES[i] if i < len(CLASS_NAMES) else f'class_{i}'), "prob": float(probs[i])} for i in range(len(probs))]
                if RETURN_ALL_PROBS else None
            ),
            "inference_time_ms": round(dt, 2),
            "tta_used": use_tta,
        }

        if explain and _CAM_AVAILABLE:
            method = (cam_method or "layercam").lower()
            if method not in ("layercam", "eigencam"):
                method = "layercam"
            heatmap = _make_heatmap(img, x, idx, method=method)
            resp["heatmap"] = heatmap
            resp["cam_method"] = method
        else:
            resp["heatmap"] = None
            resp["cam_method"] = None

        return JSONResponse(resp)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Serve UI
APP_DIR = os.path.dirname(__file__)

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(APP_DIR, "index.html"))

app.mount("/static", StaticFiles(directory=APP_DIR), name="static")
