# app.py — FastAPI server for SimSiam CBAM finetunes (ResNet50-style), with CAMs and a web UI
import io, os, time, base64, traceback
from typing import Optional, List, Tuple, Dict
import numpy as np
from PIL import Image, ImageOps
import torch, torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models.resnet import Bottleneck

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles

# ===========================
# Config (edit here or override via env)
# ===========================
MODEL_PATH = os.getenv("MODEL_PATH", r"C:\Users\saify\OneDrive\Desktop\Python\finetuned_cbam_best.pth")
IMG_SIZE   = int(os.getenv("IMG_SIZE", "224"))
CLASS_NAMES = [c.strip() for c in os.getenv("CLASSES", "Alternaria Leaf Disease,Straw Mite Leaf Disease,Healthy").split(",")]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_TTA    = os.getenv("USE_TTA", "true").lower() in ("1","true","yes")
MODEL_TEMP = float(os.getenv("MODEL_TEMP", "1.0"))
DEFAULT_CAM_METHOD = os.getenv("DEFAULT_CAM_METHOD", "layercam")
RETURN_ALL_PROBS = os.getenv("RETURN_ALL_PROBS", "true").lower() in ("1","true","yes")

# ---------------- Transforms ----------------
base_resize = T.Resize((IMG_SIZE, IMG_SIZE))
to_tensor = T.ToTensor()
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = T.Compose([base_resize, to_tensor, normalize])
visual_transform = T.Compose([base_resize])

# ---------------- CAM availability ----------------
_CAM_AVAILABLE = False
try:
    from pytorch_grad_cam import LayerCAM, EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    _CAM_AVAILABLE = True
except Exception:
    _CAM_AVAILABLE = False

# ---------------- CBAM Modules ----------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
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

class CBAM(nn.Module):
    def __init__(self, planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class CBAMBottleneck(Bottleneck):
    """ResNet50-style bottleneck with CBAM after conv3."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cbam = CBAM(self.conv3.out_channels)
    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)
        out = self.cbam(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# ---------------- SimSiam-style Backbone Model ----------------
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

    layers = []
    layers.append(block(inplanes, planes, stride, downsample, groups=1, base_width=64, dilation=dilation, norm_layer=norm_layer))
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(inplanes, planes, groups=1, base_width=64, dilation=dilation, norm_layer=norm_layer))
    return nn.Sequential(*layers), inplanes

class SimSiamCBAMResNet50(nn.Module):
    """
    A ResNet50-like backbone wrapped in nn.Sequential under 'backbone' to match checkpoints that use keys like:
      backbone.0.weight, backbone.1.weight, backbone.4.0.conv1.weight, ...
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.inplanes = 64
        norm = nn.BatchNorm2d
        conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        bn1   = norm(self.inplanes)
        relu  = nn.ReLU(inplace=True)
        maxp  = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Layers: 3,4,6,3 bottleneck blocks
        l1, c1 = _make_layer(CBAMBottleneck, self.inplanes, 64,  3, stride=1, norm_layer=norm)
        l2, c2 = _make_layer(CBAMBottleneck, c1,           128, 4, stride=2, norm_layer=norm)
        l3, c3 = _make_layer(CBAMBottleneck, c2,           256, 6, stride=2, norm_layer=norm)
        l4, c4 = _make_layer(CBAMBottleneck, c3,           512, 3, stride=2, norm_layer=norm)
        # This exact ordering gives indices matching many SimSiam 'backbone.*' dumps:
        self.backbone = nn.Sequential(
            conv1,  # 0
            bn1,    # 1
            relu,   # 2
            maxp,   # 3
            l1,     # 4
            l2,     # 5
            l3,     # 6
            l4,     # 7
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ---------------- Checkpoint normalization & remapping ----------------
def _load_ckpt(path: str):
    if not (path and os.path.exists(path)):
        print(f"[WARN] MODEL_PATH not found: {path}")
        return None
    ckpt = torch.load(path, map_location=DEVICE)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]
    if isinstance(ckpt, dict):
        new = {}
        for k, v in ckpt.items():
            k = k.replace("module.", "")
            # Map common SimSiam dumps: 'encoder' or 'backbone' -> 'backbone'
            if k.startswith("encoder."):
                k = "backbone." + k[len("encoder."):]
            new[k] = v
        return new
    return ckpt

def _strip_to_backbone(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not isinstance(sd, dict): return {}
    return {k: v for k, v in sd.items() if k.startswith("backbone.")}

def _remap_simsiam_cbam(sd):
    """Map SimSiam CBAM keys to our CBAM conv keys and reshape Linear -> Conv1x1 weights.
       - backbone.*.ca.fc1.weight -> backbone.*.cbam.ca.mlp.0.weight  (reshape [H,C] -> [H,C,1,1])
       - backbone.*.ca.fc2.weight -> backbone.*.cbam.ca.mlp.2.weight  (reshape [C,H] -> [C,H,1,1])
       - backbone.*.sa.conv.weight -> backbone.*.cbam.sa.conv.weight  (rename only)
       Drops fc biases (.ca.fc1.bias / .ca.fc2.bias) because convs are bias=False.
    """
    if not isinstance(sd, dict):
        return sd
    remapped = {}
    for k, v in sd.items():
        nk = k
        tensor = v
        if ".ca.fc1.weight" in k:
            nk = k.replace(".ca.fc1.weight", ".cbam.ca.mlp.0.weight")
            if tensor.ndim == 2:
                H, C = tensor.shape
                tensor = tensor.view(H, C, 1, 1)
        elif ".ca.fc2.weight" in k:
            nk = k.replace(".ca.fc2.weight", ".cbam.ca.mlp.2.weight")
            if tensor.ndim == 2:
                C, H = tensor.shape
                tensor = tensor.view(C, H, 1, 1)
        elif ".sa.conv.weight" in k and ".cbam.sa.conv.weight" not in k:
            nk = k.replace(".sa.conv.weight", ".cbam.sa.conv.weight")
        if ".ca.fc1.bias" in k or ".ca.fc2.bias" in k:
            continue
        remapped[nk] = tensor
    return remapped

def _remap_classifier_head(sd, model):
    """Try to map common head names to classifier.* if present in checkpoint.
       Supports: fc.*, head.*, linear.*  -> classifier.*
       Only maps when shapes match.
    """
    if not isinstance(sd, dict):
        return sd, False
    want_w = getattr(model.classifier, "weight", None)
    want_b = getattr(model.classifier, "bias", None)
    if want_w is None:
        return sd, False
    want_w_shape = tuple(want_w.shape)
    want_b_shape = tuple(want_b.shape) if want_b is not None else None

    for pref in ("classifier.", "fc.", "head.", "linear."):
        w = sd.get(pref + "weight", None)
        b = sd.get(pref + "bias", None)
        if w is not None and tuple(w.shape) == want_w_shape:
            sd["classifier.weight"] = w
            if b is not None and want_b_shape is not None and tuple(b.shape) == want_b_shape:
                sd["classifier.bias"] = b
            return sd, True
    return sd, False

# ---------------- Build & Load ----------------
model = SimSiamCBAMResNet50(num_classes=len(CLASS_NAMES)).to(DEVICE).eval()
MISSING_KEYS: List[str] = []
UNEXPECTED_KEYS: List[str] = []
RESOLVED_ARCH = "simsiam_cbam_resnet50"

CKPT = _load_ckpt(MODEL_PATH)
if isinstance(CKPT, dict):
    sd = _strip_to_backbone(CKPT)
    sd = _remap_simsiam_cbam(sd)
    sd, _mapped_cls = _remap_classifier_head(sd, model)
    res = model.load_state_dict(sd, strict=False)
    MISSING_KEYS = list(res.missing_keys)
    UNEXPECTED_KEYS = list(res.unexpected_keys)
    print(f"[INFO] Loaded checkpoint: {MODEL_PATH}")
    if MISSING_KEYS:
        print("[INFO] Missing keys:", MISSING_KEYS[:10], "..." if len(MISSING_KEYS) > 10 else "")
    if UNEXPECTED_KEYS:
        print("[INFO] Unexpected keys:", UNEXPECTED_KEYS[:10], "..." if len(UNEXPECTED_KEYS) > 10 else "")
else:
    print("[WARN] No compatible state_dict found; using randomly initialized CBAM-ResNet50 head.")

# ---------------- Pred / CAM helpers ----------------
def _forward_logits(x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        logits = model(x)
        if MODEL_TEMP != 1.0:
            logits = logits / max(MODEL_TEMP, 1e-6)
    return logits

def _predict_single(x: torch.Tensor) -> np.ndarray:
    logits = _forward_logits(x)
    probs = torch.softmax(logits, dim=1)
    return probs[0].detach().cpu().numpy()

def _tta_predict(x: torch.Tensor) -> np.ndarray:
    views = [x, torch.flip(x, dims=[-1])] if USE_TTA else [x]
    prob_list = []
    with torch.no_grad():
        for v in views:
            logits = _forward_logits(v)
            prob_list.append(torch.softmax(logits, dim=1))
    logp = torch.log(torch.stack(prob_list, dim=0) + 1e-12).mean(dim=0)
    probs = torch.exp(logp)
    probs = probs / probs.sum(dim=1, keepdim=True)
    return probs[0].detach().cpu().numpy()

def find_last_conv_layer(m: nn.Module):
    last = None
    for module in m.modules():
        if isinstance(module, nn.Conv2d):
            last = module
    return last

def make_heatmap(pil_img: Image.Image, tensor: torch.Tensor, class_idx: int, method: str = "layercam"):
    if not _CAM_AVAILABLE:
        return None

    # Normalize image for visualization
    disp = np.array(visual_transform(pil_img)).astype(np.float32) / 255.0

    # Locate last conv layer
    target_layer = find_last_conv_layer(model)
    if target_layer is None:
        return None

    # Choose CAM method
    Explainer = LayerCAM if method == "layercam" else EigenCAM
    cam = Explainer(model=model, target_layers=[target_layer])

    # Move model and tensor to device
    model.to(DEVICE)
    tensor = tensor.to(DEVICE)

    targets = [ClassifierOutputTarget(int(class_idx))]

    try:
        # Disable AMP for deterministic behavior
        with torch.cuda.amp.autocast(enabled=False):
            grayscale = cam(input_tensor=tensor, targets=targets, eigen_smooth=True)[0, :]
        heat = show_cam_on_image(disp, grayscale, use_rgb=True)

        # Encode heatmap to base64
        buf = io.BytesIO()
        Image.fromarray(heat).save(buf, format="PNG")
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

# --- Domain analysis content ---
ALT_TEXT = ("Alternaria infection in spinach initially presents as small, circular spots with distinct concentric rings. "
            "Over time, these lesions become irregular in shape. The circular spots are characterized by dark black margins "
            "surrounding a necrotic center.")
STRAW_TEXT = ("Straw mites are pests that infest Malabar spinach, often leading to substantial crop losses. "
              "Due to their tendency to reside deep within plant tissues, direct visual identification is challenging. "
              "Instead, their presence is typically inferred through symptoms such as leaf spotting and a general decline in plant vigor.")
HEALTHY_TEXT = "Congratulations — the spinach appears to be in excellent health."
STRAW_PREV = [
    ("Sanitation", "Clean plant debris, tools, and surroundings"),
    ("Monitoring", "Weekly inspections for early signs"),
    ("Biocontrol", "Predatory mites"),
]
ALTERNARIA_FUNG = [
    ("Azoxystrobin", "11", "Systemic; broad-spectrum"),
    ("Chlorothalonil", "M5", "Contact protectant"),
    ("Mancozeb", "M3", "Broad-spectrum; protectant"),
    ("Difenoconazole", "3", "Systemic; use with caution to avoid resistance"),
    ("Copper-based fungicides", "M1", "Organic-compatible; protectant only"),
]

# ---------------- FastAPI app ----------------
app = FastAPI(title="Malabar Spinach Disease Detector — SimSiam CBAM")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

@app.get("/favicon.ico")
def favicon():
    # 1x1 png
    png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
    return Response(base64.b64decode(png_base64), media_type="image/png")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "classes": CLASS_NAMES,
        "img_size": IMG_SIZE,
        # keep both names for backward/forward compatibility
        "model_path": MODEL_PATH,
        "model": MODEL_PATH,
        "resolved_arch": RESOLVED_ARCH,
        "arch": RESOLVED_ARCH,
        "tta": USE_TTA,
        "temp": MODEL_TEMP,
        "default_cam_method": DEFAULT_CAM_METHOD,
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
        img = Image.open(io.BytesIO(content)); img = ImageOps.exif_transpose(img).convert("RGB")
        x = transform(img).unsqueeze(0).to(DEVICE)

        use_tta = USE_TTA if tta is None else bool(str(tta).lower() in ("1", "true", "yes"))
        probs = _tta_predict(x) if use_tta else _predict_single(x)
        idx = int(np.argmax(probs))
        dt = (time.time() - t0) * 1000.0

        # domain analysis object
        domain_info = {"summary": None, "straw_prevention": None, "alternaria_fungicides": None}
        label_lower = CLASS_NAMES[idx].lower()
        if label_lower.startswith("alternaria"):
            domain_info["summary"] = ALT_TEXT
            domain_info["alternaria_fungicides"] = ALTERNARIA_FUNG
        elif "straw" in label_lower:
            domain_info["summary"] = STRAW_TEXT
            domain_info["straw_prevention"] = STRAW_PREV
        else:
            domain_info["summary"] = HEALTHY_TEXT

        resp = {
            "label": CLASS_NAMES[idx],
            "probability": float(probs[idx]),
            "probs": (
                [{"class": CLASS_NAMES[i], "prob": float(probs[i])} for i in range(len(CLASS_NAMES))]
                if RETURN_ALL_PROBS else None
            ),
            "inference_time_ms": round(dt, 2),
            "tta_used": use_tta,
            "domain": domain_info,
        }

        if explain and _CAM_AVAILABLE:
            method = (cam_method or DEFAULT_CAM_METHOD).lower()
            if method not in ("layercam", "eigencam"):
                method = "layercam"
            heatmap = make_heatmap(img, x, idx, method=method)
            resp["heatmap"] = heatmap
            resp["cam_method"] = method
        else:
            resp["heatmap"] = None
            resp["cam_method"] = None

        return JSONResponse(resp)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- Serve frontend: explicit index at "/", static under /static ---
APP_DIR = os.path.dirname(__file__)

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(APP_DIR, "index.html"))

app.mount("/static", StaticFiles(directory=APP_DIR), name="static")
