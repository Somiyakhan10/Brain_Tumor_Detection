"""
gradcam_utils.py
================
Grad-CAM implementation — mirrors the training notebook exactly.
Returns NumPy arrays suitable for Streamlit / Plotly display.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

from model_utils import (
    IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE,
    CLASS_NAMES, DEVICE, BrainTumorClassifier,
)


# ─── Preprocessing ────────────────────────────────────────────────────────────

_inference_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    """Return a (1, 3, H, W) tensor ready for the model."""
    tensor = _inference_transform(pil_image.convert('RGB'))
    return tensor.unsqueeze(0).to(DEVICE)


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised (3, H, W) tensor → uint8 (H, W, 3) array."""
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * np.array(IMAGENET_STD)) + np.array(IMAGENET_MEAN)
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


# ─── Grad-CAM ─────────────────────────────────────────────────────────────────

class GradCAM:
    """Exact port of the training-notebook GradCAM class."""

    def __init__(self, model: BrainTumorClassifier, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self.activations  = None
        self.gradients    = None

        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, image: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.eval()
        image = image.clone().requires_grad_(True)

        output = self.model(image)
        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1)[0]
        cam     = torch.relu(cam)

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)

        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = torch.nn.functional.interpolate(
            cam, size=(IMAGE_SIZE, IMAGE_SIZE),
            mode='bilinear', align_corners=False,
        )
        return cam[0, 0].cpu().detach().numpy()


# ─── 3-Panel Result ───────────────────────────────────────────────────────────

def generate_gradcam_panels(
    model: BrainTumorClassifier,
    pil_image: Image.Image,
) -> dict:
    """
    Run inference + Grad-CAM on *pil_image*.

    Returns a dict with:
        original_rgb   : np.ndarray (H, W, 3)  uint8
        heatmap_rgb    : np.ndarray (H, W, 3)  uint8  jet colourmap
        overlay_rgb    : np.ndarray (H, W, 3)  uint8  60 % orig + 40 % heat
        pred_class     : int
        pred_label     : str
        confidence     : float  (0-1)
        probabilities  : list[float]   length == NUM_CLASSES
        cam            : np.ndarray (H, W)  float32  0-1
    """
    input_tensor = preprocess_image(pil_image)       # (1,3,H,W)

    # ── Inference ──
    with torch.no_grad():
        output = model(input_tensor)
        probs  = torch.softmax(output, dim=1)[0].cpu().numpy()

    pred_class  = int(probs.argmax())
    confidence  = float(probs[pred_class])
    pred_label  = CLASS_NAMES[pred_class]

    # ── Grad-CAM ──
    target_layer = model.get_last_conv_layer()
    gradcam      = GradCAM(model, target_layer)
    cam          = gradcam.generate_cam(input_tensor, pred_class)

    # ── Build panels ──
    original_rgb = denormalize_image(input_tensor[0])

    heatmap_bgr  = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_rgb  = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay_rgb  = np.clip(
        heatmap_rgb * 0.4 + original_rgb * 0.6, 0, 255
    ).astype(np.uint8)

    return {
        'original_rgb':  original_rgb,
        'heatmap_rgb':   heatmap_rgb,
        'overlay_rgb':   overlay_rgb,
        'cam':           cam,
        'pred_class':    pred_class,
        'pred_label':    pred_label,
        'confidence':    confidence,
        'probabilities': probs.tolist(),
    }
