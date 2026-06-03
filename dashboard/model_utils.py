"""
model_utils.py
==============
DenseNet169-based Brain Tumor Classifier — exact architecture from training script.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import DenseNet169_Weights

# ─── Constants ────────────────────────────────────────────────────────────────

CLASS_NAMES   = ['glioma', 'meningioma', 'pituitary', 'notumor']
CLASS_LABELS  = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES   = len(CLASS_NAMES)
IMAGE_SIZE    = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_DESCRIPTIONS = {
    'glioma': (
        "Glioma is a type of tumor that starts in the glial cells of the brain or spine. "
        "It accounts for about 33% of all brain tumors. Gliomas can be benign or malignant "
        "and are classified by grade (I–IV). Grade IV glioblastoma is the most aggressive form."
    ),
    'meningioma': (
        "Meningioma is a tumor that arises from the meninges — the three layers of membranes "
        "surrounding the brain and spinal cord. Most meningiomas are benign (non-cancerous) "
        "and slow-growing. They account for about 30% of all primary brain tumors."
    ),
    'pituitary': (
        "Pituitary tumors are abnormal growths that develop in the pituitary gland at the "
        "base of the brain. Most are benign adenomas. They can cause hormonal imbalances "
        "and vision problems due to their proximity to the optic nerves."
    ),
    'notumor': (
        "No tumor detected. The MRI scan shows normal brain tissue without any detectable "
        "abnormal mass or lesion that would be consistent with a brain tumor."
    ),
}

CLASS_COLORS = {
    'glioma':      '#EF4444',
    'meningioma':  '#F59E0B',
    'pituitary':   '#3B82F6',
    'notumor':     '#10B981',
}


# ─── Model ────────────────────────────────────────────────────────────────────

class BrainTumorClassifier(nn.Module):
    """DenseNet169 with custom head — mirrors the training architecture exactly."""

    def __init__(self, num_classes: int = 4, dropout_rate: float = 0.5):
        super().__init__()

        self.backbone = models.densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1)
        num_features  = self.backbone.classifier.in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_last_conv_layer(self) -> nn.Module:
        return self.backbone.features[-1]


# ─── Loader ───────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str | None = None) -> BrainTumorClassifier:
    """
    Load the classifier.  If *checkpoint_path* is None or the file doesn't
    exist the model is returned with random weights (useful for demo mode).
    """
    model = BrainTumorClassifier(num_classes=NUM_CLASSES)

    if checkpoint_path:
        try:
            state = torch.load(checkpoint_path, map_location=DEVICE)
            model.load_state_dict(state)
            print(f"[model_utils] Loaded weights from {checkpoint_path}")
        except FileNotFoundError:
            print(f"[model_utils] WARNING: {checkpoint_path!r} not found — using random weights.")
        except Exception as exc:
            print(f"[model_utils] WARNING: could not load weights — {exc}")

    model.to(DEVICE)
    model.eval()
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
