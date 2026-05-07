
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(
            out_ch, out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity, inplace=True)

        return out


def npr_2x2_w1(x: torch.Tensor) -> torch.Tensor:
    """
    NPR 2x2 w1 transform.
    Input shape: [B, C, H, W]
    """
    if x.ndim != 4:
        raise ValueError("Expected input shape [B, C, H, W]")

    B, C, H, W = x.shape
    H_even = H - (H % 2)
    W_even = W - (W % 2)

    x = x[:, :, :H_even, :W_even]
    out = torch.zeros_like(x)

    ref = x[:, :, 0::2, 0::2]

    out[:, :, 0::2, 0::2] = x[:, :, 0::2, 0::2] - ref
    out[:, :, 0::2, 1::2] = x[:, :, 0::2, 1::2] - ref
    out[:, :, 1::2, 0::2] = x[:, :, 1::2, 0::2] - ref
    out[:, :, 1::2, 1::2] = x[:, :, 1::2, 1::2] - ref

    return out


class NPRDetector(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.layer1 = nn.Sequential(
            BasicBlock(32, 32),
            BasicBlock(32, 32)
        )

        self.layer2 = nn.Sequential(
            BasicBlock(32, 64, stride=2),
            BasicBlock(64, 64)
        )

        self.layer3 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128)
        )

        self.layer4 = nn.Sequential(
            BasicBlock(128, 256, stride=2)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = npr_2x2_w1(x)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x


class NPRInference:
    def __init__(self, package_dir: str, device: str = None):
        self.package_dir = Path(package_dir)

        metadata_path = self.package_dir / "model_metadata.json"
        checkpoint_path = self.package_dir / "best_npr_detector.pt"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint file: {checkpoint_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.class_names = {
            int(k): v for k, v in self.metadata["class_names"].items()
        }

        input_cfg = self.metadata["input"]

        self.transform = transforms.Compose([
            transforms.Resize(tuple(input_cfg["resize"])),
            transforms.CenterCrop(tuple(input_cfg["center_crop"])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=input_cfg["normalization_mean"],
                std=input_cfg["normalization_std"]
            ),
        ])

        self.model = NPRDetector(num_classes=self.metadata["num_classes"])
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]

        prob_real = float(probs[0].detach().cpu())
        prob_fake = float(probs[1].detach().cpu())

        pred_label = int(torch.argmax(probs).detach().cpu())
        pred_name = self.class_names[pred_label]

        return {
            "pred_label": pred_label,
            "pred_name": pred_name,
            "prob_real": prob_real,
            "prob_fake": prob_fake
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--package_dir", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    detector = NPRInference(args.package_dir)
    result = detector.predict(args.image)
    print(json.dumps(result, indent=2))
