"""
ResNet50 pretrenowany na Sentinel-2 (SSL4EO-S12, 3M scen z Europy).
Znacznie lepszy punkt startowy niż ImageNet dla danych EO.

torchgeo docs: https://torchgeo.readthedocs.io/en/stable/api/models.html
"""
import os
import torch
import torch.nn as nn
import timm
from torchgeo.models import ResNet50_Weights

NUM_CLASSES  = 10
IN_CHANNELS  = 4   # R, G, B, NIR (symulowany)
PATCH_SIZE   = 128  # zwiększamy z 64 → 128 dla lepszej dokładności


def build_geo_model(pretrained: bool = True) -> nn.Module:
    """
    ResNet50 z wagami SSL4EO-S12 (Sentinel-2, MoCo v3).
    Wagi pretrenowane na 3M scenach Sentinel-2 z całego świata.
    """
    if pretrained:
        weights = ResNet50_Weights.SENTINEL2_ALL_MOCO
        # Model z 13 kanałami wejściowymi (wszystkie pasma Sentinel-2)
        model = timm.create_model(
            "resnet50",
            in_chans=IN_CHANNELS,
            num_classes=NUM_CLASSES,
        )
        # Załaduj wagi SSL4EO — strict=False bo mamy 4 kanały zamiast 13
        state = weights.get_state_dict(progress=True)
        # Usuń wagi pierwszej warstwy conv (niezgodny rozmiar kanałów)
        state = {k: v for k, v in state.items() if "conv1" not in k}
        model.load_state_dict(state, strict=False)
        print("Załadowano wagi SSL4EO-S12 (Sentinel-2 MoCo)")
    else:
        model = timm.create_model(
            "resnet50",
            in_chans=IN_CHANNELS,
            num_classes=NUM_CLASSES,
        )
    return model


def export_geo_to_onnx(model: nn.Module, output_path: str):
    model.eval()
    dummy       = torch.randn(1, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE)
    dynamic     = {"x": {0: torch.export.Dim("batch", min=1, max=64)}}
    torch.onnx.export(
        model,
        kwargs={"x": dummy},
        f=output_path,
        input_names=["x"],
        output_names=["logits"],
        dynamic_shapes=dynamic,
        opset_version=18,
    )
    print(f"Model geo wyeksportowany: {output_path}")


if __name__ == "__main__":
    model = build_geo_model(pretrained=True)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parametry: {total:.1f}M")
    x   = torch.randn(2, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE)
    out = model(x)
    print(f"Output shape: {out.shape}")  # [2, 10]
