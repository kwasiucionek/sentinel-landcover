"""
SegFormer-B2 fine-tuned na EuroSAT.
3 kanały RGB, input 256x256.
"""
import torch
import torch.nn as nn
from transformers import SegformerForImageClassification, SegformerConfig

NUM_CLASSES = 10
IN_CHANNELS = 3   # RGB — bez fake NIR
IMAGE_SIZE  = 256  # zwiększony kontekst przestrzenny


def build_model(pretrained: bool = True) -> nn.Module:
    if pretrained:
        model = SegformerForImageClassification.from_pretrained(
            "nvidia/mit-b2",
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True,
        )
    else:
        config = SegformerConfig.from_pretrained("nvidia/mit-b2")
        config.num_labels = NUM_CLASSES
        model = SegformerForImageClassification(config)
    return model


def export_to_onnx(model: nn.Module, output_path: str):
    model.eval()
    dummy       = torch.randn(1, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    dynamic     = {"pixel_values": {0: torch.export.Dim("batch", min=1, max=16)}}
    torch.onnx.export(
        model,
        kwargs={"pixel_values": dummy},
        f=output_path,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_shapes=dynamic,
        opset_version=18,
    )
    print(f"Model wyeksportowany: {output_path}")


if __name__ == "__main__":
    model = build_model(pretrained=True)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parametry: {total:.1f}M")
    x   = torch.randn(2, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    out = model(pixel_values=x)
    print(f"Output shape: {out.logits.shape}")
