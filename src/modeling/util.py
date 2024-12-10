import os

import torch
import yaml
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModel, AutoTokenizer


def quantize_onnx_model(onnx_model_path: str, quantized_model_path: str):
    quantize_dynamic(onnx_model_path, quantized_model_path, weight_type=QuantType.QInt8)
    print(f"quantized model saved to:{quantized_model_path}")
    return None


def export_onnx_model(output_model_path: str, model: torch.nn.Module | AutoModel, tokenizer: AutoTokenizer):
    if os.path.exists(output_model_path):
        print(f"Warning: overwriting existing ONNX model at path {output_model_path}")
    dummy_text = "test"
    dummy_input = tokenizer(dummy_text, return_tensors="pt")
    torch.onnx.export(
        model,
        tuple([dummy_input["input_ids"], dummy_input["attention_mask"]]),
        f=output_model_path,
        export_params=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence"},
            "attention_mask": {0: "batch_size", 1: "sequence"},
            "logits": {0: "batch_size", 1: "sequence"},
        },
        do_constant_folding=True,
        opset_version=17,
    )
    print(f"Exported ONNX model to {output_model_path}.")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit()

    return cfg
