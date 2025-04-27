import os

import torch
import yaml
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
from transformers import AutoModel, AutoTokenizer


def export_onnx_model(output_model_path: str, model: torch.nn.Module | AutoModel, tokenizer: AutoTokenizer):
    if os.path.exists(output_model_path):
        print(f"Warning: overwriting existing ONNX model at path {output_model_path}")

    # Use a more representative dummy text
    dummy_text = "This is a medical test example with sufficient content to exercise the model"
    # Tokenize the text
    dummy_input = tokenizer(dummy_text, return_tensors="pt")

    # Add metadata inputs for the custom model
    dummy_jurisdiction_id = torch.tensor([2])  # 2 = Unspecified
    dummy_insurance_type_id = torch.tensor([2])  # 2 = Unspecified

    # Check if model accepts metadata
    try:
        # Test if model accepts metadata parameters
        with torch.no_grad():
            _ = model(
                input_ids=dummy_input["input_ids"],
                attention_mask=dummy_input["attention_mask"],
                jurisdiction_id=dummy_jurisdiction_id,
                insurance_type_id=dummy_insurance_type_id,
            )

        # If we got here, model accepts metadata
        print("Exporting model with metadata support...")
        torch.onnx.export(
            model,
            tuple(
                [
                    dummy_input["input_ids"],
                    dummy_input["attention_mask"],
                    dummy_jurisdiction_id,
                    dummy_insurance_type_id,
                ]
            ),
            f=output_model_path,
            export_params=True,
            input_names=["input_ids", "attention_mask", "jurisdiction_id", "insurance_type_id"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "jurisdiction_id": {0: "batch_size"},
                "insurance_type_id": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            do_constant_folding=True,
            opset_version=17,
        )
    except Exception as e:
        print(f"Model doesn't support metadata, exporting without it: {e}")
        # Export without metadata
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
                "logits": {0: "batch_size"},
            },
            do_constant_folding=True,
            opset_version=17,
        )

    print(f"Exported ONNX model to {output_model_path}.")


def quantize_onnx_model(onnx_model_path: str, quantized_model_path: str):
    # print("Preprocessing ONNX model before quantization...")
    # pre_processed_model_path = onnx_model_path + ".pre_processed.onnx"

    # quant_pre_process(onnx_model_path, pre_processed_model_path)

    # Quantize the model
    quantize_dynamic(
        onnx_model_path,
        quantized_model_path,
        per_channel=False,
        reduce_range=False,
        weight_type=QuantType.QInt8,
    )
    print(f"Quantized model saved to: {quantized_model_path}")
    return None


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit()

    return cfg
