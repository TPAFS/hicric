#!/usr/bin/env python
import argparse
import copy
import os

import numpy as np
import onnxruntime
import scipy
import torch
from colorama import Fore, Style, init
from transformers import AutoTokenizer

from src.modeling.train_outcome_predictor import TextClassificationWithMetadata
from src.modeling.util import export_onnx_model, quantize_onnx_model

# Initialize colorama
init()

MODEL_DIR = "./models/overturn_predictor"
ID2LABEL = {0: "Insufficient", 1: "Upheld", 2: "Overturned"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# Test examples with expected classifications
TEST_EXAMPLES = [
    {
        "text": "Diagnosis: Broken Ribs\nTreatment: Inpatient Hospital Admission\n\nThe insurer denied inpatient hospital admission. \n\nThe patient is an adult male. He presented by ambulance to the hospital with severe back pain. The patient had fallen down a ramp and onto his back two days prior. The patient developed back pain and had pain with deep inspiration, prompting a call to 911 for an ambulance. The patient was taking ibuprofen and Tylenol for pain at home. A computed tomography (CT) scan of the patient's chest showed a right posterior minimally displaced 9th and 10th rib fractures. There was no associated intra-abdominal injury. There was atelectasis of the lung in the region of the rib fractures. Vital signs, including oxygen saturation, were normal in the emergency department triage note. The patient did not require supplemental oxygen during the hospitalization. The patient was admitted to the acute inpatient level of care for pain control, breathing treatments, and venous thromboembolism prophylaxis. The patient was seen and cleared by Physical Therapy. The patient's pain was controlled with oral analgesia and a lidocaine patch. Total time in the hospital was less than 13 hours. The acute inpatient level of care was denied coverage by the health plan as not medically necessary.",
        "expected": "Upheld",
    },
    {
        "text": "Diagnosis: General Debility Due to Lumbar Stenosis\nTreatment: Continued Stay in Skilled Nursing Facility\n\nThe insurer denied continued stay in skilled nursing facility\n\nThe patient is an adult female with a history of general debility due to lumbar stenosis affecting her functional mobility and activities of daily living (ADLs). She has impairments of balance, mobility, and strength, with an increased risk for falls.\n\nThe patient's relevant past medical history includes obesity status post gastric sleeve times two (x2), severe knee and hip osteoarthritis, anxiety, bipolar disorder, hiatus hernia, depression, asthma, hiatus hernia/gastroesophageal reflux disease (GERD), fractured ribs, fractured ankle, sarcoidosis, and pulmonary embolism. Before admission, the patient was living with family and friends in a house, independent with activities of daily living, and with support from others. The patient was admitted to a skilled nursing facility (SNF) three months ago, requiring total dependence for most activities of daily living, and as of two months ago, the patient was non-ambulatory, requiring supervision for bed mobility, contact guard for transfers, and maximum assistance for static standing. She has limitations in completing mobility and locomotive activities due to gross weakness of the bilateral lower extremities, decreased stability and controlled mobility, increased pain, impaired coordination, and decreased aerobic capacity.",
        "expected": "Overturned",
    },
    {
        "text": "Diagnosis: Dilated CBD, distended gallbladder\n \nTreatment: Inpatient admission, diagnostic treatment and surgery\n\nThe insurer denied the inpatient admission. The patient presented with abdominal pain. He was afebrile and the vital signs were stable. There was no abdominal rebound or guarding. The WBC count was 14.5. The bilirubin was normal. A CAT scan revealed a dilated CBD and a distended gallbladder. An MRCP revealed a CBD stone with bile duct dilatation. The patient was treated with antibiotics. He underwent an ERCP with sphincterotomy and balloon sweeps. A laparoscopic cholecystectomy was then done. The patient remained hemodynamically stable and his pain was controlled.",
        "expected": "Upheld",
    },
    {
        "text": "This is a female patient with a medical history of severe bilateral proliferative diabetic retinopathy and diabetic macular edema. The patient underwent an injection of Lucentis in her left eye and treatment with panretinal photocoagulation without complications. It was reported that the patient had severe disease with many dot/blot hemorrhages. Documentation revealed the patient had arteriovenous (AV) crossing changes bilaterally with venous tortuosity. There were scattered dot/blot hemorrhages bilaterally to the macula and periphery and macular edema. Additionally, she was counseled on proper diet control, exercise and hypertension control. Avastin and Mvasi are the same drug - namely bevacizumab: as per lexicomp: 'humanized monoclonal antibody which binds to, and neutralizes, vascular endothelial growth factor (VEGF), preventing its association with endothelial receptors, Flt-1 and KDR. VEGF binding initiates angiogenesis (endothelial proliferation and the formation of new blood vessels). The inhibition of microvascular growth is believed to retard the growth of all tissues (including metastatic tissue).' Lucentis is ranibizumab: as per lexicomp: 'a recombinant humanized monoclonal antibody fragment which binds to and inhibits human vascular endothelial growth factor A (VEGF-A). Ranibizumab inhibits VEGF from binding to its receptors and thereby suppressing neovascularization and slowing vision loss.' The formulary, step therapy options and the requested drug act against VEGF. There is no suggestion that Avastin or Mvasi would cause physical or mental harm to the patient. There are no contraindications in the documentation that would put the patient at risk for adverse reactions. This patient has a diagnosis of maculopathy. Avastin and Mvasi have been shown to be helpful with this condition.",
        "expected": "Upheld",
    },
    {"text": "hello? is this relevant?", "expected": "Insufficient"},
    {
        "text": "A patient is being denied wegovy for morbid obesity. The health plan states it is not medically necessary.",
        "expected": "Overturned",
    },
    {
        "text": "This patient has extensive and inoperable carcinoma of the stomach. He was started on chemotherapy with Xeloda and Oxaliplatin, because he has less nausea with Oxaliplatin than with the alternative, Cisplatin. Oxaliplatin was denied as experimental for treatment of his gastric cancer.",
        "expected": "Overturned",
    },
    {
        "text": "This is a patient who was denied breast tomosynthesis to screen for breast cancer.",
        "expected": "Overturned",
    },
    {
        "text": "This is a patient with Crohn's Disease who is being treated with Humira. Their health plan has denied Anser ADA blood level testing for Humira, claiming it is investigational.",
        "expected": "Upheld",
    },
    {
        "text": "The patient is a 44-year-old female who initially presented with an abnormal screening mammogram. The patient was seen by a radiation oncologist who recommended treatment of the right chest wall and comprehensive nodal regions using proton beam radiation therapy.",
        "expected": "Upheld",
    },
    {
        "text": "The patient is a 10-year-old female with a history of Pitt-Hopkins syndrome and associated motor planning difficulties, possible weakness in the oral area, and receptive and expressive language delays. The provider has recommended that the patient continue to receive individual speech and language therapy sessions twice a week for 60-minute sessions. The Health Insurer has denied the requested services as not medically necessary for treatment of the patient’s medical condition.",
        "expected": "Overturned",
    },
    {
        "text": "The patient is a nine-year-old female with a history of autism spectrum disorder and a speech delay. The patient’s parent has requested reimbursement for the ABA services provided over the course of a year. The Health Insurer has denied the services at issue as not medically necessary for the treatment of the patient.",
        "expected": "Overturned",
    },
]


def run_pytorch_model(model, tokenizer, text, jurisdiction_id=2, insurance_type_id=2):
    """Run inference with the PyTorch model"""
    model.eval()
    tokenized = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        # Convert jurisdiction and insurance type to tensors
        j_id = torch.tensor([jurisdiction_id])
        i_id = torch.tensor([insurance_type_id])

        result = model(**tokenized, jurisdiction_id=j_id, insurance_type_id=i_id)

    probs = torch.softmax(result["logits"], dim=-1)
    prob, argmax = torch.max(probs, dim=-1)
    return {
        "class_id": argmax.item(),
        "class_name": ID2LABEL[argmax.item()],
        "confidence": prob.item(),
        "probs": probs[0].tolist(),
        "logits": result["logits"][0].tolist(),
    }


def run_pytorch_model_no_metadata(model, tokenizer, text):
    """Run inference with the PyTorch model without metadata"""
    model.eval()
    tokenized = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        try:
            # Try running without metadata
            result = model(**tokenized)
            probs = torch.softmax(result["logits"], dim=-1)
            prob, argmax = torch.max(probs, dim=-1)

            return {
                "class_id": argmax.item(),
                "class_name": ID2LABEL[argmax.item()],
                "confidence": prob.item(),
                "probs": probs[0].tolist(),
                "logits": result["logits"][0].tolist(),
            }
        except Exception as e:
            return {"error": str(e), "class_name": "ERROR", "confidence": 0.0, "probs": [0.0, 0.0, 0.0]}


def run_pytorch_quantized(model_int8, tokenizer, text, jurisdiction_id=2, insurance_type_id=2):
    """Run inference with the quantized PyTorch model"""
    model_int8.eval()
    tokenized = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        # Convert jurisdiction and insurance type to tensors
        j_id = torch.tensor([jurisdiction_id])
        i_id = torch.tensor([insurance_type_id])

        result = model_int8(**tokenized, jurisdiction_id=j_id, insurance_type_id=i_id)

    probs = torch.softmax(result["logits"], dim=-1)
    prob, argmax = torch.max(probs, dim=-1)

    return {
        "class_id": argmax.item(),
        "class_name": ID2LABEL[argmax.item()],
        "confidence": prob.item(),
        "probs": probs[0].tolist(),
        "logits": result["logits"][0].tolist(),
    }


def run_onnx(session, tokenizer, text, jurisdiction_id=2, insurance_type_id=2):
    """Run inference with ONNX Runtime"""
    try:
        # Tokenize input text
        inputs = tokenizer(text, return_tensors="np", truncation=True, padding=True)

        # Construct the full input feed
        input_feed = dict(inputs)

        # Check if the model expects metadata inputs
        input_names = [input.name for input in session.get_inputs()]

        # Add jurisdiction and insurance type if supported by the model
        if "jurisdiction_id" in input_names:
            input_feed["jurisdiction_id"] = np.array([jurisdiction_id], dtype=np.int64)
        if "insurance_type_id" in input_names:
            input_feed["insurance_type_id"] = np.array([insurance_type_id], dtype=np.int64)

        # Run inference
        outputs = session.run(output_names=["logits"], input_feed=input_feed)

        # Process outputs
        result = scipy.special.softmax(outputs[0], axis=-1)
        argmax = np.argmax(result[0])
        prob = result[0][argmax]
        return {
            "class_id": int(argmax),
            "class_name": ID2LABEL[int(argmax)],
            "confidence": float(prob),
            "probs": result[0].tolist(),
            "logits": outputs[0][0],
        }
    except Exception as e:
        return {"error": str(e), "class_name": "ERROR", "confidence": 0.0, "probs": [0.0, 0.0, 0.0]}


def print_result(model_name, result, expected=None, show_probs=True, show_logits=True):
    """Print inference result with color-coding based on matching expected output"""
    if "error" in result:
        print(f"{model_name}: {Fore.RED}ERROR{Style.RESET_ALL} - {result['error']}")
        return

    # Determine color based on expected value
    color = Fore.WHITE
    if expected:
        if result["class_name"] == expected:
            color = Fore.GREEN
        else:
            color = Fore.RED

    # Print result
    print(f"{model_name}: {color}{result['class_name']}{Style.RESET_ALL} ({result['confidence']:.4f})")

    # Optionally show logits
    if show_logits:
        logits_str = ", ".join([f"{ID2LABEL[i]}: {p:.4f}" for i, p in enumerate(result["logits"])])
        print(f"  Logits: {logits_str}")

    # Optionally show probabilities
    if show_probs:
        probs_str = ", ".join([f"{ID2LABEL[i]}: {p:.4f}" for i, p in enumerate(result["probs"])])
        print(f"  Probabilities: {probs_str}")


def test_all_examples(model, model_int8, onnx_session, onnx_quant_session, tokenizer):
    """Run all test examples through all model variants"""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}===== TESTING ALL EXAMPLES ====={Style.RESET_ALL}")

    for i, example in enumerate(TEST_EXAMPLES):
        text = example["text"]
        expected = example["expected"]

        # Print a shortened version of the text
        print(
            f"\n{Fore.YELLOW}{Style.BRIGHT}Example {i + 1}: '{text[:250]}...' (Expected: {expected}){Style.RESET_ALL}"
        )

        # Run through all model variants
        pt_result = run_pytorch_model(model, tokenizer, text)
        # pt_no_metadata_result = run_pytorch_model_no_metadata(model, tokenizer, text)
        pt_quant_result = run_pytorch_quantized(model_int8, tokenizer, text)

        onnx_result = run_onnx(onnx_session, tokenizer, text)
        onnx_quant_result = run_onnx(onnx_quant_session, tokenizer, text)

        # Print results
        print_result("PyTorch (with metadata)", pt_result, expected)
        # print_result("PyTorch (no metadata)", pt_no_metadata_result, expected)
        print_result("PyTorch (quantized)", pt_quant_result, expected)
        print_result("ONNX", onnx_result, expected)
        print_result("ONNX (quantized)", onnx_quant_result, expected)

        print("-" * 80)


def process_single_prompt(model, model_int8, onnx_session, onnx_quant_session, tokenizer, text):
    """Process a single user-provided prompt through all model variants"""
    print(f"\n{Fore.YELLOW}{Style.BRIGHT}Processing: '{text[:250]}...'{Style.RESET_ALL}")

    # Run through all model variants
    pt_result = run_pytorch_model(model, tokenizer, text)
    pt_no_metadata_result = run_pytorch_model_no_metadata(model, tokenizer, text)
    pt_quant_result = run_pytorch_quantized(model_int8, tokenizer, text)

    onnx_result = run_onnx(onnx_session, tokenizer, text)
    onnx_quant_result = run_onnx(onnx_quant_session, tokenizer, text)

    # Print results
    print_result("PyTorch (with metadata)", pt_result, show_probs=True)
    print_result("PyTorch (no metadata)", pt_no_metadata_result, show_probs=True)
    print_result("PyTorch (quantized)", pt_quant_result)
    print_result("ONNX", onnx_result)
    print_result("ONNX (quantized)", onnx_quant_result)


def calibrate_model(model, tokenizer, calibration_data):
    """Run calibration data through the model for quantization preparation"""
    print(f"{Fore.CYAN}Calibrating model with {len(calibration_data)} examples...{Style.RESET_ALL}")
    model.eval()

    # Define mappings for jurisdiction and insurance type if not already in the calibration data
    JURISDICTION_MAP = {"NY": 0, "CA": 1, "Unspecified": 2}
    INSURANCE_TYPE_MAP = {"Commercial": 0, "Medicaid": 1, "Unspecified": 2}

    with torch.no_grad():
        for example in calibration_data:
            text = example["text"]

            # Get jurisdiction and insurance type IDs, default to "Unspecified" (2)
            if "jurisdiction" in example:
                j_id = JURISDICTION_MAP.get(example["jurisdiction"], 2)
            else:
                j_id = example.get("jurisdiction_id", 2)

            if "insurance_type" in example:
                i_id = INSURANCE_TYPE_MAP.get(example["insurance_type"], 2)
            else:
                i_id = example.get("insurance_type_id", 2)

            # Tokenize the text
            tokenized = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

            # Convert jurisdiction and insurance type to tensors
            j_tensor = torch.tensor([j_id])
            i_tensor = torch.tensor([i_id])

            # Run forward pass for calibration
            _ = model(**tokenized, jurisdiction_id=j_tensor, insurance_type_id=i_tensor)

    return model


def quantize_model_with_proper_embedding_config(model, tokenizer, calibration_data=None):
    """Quantize model focusing on embeddings and linear layers while avoiding layer_norm issues"""
    print(f"{Fore.CYAN}Setting up quantization for embeddings and linear layers...{Style.RESET_ALL}")

    # Create a copy of the model to preserve the original
    model_copy = copy.deepcopy(model)
    model_copy.eval()

    # Dynamic quantization approach that properly handles both layer types
    model_int8 = torch.ao.quantization.quantize_dynamic(
        model_copy,
        {
            torch.nn.Linear,
            #  torch.nn.Embedding
        },  # quantize both linear and embedding layers
        dtype=torch.qint8,
    )

    print(f"{Fore.GREEN}Quantization complete (Linear and Embedding layers){Style.RESET_ALL}")
    return model_int8


def main():
    parser = argparse.ArgumentParser(description="Appeal Classification Model Inference")
    parser.add_argument("--test", action="store_true", help="Run all test examples")
    parser.add_argument("--prompt", type=str, help="Text to classify")
    args = parser.parse_args()

    if not args.test and not args.prompt:
        parser.error("Either --test or --prompt must be specified")

    # Load model and tokenizer
    dataset_name = "train_backgrounds_suff_augmented"
    checkpoints_dir = os.path.join(MODEL_DIR, dataset_name, "distilbert")
    checkpoint_dirs = sorted(os.listdir(checkpoints_dir))
    checkpoint_name = checkpoint_dirs[0]
    ckpt_path = os.path.join(checkpoints_dir, checkpoint_name)

    print(f"{Fore.CYAN}Loading models...{Style.RESET_ALL}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

    # Load PyTorch model
    model = TextClassificationWithMetadata.from_pretrained(
        ckpt_path,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    print(
        "Vanilla Pytorch Model size (MB):",
        round(os.path.getsize(os.path.join(ckpt_path, "model.safetensors")) / (1024 * 1024)),
    )

    # Prepare calibration data from test examples
    calibration_data = []
    for example in TEST_EXAMPLES:
        # Add jurisdiction and insurance_type variations for each example
        for j_id in [0, 1, 2]:  # NY, CA, Unspecified
            for i_id in [0, 1, 2]:  # Commercial, Medicaid, Unspecified
                calibration_data.append({"text": example["text"], "jurisdiction_id": j_id, "insurance_type_id": i_id})

    # Prepare calibration data from test examples
    calibration_data = []
    for example in TEST_EXAMPLES:
        # Add jurisdiction and insurance_type variations for each example
        for j_id in [0, 1, 2]:  # NY, CA, Unspecified
            for i_id in [0, 1, 2]:  # Commercial, Medicaid, Unspecified
                calibration_data.append({"text": example["text"], "jurisdiction_id": j_id, "insurance_type_id": i_id})

    # Quantize model with proper embedding configuration
    model_int8 = quantize_model_with_proper_embedding_config(model.to("cpu"), tokenizer, calibration_data)

    print(f"{Fore.GREEN}Model calibration and quantization complete{Style.RESET_ALL}")
    param_size = 0
    for param in model_int8.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / (1024 * 1024)
    print(f"Quantized Pytorch Model size (MB): {round(size_all_mb)}")

    # Load ONNX models
    onnx_model_path = os.path.join(ckpt_path, "model.onnx")
    quant_onnx_model_path = os.path.join(ckpt_path, "quant-model.onnx")

    # Ensure ONNX models exist
    if not os.path.exists(onnx_model_path):
        print("Exporting model to ONNX format...")
        export_onnx_model(onnx_model_path, model, tokenizer)
    print(
        "Onnx Model size (MB):",
        round(os.path.getsize(onnx_model_path) / (1024 * 1024)),
    )

    if not os.path.exists(quant_onnx_model_path):
        print("Quantizing ONNX model...")
        quantize_onnx_model(onnx_model_path, quant_onnx_model_path)
    print(
        "Quantized Onnx Model size (MB):",
        round(os.path.getsize(quant_onnx_model_path) / (1024 * 1024)),
    )
    # Create ONNX sessions
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    onnx_session = onnxruntime.InferenceSession(onnx_model_path, sess_options)
    onnx_quant_session = onnxruntime.InferenceSession(quant_onnx_model_path, sess_options)

    print(f"{Fore.GREEN}Models loaded successfully{Style.RESET_ALL}")

    # Run the appropriate mode
    if args.test:
        test_all_examples(model, model_int8, onnx_session, onnx_quant_session, tokenizer)
    else:
        process_single_prompt(model, model_int8, onnx_session, onnx_quant_session, tokenizer, args.prompt)


if __name__ == "__main__":
    main()
