import os
import time

import numpy as np
import onnxruntime
import scipy
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TextClassificationPipeline,
    pipeline,
)

from src.modeling.util import export_onnx_model, quantize_onnx_model

MODEL_DIR = "./models/overturn_predictor"


class ClassificationPipeline(TextClassificationPipeline):
    def postprocess(self, model_outputs):
        out_logits = model_outputs["logits"]
        return out_logits


if __name__ == "__main__":
    # TODO: get these from model checkpoints
    ID2LABEL = {0: "Insufficient", 1: "Upheld", 2: "Overturned"}
    LABEL2ID = {v: k for k, v in ID2LABEL.items()}

    # Load model and tokenizer
    pretrained_model_key = "distilbert/distilbert-base-cased"

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_key, model_max_length=512)

    dataset_name = "train_backgrounds_suff_augmented"
    checkpoints_dir = os.path.join(MODEL_DIR, dataset_name, "distilbert")
    checkpoint_dirs = sorted(os.listdir(checkpoints_dir))
    checkpoint_name = checkpoint_dirs[0]
    ckpt_path = os.path.join(checkpoints_dir, checkpoint_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt_path, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )

    # Upheld example
    text = "Diagnosis: Broken Ribs\nTreatment: Inpatient Hospital Admission\n\nThe insurer denied inpatient hospital admission. \n\nThe patient is an adult male. He presented by ambulance to the hospital with severe back pain. The patient had fallen down a ramp and onto his back two days prior. The patient developed back pain and had pain with deep inspiration, prompting a call to 911 for an ambulance. The patient was taking ibuprofen and Tylenol for pain at home. A computed tomography (CT) scan of the patient's chest showed a right posterior minimally displaced 9th and 10th rib fractures. There was no associated intra-abdominal injury. There was atelectasis of the lung in the region of the rib fractures. Vital signs, including oxygen saturation, were normal in the emergency department triage note. The patient did not require supplemental oxygen during the hospitalization. The patient was admitted to the acute inpatient level of care for pain control, breathing treatments, and venous thromboembolism prophylaxis. The patient was seen and cleared by Physical Therapy. The patient's pain was controlled with oral analgesia and a lidocaine patch. Total time in the hospital was less than 13 hours. The acute inpatient level of care was denied coverage by the health plan as not medically necessary."

    # Overturned example
    # text = "Diagnosis: General Debility Due to Lumbar Stenosis\nTreatment: Continued Stay in Skilled Nursing Facility\n\nThe insurer denied continued stay in skilled nursing facility\n\nThe patient is an adult female with a history of general debility due to lumbar stenosis affecting her functional mobility and activities of daily living (ADLs). She has impairments of balance, mobility, and strength, with an increased risk for falls.\n\nThe patient's relevant past medical history includes obesity status post gastric sleeve times two (x2), severe knee and hip osteoarthritis, anxiety, bipolar disorder, hiatus hernia, depression, asthma, hiatus hernia/gastroesophageal reflux disease (GERD), fractured ribs, fractured ankle, sarcoidosis, and pulmonary embolism. Before admission, the patient was living with family and friends in a house, independent with activities of daily living, and with support from others. The patient was admitted to a skilled nursing facility (SNF) three months ago, requiring total dependence for most activities of daily living, and as of two months ago, the patient was non-ambulatory, requiring supervision for bed mobility, contact guard for transfers, and maximum assistance for static standing. She has limitations in completing mobility and locomotive activities due to gross weakness of the bilateral lower extremities, decreased stability and controlled mobility, increased pain, impaired coordination, and decreased aerobic capacity. \n"
    # text = "Diagnosis: Dilated CBD, distended gallbladder\n \nTreatment: Inpatient admission, diagnostic treatment and surgery\n\nThe insurer denied the inpatient admission. The patient presented with abdominal pain. He was afebrile and the vital signs were stable. There was no abdominal rebound or guarding. The WBC count was 14.5. The bilirubin was normal. A CAT scan revealed a dilated CBD and a distended gallbladder. An MRCP revealed a CBD stone with bile duct dilatation. The patient was treated with antibiotics. He underwent an ERCP with sphincterotomy and balloon sweeps. A laparoscopic cholecystectomy was then done. The patient remained hemodynamically stable and his pain was controlled."
    # text = "This is a female patient with a medical history of severe bilateral proliferative diabetic retinopathy and diabetic macular edema. The patient underwent an injection of Lucentis in her left eye and treatment with panretinal photocoagulation without complications. It was reported that the patient had severe disease with many dot/blot hemorrhages. Documentation revealed the patient had arteriovenous (AV) crossing changes bilaterally with venous tortuosity. There were scattered dot/blot hemorrhages bilaterally to the macula and periphery and macular edema. Additionally, she was counseled on proper diet control, exercise and hypertension control. Avastin and Mvasi are the same drug - namely bevacizumab: as per lexicomp: 'humanized monoclonal antibody which binds to, and neutralizes, vascular endothelial growth factor (VEGF), preventing its association with endothelial receptors, Flt-1 and KDR. VEGF binding initiates angiogenesis (endothelial proliferation and the formation of new blood vessels). The inhibition of microvascular growth is believed to retard the growth of all tissues (including metastatic tissue).' Lucentis is ranibizumab: as per lexicomp: 'a recombinant humanized monoclonal antibody fragment which binds to and inhibits human vascular endothelial growth factor A (VEGF-A). Ranibizumab inhibits VEGF from binding to its receptors and thereby suppressing neovascularization and slowing vision loss.' The formulary, step therapy options and the requested drug act against VEGF. There is no suggestion that Avastin or Mvasi would cause physical or mental harm to the patient. There are no contraindications in the documentation that would put the patient at risk for adverse reactions. This patient has a diagnosis of maculopathy. Avastin and Mvasi have been shown to be helpful with this condition."
    # tokens = tokenizer(text)
    # with torch.no_grad():
    #     output = model(torch.tensor(tokens["input_ids"]))
    # print(output)

    # Upheld
    # text = "A patient is being denied wegovy for morbid obesity. The health plan states it is not medically necessary."

    classifier = ClassificationPipeline(model=model, tokenizer=tokenizer)
    classifier2 = pipeline("text-classification", model=model, tokenizer=tokenizer)
    start = time.time()
    result = classifier(text)
    # result = classifier2(text)
    end = time.time()
    print("Vanilla HF pipeline:")
    print(f"Logits: {result[0]}")
    probs = torch.softmax(result[0], dim=-1)
    print(f"Probs: {probs}")
    prob, argmax = torch.max(probs, dim=-1)
    print(f"Class pred: {argmax.item()}, Score: {prob.item()}")
    print(f"Latency: {end-start}")
    print(
        "Model size (MB):",
        round(os.path.getsize(os.path.join(ckpt_path, "model.safetensors")) / (1024 * 1024)),
        "\n",
    )

    # Pytorch quantized model
    model = classifier.model.to("cpu")
    # TODO: Fix this, this is not the right model to be quantizing via the torch or onnx ops below.
    model_int8 = torch.ao.quantization.quantize_dynamic(
        model,  # the original model
        {torch.nn.Linear},  # a set of layers to dynamically quantize
        dtype=torch.qint8,
    )  # the target dtype for quantized weights
    model_int8.eval()
    start = time.time()
    tokenized = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        result = model_int8(**tokenized)
    end = time.time()
    print("Quantized pytorch model:")
    probs = torch.softmax(result.logits, dim=-1)
    print(f"Probs: {probs}")
    prob, argmax = torch.max(probs, dim=-1)
    print(f"Class pred: {argmax.item()}, Score: {prob.item()}")
    print(f"Latency: {end-start}")
    param_size = 0
    for param in model_int8.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / (1024 * 1024)
    print(f"Model size (MB): {round(size_all_mb)}\n")

    # Export onxx model and quantized version, if nonexistent
    onnx_file_name = "model.onnx"
    onnx_model_path = os.path.join(ckpt_path, onnx_file_name)
    quant_onnx_model_path = os.path.join(ckpt_path, "quant-model.onnx")
    export_onnx_model(onnx_model_path, model, tokenizer)

    if not os.path.exists(quant_onnx_model_path):
        quantize_onnx_model(onnx_model_path, quant_onnx_model_path)

        print(
            "ONNX full precision model size (MB):",
            os.path.getsize(onnx_model_path) / (1024 * 1024),
        )

    # Try running with onnx runtime (quantized)
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = onnxruntime.InferenceSession(quant_onnx_model_path, sess_options)
    start = time.time()
    inputs = tokenizer(text, return_tensors="np")
    outputs = session.run(output_names=["logits"], input_feed=dict(inputs))
    result = scipy.special.softmax(outputs[0], axis=-1)
    end = time.time()
    print("Quantized  Onnx:")
    print(f"Probs: {result[0]}")
    argmax = np.argmax(result[0], axis=-1)
    prob = result[0][argmax]
    print(f"Class pred: {argmax}, Score: {prob}")
    print(f"Latency: {end-start}")
    print(
        "Model size (MB):",
        round(os.path.getsize(quant_onnx_model_path) / (1024 * 1024)),
        "\n",
    )

    # Try running with onnx runtime
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = onnxruntime.InferenceSession(onnx_model_path, sess_options)
    start = time.time()
    inputs = tokenizer(text, return_tensors="np")
    outputs = session.run(output_names=["logits"], input_feed=dict(inputs))
    result = scipy.special.softmax(outputs[0], axis=-1)
    end = time.time()
    print("Onnx:")
    print(f"Probs: {result}")
    argmax = np.argmax(result[0], axis=-1)
    prob = result[0][argmax]
    print(f"Class pred: {argmax}, Score: {prob}")
    print(f"Latency: {end-start}")
    print(
        "Model size (MB):",
        round(os.path.getsize(onnx_model_path) / (1024 * 1024)),
    )
