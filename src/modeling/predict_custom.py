import functools
import os
import time

import torch
from transformers import (
    AutoTokenizer,
)

from src.modeling.train_custom_outcome_predictor import OverturnModel


@functools.cache
def load_model(model_ckpt_path: str, device: str, quantized: bool = True):
    """Return model"""
    print("Loading model...")

    # Instantiate model
    checkpoint = torch.load(model_ckpt_path, map_location=torch.device(device))
    model = OverturnModel().to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Quantize for performance
    if quantized:
        model = torch.ao.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},  # a set of layers to dynamically quantize
            dtype=torch.qint8,
        )
    print("Done loading model.\n")
    print(
        f"Model stats:\n\tCheckpoint size (MB): {round(os.path.getsize(model_ckpt_path) / (1024 * 1024))}\n\tNum Params: {sum([p.numel() for p in model.parameters()])}"
    )
    return model


if __name__ == "__main__":
    # Load model and tokenizer
    device = "cpu"
    MODEL_DIR = "./models/overturn_predictor/train_backgrounds/intfloat/e5-base-v2/"
    model_ckpt_name = "model.ckpt"
    model_path = os.path.join(MODEL_DIR, model_ckpt_name)
    model = load_model(model_path, device, quantized=False)
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(MODEL_DIR, "tokenizer"),
        model_max_length=model.embedding_backbone.config.max_position_embeddings,
    )

    # Upheld example
    # text = "Diagnosis: Broken Ribs\nTreatment: Inpatient Hospital Admission\n\nThe insurer denied inpatient hospital admission. \n\nThe patient is an adult male. He presented by ambulance to the hospital with severe back pain. The patient had fallen down a ramp and onto his back two days prior. The patient developed back pain and had pain with deep inspiration, prompting a call to 911 for an ambulance. The patient was taking ibuprofen and Tylenol for pain at home. A computed tomography (CT) scan of the patient's chest showed a right posterior minimally displaced 9th and 10th rib fractures. There was no associated intra-abdominal injury. There was atelectasis of the lung in the region of the rib fractures. Vital signs, including oxygen saturation, were normal in the emergency department triage note. The patient did not require supplemental oxygen during the hospitalization. The patient was admitted to the acute inpatient level of care for pain control, breathing treatments, and venous thromboembolism prophylaxis. The patient was seen and cleared by Physical Therapy. The patient's pain was controlled with oral analgesia and a lidocaine patch. Total time in the hospital was less than 13 hours. The acute inpatient level of care was denied coverage by the health plan as not medically necessary."

    # Overturned example
    # text = "Diagnosis: General Debility Due to Lumbar Stenosis\nTreatment: Continued Stay in Skilled Nursing Facility\n\nThe insurer denied continued stay in skilled nursing facility\n\nThe patient is an adult female with a history of general debility due to lumbar stenosis affecting her functional mobility and activities of daily living (ADLs). She has impairments of balance, mobility, and strength, with an increased risk for falls.\n\nThe patient's relevant past medical history includes obesity status post gastric sleeve times two (x2), severe knee and hip osteoarthritis, anxiety, bipolar disorder, hiatus hernia, depression, asthma, hiatus hernia/gastroesophageal reflux disease (GERD), fractured ribs, fractured ankle, sarcoidosis, and pulmonary embolism. Before admission, the patient was living with family and friends in a house, independent with activities of daily living, and with support from others. The patient was admitted to a skilled nursing facility (SNF) three months ago, requiring total dependence for most activities of daily living, and as of two months ago, the patient was non-ambulatory, requiring supervision for bed mobility, contact guard for transfers, and maximum assistance for static standing. She has limitations in completing mobility and locomotive activities due to gross weakness of the bilateral lower extremities, decreased stability and controlled mobility, increased pain, impaired coordination, and decreased aerobic capacity. \n"
    # text = "Diagnosis: Dilated CBD, distended gallbladder\n \nTreatment: Inpatient admission, diagnostic treatment and surgery\n\nThe insurer denied the inpatient admission. The patient presented with abdominal pain. He was afebrile and the vital signs were stable. There was no abdominal rebound or guarding. The WBC count was 14.5. The bilirubin was normal. A CAT scan revealed a dilated CBD and a distended gallbladder. An MRCP revealed a CBD stone with bile duct dilatation. The patient was treated with antibiotics. He underwent an ERCP with sphincterotomy and balloon sweeps. A laparoscopic cholecystectomy was then done. The patient remained hemodynamically stable and his pain was controlled."
    # text = "This is a female patient with a medical history of severe bilateral proliferative diabetic retinopathy and diabetic macular edema. The patient underwent an injection of Lucentis in her left eye and treatment with panretinal photocoagulation without complications. It was reported that the patient had severe disease with many dot/blot hemorrhages. Documentation revealed the patient had arteriovenous (AV) crossing changes bilaterally with venous tortuosity. There were scattered dot/blot hemorrhages bilaterally to the macula and periphery and macular edema. Additionally, she was counseled on proper diet control, exercise and hypertension control. Avastin and Mvasi are the same drug - namely bevacizumab: as per lexicomp: 'humanized monoclonal antibody which binds to, and neutralizes, vascular endothelial growth factor (VEGF), preventing its association with endothelial receptors, Flt-1 and KDR. VEGF binding initiates angiogenesis (endothelial proliferation and the formation of new blood vessels). The inhibition of microvascular growth is believed to retard the growth of all tissues (including metastatic tissue).' Lucentis is ranibizumab: as per lexicomp: 'a recombinant humanized monoclonal antibody fragment which binds to and inhibits human vascular endothelial growth factor A (VEGF-A). Ranibizumab inhibits VEGF from binding to its receptors and thereby suppressing neovascularization and slowing vision loss.' The formulary, step therapy options and the requested drug act against VEGF. There is no suggestion that Avastin or Mvasi would cause physical or mental harm to the patient. There are no contraindications in the documentation that would put the patient at risk for adverse reactions. This patient has a diagnosis of maculopathy. Avastin and Mvasi have been shown to be helpful with this condition."
    # Upheld
    text = "A female patient is being denied wegovy for morbid obesity. The health plan states it is not medically necessary. Doctor recommends it."

    start = time.time()
    tokenized = tokenizer("query" + text, return_tensors="pt").to(device)
    with torch.no_grad():
        result = model(tokenized["input_ids"].to(device), tokenized["attention_mask"].to(device))
    probs = torch.softmax(result[0], dim=-1)
    prob, argmax = torch.max(probs, dim=-1)
    print(f"Class pred: {argmax.item()}, Score: {prob.item()}")
    end = time.time()
    print(f"Latency: {end-start}")
