import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor, CLIPTokenizer

from model import CoTGuidedDiffusion

# --- Configuration ---
JSON_PATH = "appliances_cot_ready.json"
IMAGE_DIR = "downloaded_images"
CHECKPOINT_PATH = "model_checkpoints/cot_diffusion_epoch_100.pt" 

def evaluate_recall():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Recall Evaluation on {device}...")

    # 1. Load the Ground Truth Data
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        data_list = []
        for key, value in data.items():
            item = value.copy()
            if 'id' not in item:
                item['id'] = key 
            data_list.append(item)
        data = data_list

    data = data[:1000]
    num_items = len(data)

    # 2. Initialize the Vision Pathway
    print("Loading Frozen CLIP Vision Encoder...")
    vision_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device)
    vision_model.eval()

    # 3. Initialize Your Trained LoRA Model
    print("Loading Trained LoRA Text Encoder...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    custom_model = CoTGuidedDiffusion().to(device)
    
    # --- BASELINE TOGGLE ---
    # To run the Baseline, comment out the next line.
    # To run your Experimental results, leave it active.
   # custom_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    # -----------------------
    
    custom_model.eval()
    text_encoder = custom_model.text_encoder

    # 4. Build the Vector Database
    image_embeddings = torch.zeros((num_items, 512)).to(device)
    text_embeddings = torch.zeros((num_items, 512)).to(device)
    valid_indices = []

    print("Building Vector Database & Encoding Queries...")
    with torch.no_grad():
        for i, item in enumerate(tqdm(data, desc="Encoding vectors")):
            
            # --- A. Encode the Image ---
            image_path = os.path.join(IMAGE_DIR, f"{item['id']}.jpg")
            if not os.path.exists(image_path): continue
            try: raw_image = Image.open(image_path).convert("RGB")
            except: continue
                
            img_inputs = vision_processor(images=raw_image, return_tensors="pt").to(device)
            img_out = vision_model(**img_inputs)
            image_embeddings[i] = F.normalize(img_out.image_embeds[0], p=2, dim=-1)

            # --- B. Encode the CoT Text ---
            cot_string = f"Item: {item['cot_core']}. Visual: {item['cot_visual']}. Intent: {item['cot_intent']}."
            text_inputs = tokenizer(
                cot_string, padding="max_length", truncation=True, max_length=77, return_tensors="pt"
            ).to(device)
            
            with torch.cuda.amp.autocast():
                txt_out = text_encoder(
                    input_ids=text_inputs.input_ids,
                    attention_mask=text_inputs.attention_mask
                )
            
            projected_text_embeds = txt_out.text_embeds
            
            # Normalize the correctly projected vector
            text_embeddings[i] = F.normalize(projected_text_embeds[0].float(), p=2, dim=-1)
            valid_indices.append(i)

    # 5. Calculate Metrics
    final_num_items = len(valid_indices)
    valid_image_embeddings = image_embeddings[valid_indices]
    valid_text_embeddings = text_embeddings[valid_indices]
    
    similarity_matrix = torch.matmul(valid_text_embeddings, valid_image_embeddings.T)
    
    hits_at_1, hits_at_5, hits_at_10 = 0, 0, 0

    for i in range(final_num_items):
        scores = similarity_matrix[i]
        top_10_indices = torch.topk(scores, k=10).indices.tolist()
        if i in top_10_indices[:1]: hits_at_1 += 1
        if i in top_10_indices[:5]: hits_at_5 += 1
        if i in top_10_indices: hits_at_10 += 1

    print("="*40)
    print("      RETRIEVAL EVALUATION RESULTS      ")
    print("="*40)
    print(f"Total Queries Evaluated: {final_num_items}")
    print(f"Recall@1:  {(hits_at_1 / final_num_items) * 100:.2f}%")
    print(f"Recall@5:  {(hits_at_5 / final_num_items) * 100:.2f}%")
    print(f"Recall@10: {(hits_at_10 / final_num_items) * 100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    evaluate_recall()
