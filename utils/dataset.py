import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import CLIPTokenizer

class AmazonCoTDiffusionDataset(Dataset):
    def __init__(self, json_path, image_dir, tokenizer_name="openai/clip-vit-base-patch32"):
        """
        Initializes the dataset for the CoT-Guided Diffusion Bridge.
        """
        print(f"Loading dataset from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        # Extract keys to iterate by index
        self.item_ids = list(self.data.keys())
        self.image_dir = image_dir
        
        # We use a standard CLIP tokenizer for the text path
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            # Normalize to [-1, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.item_ids)

    def __getitem__(self, idx):
        item_id = self.item_ids[idx]
        item = self.data[item_id]
        
        # --- 1. Load and Transform the Image (The Target $x_0$) ---
        img_path = os.path.join(self.image_dir, f"{item_id}.jpg")
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = self.transform(image)
        except Exception as e:
            image_tensor = torch.zeros((3, 256, 256))

        # --- 2. Format the CoT Sequence (The Conditioning $Z$) ---
        cot_core = item.get("cot_core", "")
        cot_visual = item.get("cot_visual", "")
        cot_intent = item.get("cot_intent", "")
        
        # Example: "Item: blender. Visual: glass pitcher, black base. Intent: making smoothies."
        structured_cot_string = f"Item: {cot_core}. Visual: {cot_visual}. Intent: {cot_intent}."
        
        # --- 3. Tokenize the Text ---
        tokens = self.tokenizer(
            structured_cot_string,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        
        return {
            "image": image_tensor,                              
            "input_ids": tokens.input_ids.squeeze(0),           
            "attention_mask": tokens.attention_mask.squeeze(0), 
            "raw_text": item.get("text", "")                    
        }

if __name__ == "__main__":
    dataset = AmazonCoTDiffusionDataset(
        json_path="appliances_cot_ready.json", 
        image_dir="downloaded_images"
    )
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    batch = next(iter(dataloader))
    print("Batch Image Shape:", batch["image"].shape)               
    print("Batch Input IDs Shape:", batch["input_ids"].shape)
    print("Sample CoT Tokens:", batch["input_ids"][0][:10])
