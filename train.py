import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection

from dataset import AmazonCoTDiffusionDataset
from scheduler import DDPMNoiseScheduler
from model import CoTGuidedDiffusion

# --- Hyperparameters ---
JSON_PATH = "appliances_cot_ready.json"
IMAGE_DIR = "downloaded_images"
BATCH_SIZE = 2      
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
SAVE_DIR = "model_checkpoints"
LAMBDA_CONTRASTIVE = 2 # Balances the generative math with the retrieval math

os.makedirs(SAVE_DIR, exist_ok=True)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 1. Initialize Dataset & Dataloader
    dataset = AmazonCoTDiffusionDataset(json_path=JSON_PATH, image_dir=IMAGE_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # 2. Initialize The "Anchor" (Frozen Vision Model)
    print("Loading Frozen Vision Anchor...")
    vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device)
    vision_model.eval()
    for param in vision_model.parameters():
        param.requires_grad = False

    # 3. Initialize Generative Models
    noise_scheduler = DDPMNoiseScheduler(num_train_timesteps=1000)
    model = CoTGuidedDiffusion().to(device)

    # 4. Optimizer targets U-Net + LoRA Text Layers
    optimizer = AdamW(
        list(model.unet.parameters()) + list(model.text_encoder.parameters()), 
        lr=LEARNING_RATE
    )

    print("Starting Joint-Loss Training...")
    model.train()

    # Pre-compute CLIP normalization constants for the GPU
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)

    for epoch in range(NUM_EPOCHS):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        epoch_loss_diff = 0.0
        epoch_loss_cont = 0.0

        for step, batch in enumerate(progress_bar):
            clean_images = batch["image"].to(device)
            cot_input_ids = batch["input_ids"].to(device)
            cot_attention_mask = batch["attention_mask"].to(device)

            current_batch_size = clean_images.shape[0]

            # --- A. Generate the Anchor Image Vectors ---
            with torch.no_grad():
                # Un-normalize from [-1, 1] back to [0, 1]
                img_0_1 = (clean_images + 1.0) / 2.0
                # Scale up to 224x224 for CLIP
                img_224 = F.interpolate(img_0_1, size=(224, 224), mode="bicubic", align_corners=False)
                # Apply CLIP's exact color normalization
                clip_images = (img_224 - clip_mean) / clip_std
                
                # Extract the 512-dim Frozen Anchor Vector
                image_embeds = vision_model(clip_images).image_embeds
                image_embeds = F.normalize(image_embeds, p=2, dim=-1)

            # --- B. The Generative Forward Pass ---
            noise = torch.randn_like(clean_images).to(device)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (current_batch_size,), device=device).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Notice we now unpack the TWO outputs
            noise_pred, text_embeds = model(
                noisy_images=noisy_images, 
                timesteps=timesteps, 
                cot_input_ids=cot_input_ids, 
                cot_attention_mask=cot_attention_mask
            )
            
            # Normalize the text vector for search
            text_embeds = F.normalize(text_embeds, p=2, dim=-1)

            # --- C. The Joint Loss Math ---
            # 1. Did you draw the image correctly?
            loss_diffusion = F.mse_loss(noise_pred, noise)
            
            # 2. Did you stay close to the vision anchor? (1.0 - Cosine Similarity)
            loss_contrastive = 1.0 - F.cosine_similarity(text_embeds, image_embeds).mean()

            # 3. Combine them!
            total_loss = loss_diffusion + (LAMBDA_CONTRASTIVE * loss_contrastive)

            # --- D. Backpropagation ---
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Logging
            epoch_loss_diff += loss_diffusion.item()
            epoch_loss_cont += loss_contrastive.item()
            progress_bar.set_postfix({
                "diff_loss": f"{loss_diffusion.item():.4f}", 
                "cont_loss": f"{loss_contrastive.item():.4f}"
            })

        # Save checkpoint
        checkpoint_path = os.path.join(SAVE_DIR, f"cot_diffusion_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    print("Training Complete!")

if __name__ == "__main__":
    train()
