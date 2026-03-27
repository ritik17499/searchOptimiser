import torch
import torch.nn as nn
from transformers import CLIPTextModelWithProjection
from diffusers import UNet2DConditionModel
from peft import LoraConfig, get_peft_model

class CoTGuidedDiffusion(nn.Module):
    def __init__(self, text_encoder_name="openai/clip-vit-base-patch32"):
        super().__init__()
        
        print("Initializing CoT-Guided Diffusion Bridge with Joint Output...")
        
        # --- 1. The Text Encoder ---
        base_text_encoder = CLIPTextModelWithProjection.from_pretrained(text_encoder_name)
        
        lora_config = LoraConfig(
            r=8, 
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"], 
            lora_dropout=0.05,
            bias="none"
        )
        self.text_encoder = get_peft_model(base_text_encoder, lora_config)
        self.text_encoder.print_trainable_parameters()
        
        cross_attention_dim = self.text_encoder.config.hidden_size 

        # --- 2. The U-Net (The Visual Reconstructor) ---
        self.unet = UNet2DConditionModel(
            sample_size=64,           
            in_channels=3,             
            out_channels=3,            
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512), 
            down_block_types=(
                "CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D", "DownBlock2D",           
            ),
            up_block_types=(
                "UpBlock2D", "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",
            ),
            cross_attention_dim=cross_attention_dim, 
        )
        self.unet.enable_gradient_checkpointing()

    def forward(self, noisy_images, timesteps, cot_input_ids, cot_attention_mask):
        # Step 1: Pass text through the LoRA encoder
        encoder_outputs = self.text_encoder(
            input_ids=cot_input_ids,
            attention_mask=cot_attention_mask
        )
        
        # The deep tokens for the U-Net to draw the image
        cot_embeddings = encoder_outputs.last_hidden_state
        
        # The final 512-dim vector for the Search Database
        text_embeds = encoder_outputs.text_embeds
            
        # Step 2: Predict the Noise
        noise_prediction = self.unet(
            sample=noisy_images,
            timestep=timesteps,
            encoder_hidden_states=cot_embeddings
        ).sample
        
        # We now return BOTH outputs
        return noise_prediction, text_embeds
