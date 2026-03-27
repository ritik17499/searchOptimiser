import torch

class DDPMNoiseScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        """
        Initializes the linear variance schedule.
        beta represents how much variance/noise is added at each step.
        """
        self.num_train_timesteps = num_train_timesteps
        
        # Create a linear schedule of betas from start to end
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        
        # alpha = 1 - beta
        self.alphas = 1.0 - self.betas
        
        # alpha_bar (cumulative product of alphas). This is what allows us to 
        # jump to any timestep 't' instantly without iterating.
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, original_samples, noise, timesteps):
        """
        Takes the clean images, target noise, and a batch of timesteps,
        and returns the noisy images (x_t).
        
        original_samples: Clean image tensors (Batch, Channels, Height, Width)
        noise: Random Gaussian noise of the exact same shape
        timesteps: 1D tensor of random integer timesteps for each image in the batch
        """
        # Ensure the alphas tensor is on the same device as our images 
        alphas_cumprod = self.alphas_cumprod.to(original_samples.device)
        
        # Extract the specific alpha_bar values for our batch of timesteps
        alphas_cumprod_t = alphas_cumprod[timesteps]
        
        # Reshape so we can broadcast these scalar values across the whole image tensor
        # Shape becomes (Batch, 1, 1, 1)
        sqrt_alpha_prod = torch.sqrt(alphas_cumprod_t).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = torch.sqrt(1 - alphas_cumprod_t).view(-1, 1, 1, 1)

        # The core DDPM formula: x_t = sqrt(alpha_bar)*x_0 + sqrt(1-alpha_bar)*epsilon
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        
        return noisy_samples
