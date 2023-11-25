import torch
from torch import nn
import pytorch_lightning as pl
from diffusers import AutoencoderKL, UNet2DConditionModel,DDPMScheduler, PNDMScheduler
from torch.optim import AdamW
from transformers import CLIPTextModel, CLIPTokenizer


# Function to convert label tensor to descriptive text
def label2text(label):
    findings_list = ["atelectasis",
                     "cardiomegaly", 
                     "consolidation", 
                     "edema", 
                     "enlarged cardiomediastinum", 
                     "fracture", "lung lesion", 
                     "lung opacity", 
                     "no finding", 
                     "pleural effusion", 
                     "pleural other", 
                     "pneumonia", 
                     "pneumothorax", 
                     "support devices"]
    text_list = []
    batch_size = label.shape[0]
    for i in range(batch_size):
        lab = label[i]
        positive_list = []
        negative_list = []
        uncertain_list = []
        for j in range(len(lab)):
            if lab[j] == 1.0:
                positive_list.append(findings_list[j])
            elif lab[j] == -1.0:
                negative_list.append(findings_list[j])
            elif lab[j] == -2.0:
                uncertain_list.append(findings_list[j])
        text = "A chest X ray image"
        if len(positive_list) > 0:
            text += " with " + " and ".join(positive_list)
        if len(negative_list) > 0:
            text += ", without " + " and ".join(negative_list)
        if len(uncertain_list) > 0:
            text += ", and with uncertain " + " and ".join(uncertain_list)
        text_list.append(text)
    return text_list

# LightningModule for the diffusion model using Stable diffusion architecture
class DiffusionModel_t2iSD(pl.LightningModule):
    """Diffusion model based on Stable diffusion architecture"""
    def __init__(self, stable_diffusion_name="CompVis/stable-diffusion-v1-4"):
        super().__init__()
        # Load pre-trained models and tokenizers
        self.unet = UNet2DConditionModel.from_pretrained(stable_diffusion_name, subfolder="unet")
        self.unet.to("cuda")

        self.vae = AutoencoderKL.from_pretrained(stable_diffusion_name, subfolder = "vae")
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.vae.to("cuda")

        self.tokenizer = CLIPTokenizer.from_pretrained(stable_diffusion_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(stable_diffusion_name, subfolder="text_encoder")
        self.text_encoder.to("cuda")
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()

        # Load noise and inference schedulers
        self.noise_scheduler = DDPMScheduler.from_config(stable_diffusion_name, subfolder="scheduler")
        self.inference_scheduler = PNDMScheduler.from_config(stable_diffusion_name, subfolder="scheduler")


    # Training step for LightningModule
    def training_step(self, batch, batch_idx):
        image_tensor, label, clip_image_tensor = batch
        image_tensor = image_tensor.to("cuda")
        clip_image_tensor = clip_image_tensor.to("cuda")

        # Generate descriptive text from labels
        text = label2text(label)
        text_token = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda")
        text_embedding = self.text_encoder(text_token)[0]

        # Concatenate text embedding with clip image tensor
        text_embedding = torch.cat((text_embedding, clip_image_tensor), dim=1)

        # Encode image tensor with VAE and adjust scale
        image_latents = self.vae.encode(image_tensor).latent_dist.sample() * 0.18215

        # Generate random noise and timestep information
        noise = torch.randn_like(image_latents)
        batch_size = image_latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (batch_size,), device = torch.device("cuda"))
        
        # Add noise to image latents using noise scheduler
        noisy_latents = self.noise_scheduler.add_noise(image_latents, noise, timesteps)
        
        # Generate noise prediction using UNet
        noise_pred = self.unet(noisy_latents, timesteps, text_embedding).sample
        loss = nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        self.log("train_loss", loss)
        return loss

    # Validation step for LightningModule
    def validation_step(self, batch, batch_idx):
        image_tensor, label, clip_image_tensor = batch
        image_tensor = image_tensor.to("cuda")
        clip_image_tensor = clip_image_tensor.to("cuda")

        # Generate descriptive text from labels
        text = label2text(label)
        text_token = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda")
        text_embedding = self.text_encoder(text_token)[0]

        # Concatenate text embedding with clip image tensor
        text_embedding = torch.cat((text_embedding, clip_image_tensor), dim=1)

        # Encode image tensor with VAE and adjust scale
        image_latents = self.vae.encode(image_tensor).latent_dist.sample() * 0.18215

        # Generate random noise and timestep information
        noise = torch.randn_like(image_latents)
        batch_size = image_latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (batch_size,), device = torch.device("cuda"))
        
        # Add noise to image latents using noise scheduler
        noisy_latents = self.noise_scheduler.add_noise(image_latents, noise, timesteps)
        
        # Generate noise prediction using UNet
        noise_pred = self.unet(noisy_latents, timesteps, text_embedding).sample
        loss = nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        self.log("val_loss", loss)
        

    # Configure optimizer for training
    def configure_optimizers(self):
        optimizer = AdamW(self.unet.parameters(),
                          lr = 5e-5,
                          betas=(0.9, 0.999),
                          eps=1e-8,
                          weight_decay=0.01)
        return {"optimizer": optimizer}

