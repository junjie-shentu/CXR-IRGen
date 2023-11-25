import torch
from torch import nn
import pytorch_lightning as pl
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler
from .uvit import UViT
from torch.optim import AdamW
from transformers import CLIPTextModel, CLIPTokenizer

# Function to convert label tensor to text description
def label2text(label):
    findings_list = ["atelectasis", 
                     "cardiomegaly", 
                     "consolidation", 
                     "edema", 
                     "enlarged cardiomediastinum", 
                     "fracture", 
                     "lung lesion", 
                     "lung opacity", 
                     "no finding", 
                     "pleural effusion", 
                     "pleural other", 
                     "pneumonia", 
                     "pneumothorax", 
                     "support devices"]
    text_list = []
    batch_size = label.shape[0]

    # Iterate over batch and convert each label to text description
    for i in range(batch_size):
        lab = label[i]
        positive_list = []
        negative_list = []
        uncertain_list = []

        # Check label values and categorize findings
        for j in range(len(lab)):
            if lab[j] == 1.0:
                positive_list.append(findings_list[j])
            elif lab[j] == -1.0:
                negative_list.append(findings_list[j])
            elif lab[j] == -2.0:
                uncertain_list.append(findings_list[j])

        # Construct text description based on findings
        text = "A chest X ray image"
        if len(positive_list) > 0:
            text += " with " + " and ".join(positive_list)
        if len(negative_list) > 0:
            text += ", without " + " and ".join(negative_list)
        if len(uncertain_list) > 0:
            text += ", and with uncertain " + " and ".join(uncertain_list)
        text_list.append(text)
    return text_list


# Class defining UViT conditioned diffusion model
class T2iConditionedUViT(nn.Module):
    def __init__(self, uvit):
        super().__init__()
        self.uvit = uvit
        self.text_embed = nn.Linear(768, 1024)# 1024 is the hidden size of uvit

    def forward(self, noisy_latents, timesteps, text_embedding):
        text_embedding = self.text_embed(text_embedding)
        noise_pred = self.uvit(noisy_latents, timesteps, text_embedding)
        return noise_pred

        text_embedding = self.text_embed(text_embedding)

class DiffusionModel_t2iUViT(pl.LightningModule):
    """Diffusion model based on UViT architecture"""
    def __init__(self, pretrained_uvit_path="./assets/imagenet256_uvit_large.pth", img_size=32, patch_size=2, in_chans=4, embed_dim=1024, depth=20, num_heads=16, mlp_ratio=4.,
                 qkv_bias=False, norm_layer=nn.LayerNorm, mlp_time_embed=False, use_checkpoint=True):
        super().__init__()
        # UViT model for image conditioning
        self.uvit = UViT(img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio,
                 qkv_bias, norm_layer, mlp_time_embed, use_checkpoint)
        self.uvit.load_state_dict(torch.load(pretrained_uvit_path, map_location = "cpu"), strict = False)
        self.uvit = T2iConditionedUViT(self.uvit)
        self.uvit.to("cuda")

        # Autoencoder for image encoding
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder = "vae")
        self.vae.eval()
        self.vae.to("cuda")

        # Tokenizer and text encoder for processing textual information
        self.tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder")
        self.text_encoder.to("cuda")
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()

        # Noise scheduler for adding noise during training
        self.noise_scheduler = DDPMScheduler.from_config("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
        self.inference_scheduler = PNDMScheduler.from_config("CompVis/stable-diffusion-v1-4", subfolder="scheduler")#or we can try using the DPMSolver


    def training_step(self, batch, batch_idx):
        image_tensor, label, clip_image_tensor = batch
        image_tensor = image_tensor.to("cuda")
        clip_image_tensor = clip_image_tensor.to("cuda")

        text = label2text(label)
        text_token = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda")
        text_embedding = self.text_encoder(text_token)[1].unsqueeze(1)


        text_embedding = (text_embedding + clip_image_tensor) / 2

        image_latents = self.vae.encode(image_tensor).latent_dist.sample() * 0.18215
        noise = torch.randn_like(image_latents)
        batch_size = image_latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (batch_size,), device = torch.device("cuda"))
        noisy_latents = self.noise_scheduler.add_noise(image_latents, noise, timesteps)
        noise_pred = self.uvit(noisy_latents, timesteps, text_embedding)
        loss = nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image_tensor, label, clip_image_tensor = batch
        image_tensor = image_tensor.to("cuda")
        clip_image_tensor = clip_image_tensor.to("cuda")


        text = label2text(label)
        text_token = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda")
        text_embedding = self.text_encoder(text_token)[1].unsqueeze(1)#1:pooler output

        #we take the mean value of the clip_image_tensor and text_embedding as the final text_embedding
        text_embedding = (text_embedding + clip_image_tensor) / 2

        image_latents = self.vae.encode(image_tensor).latent_dist.sample() * 0.18215
        noise = torch.randn_like(image_latents)
        batch_size = image_latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (batch_size,), device = torch.device("cuda"))
        noisy_latents = self.noise_scheduler.add_noise(image_latents, noise, timesteps)
        noise_pred = self.uvit(noisy_latents, timesteps, text_embedding)
        loss = nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        self.log("val_loss", loss)


    def configure_optimizers(self):
        optimizer = AdamW(self.uvit.parameters(),
                          lr = 5e-5,
                          betas=(0.9, 0.999),
                          eps=1e-8,
                          weight_decay=0.01)
        return {"optimizer": optimizer}