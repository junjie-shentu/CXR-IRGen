import torch
import inspect
import os
import pytorch_lightning as pl
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from model.diffusion_model_UViT import DiffusionModel_t2iUViT
from diffusers import AutoencoderKL, PNDMScheduler
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModel
from pytorch_lightning import seed_everything
from tqdm import tqdm
from model.diffusion_model_UNet import label2text
from dataset.diffusion_model_dataset import DiffusionDatasetModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ramdom_seed = 1234
seed_everything(ramdom_seed, workers = True)


class T2iUViTPipeline(nn.Module):
    def __init__(self,
                 vae = None,
                 unet = None,
                 text_encoder = None,
                 text_embedding_layer = None,
                 scheduler = None,
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__()
        self.vae = vae
        self.unet = unet
        self.text_encoder = text_encoder
        self.text_embedding_layer = text_embedding_layer
        self.scheduler = scheduler
        self.device = device


    def prepare_extra_step_kwargs(self, generator, eta):

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs


    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def numpy_to_pil(self, images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images


    def __call__(self, 
                 prompts: torch.Tensor,
                 height: int = 32,
                 width: int = 32,
                 guidance_scale: float = 7.5,
                 num_inference_steps: int = 50,
                 output_type = "pil",
                 generator = torch.Generator(device="cuda"),
                 latents = None,
    ):

        batch_size = prompts.shape[0]

        do_classifier_free_guidence = guidance_scale > 1.0

        #prepare prompts
        if do_classifier_free_guidence:
            negative_token = torch.tensor([[49407]*77 for _ in range(batch_size)]).to("cuda") # 49407 is the id of EOS/PAD token in CLIPTokenizer, 77 is the maximim length of token in CLIPTokenizer
            negative_prompts = self.text_embedding_layer(self.text_encoder(negative_token)[1]).unsqueeze(1)
            prompts = torch.cat([negative_prompts, prompts])
        else: 
            prompts = prompts

        #prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device = self.vae.device)
        timesteps = self.scheduler.timesteps

        #prepare latents
        num_channels_latents = self.unet.in_chans

        single_latents_shape = (1, num_channels_latents, height, width)
        if latents is None:
            latents = [
                torch.randn(single_latents_shape, generator = generator[i], device = self.vae.device, dtype = self.vae.dtype)
                for i in range(batch_size)
            ]        
            latents = torch.cat(latents, dim = 0).to(self.vae.device)
        else:
            latents = latents
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        #Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator=generator, eta=0.0)

        #denoising loop
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents]*2) if do_classifier_free_guidence else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            t_tensor = torch.ones((2 * batch_size, )).cuda() * t if do_classifier_free_guidence else torch.ones((batch_size, )).cuda() * t
            noise_pred = self.unet(latent_model_input, t_tensor, prompts)

            if do_classifier_free_guidence:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)


            latents = self.scheduler.step(noise_pred,  #latents is the previous noisy sample: x_t -> x_t-1
                                          t,
                                          latents,
                                          **extra_step_kwargs
                                        ).prev_sample
            
        if output_type == "latent":
            images = latents
        elif output_type == "pil":
            images = self.decode_latents(latents)
            images = self.numpy_to_pil(images)

        return images



def prepare_inference_data(data_file, data_folder_dir):
    all_image_data = pd.read_csv(data_file, usecols=["label", "ViewPosition", "split", "image_path"])
    all_image_data = all_image_data.dropna(subset = ["image_path"])
    all_image_data = all_image_data.dropna(subset = ["label"])

    all_image_data = all_image_data.loc[all_image_data["image_path"].str.contains("/p19/",na=False)]
    all_image_data = all_image_data[all_image_data["ViewPosition"] == "PA"]
    all_image_data["image_path"] = all_image_data["image_path"].apply(lambda x: x.replace(".dcm", ".jpg"))
    all_image_data["image_path"] = all_image_data["image_path"].apply(lambda x: data_folder_dir + x)
    all_image_data=all_image_data.reset_index(drop=True)

    all_image_data=all_image_data[:1000]#only use 1000 images for evaluation

    return all_image_data

class DiffusionDatasetBase(Dataset):
    def __init__(self, image_text_data: pd.DataFrame, clip_reference_data: pd.DataFrame):
        super().__init__()
        self.image_text_data = image_text_data
        self.clip_reference_data = clip_reference_data

        self.clipvisionmodel = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clipvisionmodel.requires_grad_(False)



    def image_transform(self, image, size):
        image_transform = transforms.Compose(
            [
                transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        processed_image = image_transform(image)
        return processed_image

    def __len__(self):
        return len(self.image_text_data)

    def __getitem__(self, idx):
        image_path = self.image_text_data.iloc[idx]["image_path"]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image, 256)
        label = self.image_text_data.iloc[idx]["label"]
        label = torch.tensor(label)

        clip_image_exist = False
        for i in range(self.clip_reference_data.shape[0]):
            row_label = self.clip_reference_data.iloc[i]["label"]
            if row_label == self.image_text_data.iloc[idx]["label"]:
                clip_image_exist = True
                clip_image_path = self.clip_reference_data.iloc[i]["image_path"]
                break

        if clip_image_exist:
            clip_image = Image.open(clip_image_path).convert("RGB")
            clip_image_tensor = self.image_transform(clip_image, 224)
            clip_image_tensor = self.clipvisionmodel(clip_image_tensor.unsqueeze(0))[1]
        else:
            clip_image_tensor = torch.zeros(1,768)#768 is the hidden size of CLIP; there can be multiple selection for non-existing clip image, including all zeros, the similar ones, or no findings
        return image_tensor, label, clip_image_tensor

def dl_collate_fn(batch):
    return torch.stack([row[0] for row in batch]), torch.stack([row[1] for row in batch]), torch.stack([row[2] for row in batch])


class DiffusionModelEvaluation(pl.LightningModule):
    def __init__(self, data, batch_size, weight_path, guidance_scale = 7.5):
        super().__init__()

        self.data = data
        self.batch_size = batch_size

        self.clip_reference_data = DiffusionDatasetModule("./data/data_training_SD.csv").train_data_info_text
        self.inference_dataset = DiffusionDatasetBase(self.data, self.clip_reference_data)
        self.inference_dataloader = DataLoader(self.inference_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, collate_fn=dl_collate_fn)
        self.unet = DiffusionModel_t2iUViT.load_from_checkpoint(weight_path).uvit.uvit
        self.text_embed = DiffusionModel_t2iUViT.load_from_checkpoint(weight_path).uvit.text_embed
        self.text_embed.eval()
        self.text_embed.to("cuda")

        self.unet.requires_grad_(False)
        self.unet.eval()
        self.unet.to("cuda")
        
        
        self.tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder")

        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder = "vae")
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae.to("cuda")


        self.scheduler = PNDMScheduler.from_config("CompVis/stable-diffusion-v1-4", subfolder="scheduler")


        self.guidance_scale = guidance_scale

        self.pipeline = T2iUViTPipeline(
                 vae = self.vae,
                 unet = self.unet,
                 text_encoder = self.text_encoder,
                 text_embedding_layer = self.text_embed,
                 scheduler = self.scheduler,
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        self.pipeline.to("cuda")
        

    def sample(self, output_path):
        pipeline = self.pipeline

        os.makedirs(output_path, exist_ok=True)

        with torch.no_grad():
            i = 0
            for batch in tqdm(self.inference_dataloader):
                image_tensor, label, clip_image_tensor = batch
                clip_image_tensor = clip_image_tensor.to("cuda")

                text = label2text(label)
                text_token = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda")
                text_embedding = self.text_encoder(text_token)[1].unsqueeze(1)#1:pooler output

                text_embedding = (text_embedding + clip_image_tensor) / 2

                text_embedding = self.text_embed(text_embedding)

                z_init = torch.randn((label.shape[0],4,32,32), device = torch.device("cuda"))
                images = pipeline(prompts=text_embedding,
                                  height=256,
                                  width=256,
                                  guidance_scale = self.guidance_scale,
                                  output_type="pil",
                                  latents=z_init)

                for j, image in enumerate(images):
                    image.save(os.path.join(output_path, f'{i*self.batch_size +j}.png'))

                i += 1



    







if __name__ == "__main__":
    data_folder_dir = "path to the folder containing the images"
    output_path = "path to the folder for saving the generated images"
    weight_path= "path to the model checkpoint"
    data = prepare_inference_data("./data/data_training_SD.csv", data_folder_dir)
    model = DiffusionModelEvaluation(data, data_amount=data.shape[0], batch_size=20, weight_path=weight_path)
    model.sample(output_path)


