import torch
import os
import ast
import pytorch_lightning as pl
import pandas as pd
from PIL import Image
from model.diffusion_model_UNet import DiffusionModel_t2iSD
from diffusers import AutoencoderKL, PNDMScheduler, StableDiffusionPipeline
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModel
from pytorch_lightning import seed_everything
from model.diffusion_model_UNet import label2text
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from dataset.diffusion_model_dataset import DiffusionDatasetModule
from tqdm import tqdm


ramdom_seed = 1234
seed_everything(ramdom_seed, workers = True)

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
        label = ast.literal_eval(self.image_text_data.iloc[idx]["label"])
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
    def __init__(self, data, data_amount, batch_size,weight_path, guidance_scale = 7.5):
        super().__init__()

        self.data = data
        self.data = self.data[:data_amount]
        self.batch_size = batch_size

        self.clip_reference_data = DiffusionDatasetModule("./data/data_training_SD.csv").train_data_info_text
        self.inference_dataset = DiffusionDatasetBase(self.data, self.clip_reference_data)
        self.inference_dataloader = DataLoader(self.inference_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, collate_fn=dl_collate_fn)


        self.unet = DiffusionModel_t2iSD.load_from_checkpoint(weight_path).unet
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

        self.pipeline = StableDiffusionPipeline(
            text_encoder=self.text_encoder,
            vae=self.vae,
            unet=self.unet,
            tokenizer=self.tokenizer,
            scheduler=self.scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker = False,
        )
        self.pipeline.to("cuda")
        

    def sample(self,output_path):
        pipeline = self.pipeline
        os.makedirs(output_path, exist_ok=True)

        with torch.no_grad():
            i = 0
            for batch in tqdm(self.inference_dataloader):
                image_tensor, label, clip_image_tensor = batch
                clip_image_tensor = clip_image_tensor.to("cuda")

                text = label2text(label)
                text_token = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda")
                text_embedding = self.text_encoder(text_token)[0]

                text_embedding = torch.cat((text_embedding, clip_image_tensor), dim=1)

                z_init = torch.randn((label.shape[0],4,32,32), device = torch.device("cuda"))
                images = pipeline(prompt=None,
                                  height=256,
                                  width=256,
                                  prompt_embeds=text_embedding,
                                  guidance_scale = self.guidance_scale,
                                  output_type="pil",
                                  latents=z_init).images

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


