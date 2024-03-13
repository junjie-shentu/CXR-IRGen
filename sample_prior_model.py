import torch
import ast
import os
import pandas as pd
import pytorch_lightning as pl
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL
from transformers import BartTokenizer
from model.prior_model import PriorModel
from model.lm_model import LMModel
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class PriorEvaluationDataset(Dataset):
    def __init__(self, image_text_data: pd.DataFrame):
        super().__init__()
        self.image_text_data = image_text_data

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.image_text_data)

    def __getitem__(self, idx):
        image_path = self.image_text_data.iloc[idx]["image_path"]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image)
        return image_tensor
    
def collate_fn(batch):
    batch = torch.stack(batch, dim=0)
    return batch


def prepare_inference_data(data_file, data_folder_dir):
    all_data = pd.read_csv(data_file, usecols=["text", "label", "split", "image_path"])
    all_data = all_data.dropna(subset = ["text"])
    all_data = all_data.drop(all_data.loc[all_data["text"]=="nan"].index)
    all_data = all_data.dropna(subset = ["label"])
    all_data = all_data.dropna(subset = ["split"])
    all_data = all_data.dropna(subset = ["image_path"])

    test_split = all_data[all_data["split"] == "test"]
    test_split=test_split.reset_index(drop=True)

    test_split["image_path"] = test_split["image_path"].apply(lambda x: x.replace(".dcm", ".jpg"))
    test_split["image_path"] = test_split["image_path"].apply(lambda x: data_folder_dir + x)

    original_text = test_split["text"]

    for i in range(test_split.shape[0]):
        test_split["label"][i] = ast.literal_eval(test_split["label"][i])
    return test_split, original_text

class PriorModelEvaluation(pl.LightningModule):
    def __init__(self, test_split, batch_size, weight_path, lm_weight_path):
        super().__init__()
        self.prior_model = PriorModel.load_from_checkpoint(weight_path).prior_model
        self.prior_model.eval()
        self.prior_model.to("cuda")

        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder = "vae")
        self.vae.eval()
        self.vae.to("cuda")

        self.lm_model = LMModel.load_from_checkpoint(lm_weight_path).model
        self.lm_model.eval()
        self.lm_model.to("cuda")

        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

        self.test_dataset = PriorEvaluationDataset(test_split)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    def forward(self):
        with torch.no_grad():
            prediction_list = []
            dataloader = tqdm(self.test_dataloader)
            for batch in dataloader:
                image_tensor = batch
                image_tensor = image_tensor.to("cuda")
                image_embedding = self.vae.encode(image_tensor).latent_dist.sample() * 0.18215
                pred_text_embedding = self.prior_model(image_embedding).pooler_output
                #infer text
                output_tokens = torch.tensor([self.lm_model.config.eos_token_id]).expand(pred_text_embedding.shape[0], 1).to("cuda")
                for i in range(512):
                    logits = self.lm_model.get_decoder()(input_ids = output_tokens, encoder_hidden_states = pred_text_embedding).last_hidden_state[:, -1, :]
                    logits = self.lm_model.lm_head(logits) + self.lm_model.final_logits_bias
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    output_tokens = torch.cat([output_tokens, next_token], dim=-1)
                prediction = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
                for sentence in prediction:
                    prediction_list.append(sentence)
        return prediction_list

    
if __name__ == "__main__":
    weight_path = "path to the model checkpoint"
    lm_weight_path = "path to the model checkpoint"

    test_split, original_text = prepare_inference_data("./datasets/data_training_prior.csv")
    prior_model = PriorModelEvaluation(test_split, 20, weight_path, lm_weight_path)
    prediction = prior_model()

    output_path = "path to the folder for saving the generated images"
    os.makedirs(output_path, exist_ok=True)
    prediction = pd.DataFrame({"prediction":prediction})
    prediction.to_csv(os.path.join(output_path, "prior_prediction.csv"), index=False, header=False)




            