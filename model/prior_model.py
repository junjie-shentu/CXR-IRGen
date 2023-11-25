import torch
from transformers import ViTConfig, ViTModel
from torch import nn
import pytorch_lightning as pl
from diffusers import AutoencoderKL
from transformers import BartTokenizer
from model.lm_model import LMModel
from torch.optim import AdamW


class PriorModel(pl.LightningModule):
    def __init__(self, max_text_length = 512, embedding_plan = "MeanPooling", pretrained_lm_model=None):
        super().__init__()
        # ViT configuration for the prior model
        self.vitconfig  = ViTConfig(image_size=32, patch_size=2, num_channels=4)

        # Prior model based on ViT architecture
        self.prior_model = ViTModel(self.vitconfig)
        self.prior_model.load_state_dict(torch.load("./assets/vit-base-patch16-224.bin", map_location = "cpu"), strict = False)#load a pretrained model (different from version 1) 
        self.prior_model.to("cuda")

        # Maximum length of text sequences and embedding plan
        self.max_text_length = max_text_length
        self.embedding_plan = embedding_plan

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

        # Autoencoder for image encoding
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder = "vae")
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae.to("cuda")

        # Tokenizer for processing text
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

         # Language model for obtaining text embeddings
        self.language_model = LMModel.load_from_checkpoint(pretrained_lm_model).model
        self.language_model.eval()
        self.language_model.to("cuda")

    
    def training_step(self, batch, batch_idx):

        # Extract image and text data from the batch
        image, text = batch
        image = image.to("cuda")

        # Obtain image embeddings using the VAE
        image_embedding = self.vae.encode(image).latent_dist.sample() * 0.18215

        # Tokenize and process text data
        tokenized_text = self.tokenizer.batch_encode_plus(text, pad_to_max_length=True, max_length=self.max_text_length, truncation=True, return_tensors="pt")
        text_ids = tokenized_text["input_ids"].to("cuda")
        attention_mask = tokenized_text["attention_mask"].to("cuda")

        # Obtain text embeddings based on the specified embedding plan
        encoder_last_hidden_state = self.language_model.get_encoder()(text_ids, attention_mask).last_hidden_state
        if self.embedding_plan == "EOS":
            eos_index = (text_ids == self.tokenizer.eos_token_id).nonzero()[:,1]
            text_embedding = torch.gather(encoder_last_hidden_state, 1, eos_index.unsqueeze(1).unsqueeze(2).repeat(1,1,encoder_last_hidden_state.shape[-1])).squeeze(1)
        elif self.embedding_plan == "BOS":
            text_embedding = encoder_last_hidden_state[:, 0, :]
        elif self.embedding_plan == "MeanPooling":
            text_embedding = (encoder_last_hidden_state * attention_mask.unsqueeze(2)).sum(axis=1) / attention_mask.sum(axis=1).unsqueeze(1)

        # Predict text embeddings using the Prior Model
        pred_text_embedding = self.prior_model(image_embedding).pooler_output

        # Compute MSE loss and cosine similarity loss, combined with a coefficient
        mse_loss = self.mse_loss(pred_text_embedding, text_embedding)
        cosine_similarity_loss = 1.0 - self.cosine_similarity(pred_text_embedding, text_embedding)
        train_loss = (mse_loss + 0.01*(cosine_similarity_loss)).mean()  #combine mse loss and cosine similarity loss, with coefficient 0.01

        # Log and return the training loss
        self.log("mse_loss", mse_loss.mean())
        self.log("cosine_similarity_loss", cosine_similarity_loss.mean())
        self.log("train_loss", train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):

        # Similar structure as the training step for validation

        image, text = batch
        image = image.to("cuda")

        image_embedding = self.vae.encode(image).latent_dist.sample() * 0.18215

        tokenized_text = self.tokenizer.batch_encode_plus(text, pad_to_max_length=True, max_length=self.max_text_length, truncation=True, return_tensors="pt")
        text_ids = tokenized_text["input_ids"].to("cuda")
        attention_mask = tokenized_text["attention_mask"].to("cuda")
        encoder_last_hidden_state = self.language_model.get_encoder()(text_ids, attention_mask).last_hidden_state
        if self.embedding_plan == "EOS":
            eos_index = (text_ids == self.tokenizer.eos_token_id).nonzero()[:,1]
            text_embedding = torch.gather(encoder_last_hidden_state, 1, eos_index.unsqueeze(1).unsqueeze(2).repeat(1,1,encoder_last_hidden_state.shape[-1])).squeeze(1)
        elif self.embedding_plan == "BOS":
            text_embedding = encoder_last_hidden_state[:, 0, :]
        elif self.embedding_plan == "MeanPooling":
            text_embedding = (encoder_last_hidden_state * attention_mask.unsqueeze(2)).sum(axis=1) / attention_mask.sum(axis=1).unsqueeze(1)


        pred_text_embedding = self.prior_model(image_embedding).pooler_output

        mse_loss = self.mse_loss(pred_text_embedding, text_embedding)
        cosine_similarity_loss = 1.0 - self.cosine_similarity(pred_text_embedding, text_embedding)
        val_loss = (mse_loss + 0.01*(cosine_similarity_loss)).mean()
        self.log("val_mse_loss", mse_loss.mean())
        self.log("val_cosine_similarity_loss", cosine_similarity_loss.mean())
        self.log("val_loss", val_loss)


    def forward(self, x):
        x = self.prior_model(x)
        return x

    def configure_optimizers(self):
        optimizer = AdamW(self.prior_model.parameters(),
                          lr = 5e-5,
                          betas=(0.9, 0.999),
                          eps=1e-8,
                          weight_decay=0.01)
        return {"optimizer": optimizer}





