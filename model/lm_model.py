import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import pytorch_lightning as pl
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right
from torch.optim import AdamW
from transformers.modeling_outputs import Seq2SeqLMOutput


# Define a LightningModule for Language Modeling
class LMModel(pl.LightningModule):
    def __init__(self, max_length = 512, model_name="facebook/bart-base", embedding_plan = "MeanPooling"):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to("cuda")
        self.embedding_plan = embedding_plan
        self.loss_f = CrossEntropyLoss()

    # Training step for LightningModule
    def training_step(self, train_batch, idx):
        batch = train_batch
        tokrnized_batch = self.tokenizer.batch_encode_plus(batch, pad_to_max_length=True, max_length=self.max_length, truncation=True, return_tensors="pt")
        input_ids = tokrnized_batch["input_ids"].to("cuda")
        attention_mask = tokrnized_batch["attention_mask"].to("cuda")
        encoder_last_hidden_state = self.model.get_encoder()(input_ids, attention_mask).last_hidden_state

        # Different embedding plans based on the configuration
        if self.embedding_plan == "EOS":
            # Extract encoder output based on the position of the EOS token
            eos_index = (input_ids == self.tokenizer.eos_token_id).nonzero()[:,1]
            encoder_outputs = torch.gather(encoder_last_hidden_state, 1, eos_index.unsqueeze(1).unsqueeze(2).repeat(1,1,encoder_last_hidden_state.shape[-1])).squeeze(1)
        elif self.embedding_plan == "BOS":
            # Use the hidden state corresponding to the BOS token
            encoder_outputs = encoder_last_hidden_state[:, 0, :]
        elif self.embedding_plan == "MeanPooling":
            # Use mean pooling of the encoder hidden states
            encoder_outputs = (encoder_last_hidden_state * attention_mask.unsqueeze(2)).sum(axis=1) / attention_mask.sum(axis=1).unsqueeze(1)

        # Prepare labels for masked language model (MLM) training
        shifted_labels = shift_tokens_right(input_ids, self.tokenizer.pad_token_id, self.tokenizer.eos_token_id).to("cuda")
        shifted_attention_mask = torch.cat([torch.ones(attention_mask.shape[0], 1).to("cuda"), attention_mask[:, :-1]], dim=1).to("cuda")

        # Generate predictions and compute MLM loss
        preds = self.model.get_decoder()(input_ids = shifted_labels, attention_mask = shifted_attention_mask, encoder_hidden_states = encoder_outputs)
        preds = self.model.lm_head(preds[0]) + self.model.final_logits_bias
        train_loss = self.loss_f(preds.view(-1, self.tokenizer.vocab_size), input_ids.view(-1))
        self.log("train_loss", train_loss)
        return train_loss
    
    # Validation step for LightningModule
    def validation_step(self, val_batch, idx):
        batch = val_batch
        tokrnized_batch = self.tokenizer.batch_encode_plus(batch, pad_to_max_length=True, max_length=self.max_length, truncation=True, return_tensors="pt")
        input_ids = tokrnized_batch["input_ids"].to("cuda")
        attention_mask = tokrnized_batch["attention_mask"].to("cuda")
        encoder_last_hidden_state = self.model.get_encoder()(input_ids, attention_mask).last_hidden_state

        # Similar embedding plans for validation
        if self.embedding_plan == "EOS":
            eos_index = (input_ids == self.tokenizer.eos_token_id).nonzero()[:,1]
            encoder_outputs = torch.gather(encoder_last_hidden_state, 1, eos_index.unsqueeze(1).unsqueeze(2).repeat(1,1,encoder_last_hidden_state.shape[-1])).squeeze(1)
        elif self.embedding_plan == "BOS":
            encoder_outputs = encoder_last_hidden_state[:, 0, :]
        elif self.embedding_plan == "MeanPooling":
            encoder_outputs = (encoder_last_hidden_state * attention_mask.unsqueeze(2)).sum(axis=1) / attention_mask.sum(axis=1).unsqueeze(1)

        
        shifted_labels = shift_tokens_right(input_ids, self.tokenizer.pad_token_id, self.tokenizer.eos_token_id).to("cuda")
        shifted_attention_mask = torch.cat([torch.ones(attention_mask.shape[0], 1).to("cuda"), attention_mask[:, :-1]], dim=1).to("cuda")
        preds = self.model.get_decoder()(input_ids = shifted_labels, attention_mask = shifted_attention_mask, encoder_hidden_states = encoder_outputs)
        preds = self.model.lm_head(preds[0]) + self.model.final_logits_bias
        val_loss = self.loss_f(preds.view(-1, self.tokenizer.vocab_size), input_ids.view(-1))
        self.log("val_loss", val_loss, batch_size=len(batch))

    # Forward method for the model
    def forward(self, text):
        tokrnized_batch = self.tokenizer.batch_encode_plus(text, pad_to_max_length=True, max_length=self.max_length, truncation=True, return_tensors="pt")
        input_ids = tokrnized_batch["input_ids"].to("cuda")
        attention_mask = tokrnized_batch["attention_mask"].to("cuda")
        encoder_last_hidden_state = self.model.get_encoder()(input_ids, attention_mask).last_hidden_state

        if self.embedding_plan == "EOS":
            eos_index = (input_ids == self.tokenizer.eos_token_id).nonzero()[:,1]
            encoder_outputs = torch.gather(encoder_last_hidden_state, 1, eos_index.unsqueeze(1).unsqueeze(2).repeat(1,1,encoder_last_hidden_state.shape[-1])).squeeze(1)
        elif self.embedding_plan == "BOS":
            encoder_outputs = encoder_last_hidden_state[:, 0, :]
        elif self.embedding_plan == "MeanPooling":
            encoder_outputs = (encoder_last_hidden_state * attention_mask.unsqueeze(2)).sum(axis=1) / attention_mask.sum(axis=1).unsqueeze(1)

        shifted_labels = shift_tokens_right(input_ids, self.tokenizer.pad_token_id, self.tokenizer.eos_token_id).to("cuda")
        shifted_attention_mask = torch.cat([torch.ones(attention_mask.shape[0], 1).to("cuda"), attention_mask[:, :-1]], dim=1).to("cuda")
        preds = self.model.get_decoder()(input_ids = shifted_labels, attention_mask = shifted_attention_mask, encoder_hidden_states = encoder_outputs)
        preds = self.model.lm_head(preds[0]) + self.model.final_logits_bias

        masked_lm_loss = self.loss_f(preds.view(-1, self.tokenizer.vocab_size), input_ids.view(-1))

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=preds,
        )

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(),
                          lr = 5e-5,
                          betas=(0.9, 0.999),
                          eps=1e-8,
                          weight_decay=0.01)
        return {"optimizer": optimizer}
    
