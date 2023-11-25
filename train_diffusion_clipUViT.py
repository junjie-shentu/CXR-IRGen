from configs.train_UViT_config import get_config
from dataset.diffusion_model_dataset import DiffusionDatasetModule
from model.diffusion_model_UViT import DiffusionModel_t2iUViT
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import wandb
from pytorch_lightning.callbacks import ModelCheckpoint

# Main function to train the diffusion model with UViT architecture
def main(config):
    # Set random seed for reproducibility
    seed_everything(config.random_seed)

    # Create an instance of DiffusionDatasetModule for managing the dataset
    dataset = DiffusionDatasetModule(config.dataset.data_info_text_path, config.dataset.batch_size, config.dataset.num_workers)

    # Create an instance of DiffusionModel_t2iUViT with specified configuration
    model = DiffusionModel_t2iUViT(config.uvit.pretrained_uvit_path,
                           config.uvit.img_size,
                           config.uvit.patch_size,
                           config.uvit.in_chans,
                           config.uvit.embed_dim,
                           config.uvit.depth,
                           config.uvit.num_heads,
                           config.uvit.mlp_ratio,
                           config.uvit.qkv_bias,
                           config.uvit.norm_layer,
                           config.uvit.mlp_time_embed,
                           config.uvit.use_checkpoint)
    
    # Set up Wandb logger for logging training information
    logger = wandb.WandbLogger(project=config.project, name=config.run_name)

    # Set up ModelCheckpoint callback for saving the model during training
    callbacks = [ModelCheckpoint(config.modelcheckpoint.dirpath,config.modelcheckpoint.filename, monitor='val_loss', mode='min', every_n_train_steps=5e3, save_top_k=1, save_last=True)]

    # Set up Trainer with specified configuration
    trainer = Trainer(accelerator = config.trainer.accelerator, 
                      precision = config.trainer.precision, 
                      max_epochs = config.trainer.max_epochs,
                      callbacks=callbacks, 
                      logger=logger, 
                      val_check_interval=config.trainer.val_interval)
    
    trainer.fit(model, dataset)



if __name__ == "__main__":
    # Load configuration using the get_config function
    config = get_config()

    # Call the main function with the loaded configuration
    main(config)
