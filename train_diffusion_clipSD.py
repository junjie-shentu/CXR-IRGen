from configs.train_UNet_config import get_config
from dataset.diffusion_model_dataset import DiffusionDatasetModule
from model.diffusion_model_UNet import DiffusionModel_t2iSD
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import wandb
from pytorch_lightning.callbacks import ModelCheckpoint

# Define the main function for training the diffusion model
def main(config):
    # Set random seed for reproducibility
    seed_everything(config.random_seed)

    # Create an instance of DiffusionDatasetModule for handling the dataset
    dataset = DiffusionDatasetModule(config.dataset.data_info_text_path, config.dataset.batch_size, config.dataset.num_workers)

    # Create an instance of DiffusionModel_t2iSD with specified parameters
    model = DiffusionModel_t2iSD(stable_diffusion_name=config.stable_diffusion_name)

    # Set up WandB logger for experiment tracking
    logger = wandb.WandbLogger(project=config.project, name=config.run_name)

    # Set up ModelCheckpoint callback to save the best model during training
    callbacks = [ModelCheckpoint(config.modelcheckpoint.dirpath,config.modelcheckpoint.filename, monitor='val_loss', mode='min', every_n_train_steps= 5e3, save_top_k=1, save_last=True)]

    # Set up the Trainer with specified configurations
    trainer = Trainer(accelerator = config.trainer.accelerator, 
                      precision = config.trainer.precision, 
                      max_epochs = config.trainer.max_epochs,
                      callbacks=callbacks, 
                      logger=logger, 
                      val_check_interval=config.trainer.val_interval)
    
    trainer.fit(model, dataset)



if __name__ == "__main__":
    # Load the configuration settings for training
    config = get_config()

    # Call the main function with the loaded configuration
    main(config)
