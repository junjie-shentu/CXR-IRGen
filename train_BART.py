from configs.train_BART_config import get_config
from dataset.lm_dataset import LMDatasetModule
from model.lm_model import LMModel
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import wandb
from pytorch_lightning.callbacks import ModelCheckpoint

def main(config):
    # Set random seed for reproducibility
    seed_everything(config.random_seed)

    # Create an instance of LMDatasetModule for handling the dataset
    dataset = LMDatasetModule(config.dataset.data_info_text_path, config.dataset.batch_size, config.dataset.num_workers)

    # Create an instance of LMModel with specified configuration
    model = LMModel(config.model.max_len, config.model.model_name, config.model.embedding_plan)

    # Set up WandB logger for experiment tracking
    logger = wandb.WandbLogger(project=config.project, name=config.run_name)

    # Set up ModelCheckpoint callback to save the best model during training
    callbacks = [ModelCheckpoint(config.modelcheckpoint.dirpath,config.modelcheckpoint.filename, monitor='val_loss', mode='min', save_top_k=1, save_last=True)]

    # Set up the Trainer with specified configurations
    trainer = Trainer(accelerator = config.trainer.accelerator, 
                      precision = config.trainer.precision, 
                      max_epochs = config.trainer.max_epochs,
                      callbacks=callbacks, 
                      logger=logger, 
                      val_check_interval=config.trainer.val_interval)

    # Train the model using the Trainer
    trainer.fit(model, dataset)



if __name__ == "__main__":
    # Load the configuration settings for training
    config = get_config()
    # Call the main function with the loaded configuration
    main(config)
