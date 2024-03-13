from configs.train_prior_config import get_config
from dataset.prior_dataset import PriorDatasetModule
from model.prior_model import PriorModel
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# Main function to train the Prior Model
def main(config):
    # Set random seed for reproducibility
    seed_everything(config.random_seed)

    # Create Prior Dataset Module using configuration
    dataset = PriorDatasetModule(config.dataset.data_info_text_path, config.dataset.batch_size, config.dataset.num_workers)

    # Instantiate Prior Model using configuration
    model = PriorModel(config.model.max_len, config.model.embedding_plan, config.model.pretrained_vit_path, config.model.pretrained_lm_model)

    # Configure logger and callbacks, including ModelCheckpoint and LearningRateMonitor
    logger = wandb.WandbLogger(project=config.project, name=config.run_name)
    callbacks = [ModelCheckpoint(config.modelcheckpoint.dirpath,config.modelcheckpoint.filename, monitor='val_loss', mode='min', save_top_k=1, save_last=True)]
    
    # Configure Trainer with desired settings
    trainer = Trainer(accelerator = "auto", precision = 32, max_epochs = 20, callbacks=callbacks, logger=logger, val_check_interval=0.1)

    trainer.fit(model, dataset)



if __name__ == "__main__":
    # Get configuration using the get_config function
    config = get_config
    # Call the main function with the obtained configuration
    main(config)
