import ml_collections
import torch
import torch.nn as nn


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.device = torch.device("cuda")
    config.random_seed = 1234
    config.project = "CXR-IRGen"
    config.run_name = "train_UNet"
    config.save_model_path = "./result/" + f"{config.project}" + "/" +f"{config.run_name}"
    config.stable_diffusion_name = "CompVis/stable-diffusion-v1-4"

    config.dataset = d(
        data_info_text_path = "./dataset/data_training_SD",
        batch_size = 32,
        num_workers = 0
    )

    config.modelcheckpoint = d(
        dirpath = config.save_model_path,
        filename = '{epoch}-{val_loss:.3f}',
    )

    config.trainer = d(
        accelerator="auto",
        precision=32,
        max_epochs=10,
        val_interval = 0.1,
    )

    return config