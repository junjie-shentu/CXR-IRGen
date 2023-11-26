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
    config.run_name = "train_prior_model"

    config.save_model_path = "./result/" + f"{config.project}" + "/" +f"{config.run_name}"


    config.dataset = d(
        data_info_text_path = "/path/to/your/data/info/text/file",
        batch_size = 16,
        num_workers = 0
    )

    config.model = d(
        max_len = 512,
        embedding_plan = "MeanPooling",
        pretrained_lm_model="/path/to/your/pretrained/lm/model",
    )

    config.modelcheckpoint = d(
        dirpath = config.save_model_path,
        filename = '{epoch}-{val_loss:.2f}',
    )




    config.trainer = d(
        accelerator="auto",
        precision=16,
        max_epochs=10,
        gradient_clip_val = 0.0,
        val_interval = 1,
        eval_interval = 1,
        accum_iter = 4
    )

    return config