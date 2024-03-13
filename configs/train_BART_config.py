import ml_collections
import torch


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.device = torch.device("cuda")
    config.random_seed = 1234
    config.project = "CXR-IRGen"
    config.run_name = "train_BART"
    config.save_model_path = "./result/" + f"{config.project}" + "/" +f"{config.run_name}"

    config.dataset = d(
        data_path = "./dataset/data_training_LM.csv",
        batch_size = 8,
        num_workers = 0
    )

    config.model = d(
        max_len = 512,
        model_name="facebook/bart-base",
        embedding_plan = "MeanPooling"
    )

    config.modelcheckpoint = d(
        dirpath = config.save_model_path,
        filename = '{epoch}-{val_loss:.2f}',
    )

    config.trainer = d(
        accelerator="auto",
        precision=32,
        max_epochs=20,
        val_interval = 0.1,
    )

    return config