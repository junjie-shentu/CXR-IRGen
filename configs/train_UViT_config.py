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
    config.run_name = "train_UViT"
    config.save_model_path = "./result/" + f"{config.project}" + "/" +f"{config.run_name}"

    config.dataset = d(
        data_info_text_path = "/path/to/your/dataset.csv",
        batch_size = 32,
        num_workers = 0
    )

    config.uvit = d(
        pretrained_uvit_path = "./assets/imagenet256_uvit_large.pth",
        img_size = 32,
        patch_size = 2,
        in_chans = 4,
        embed_dim = 1024,
        depth = 20,
        num_heads = 16,
        mlp_ratio = 4.,
        qkv_bias = False,
        norm_layer = nn.LayerNorm,
        mlp_time_embed = False,
        use_checkpoint=True,
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