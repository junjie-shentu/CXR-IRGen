# CXR-IRGen: An Integrated Vision and Language Model for the Generation of Clinically Accurate Chest X-Ray Image-Report Pairs (WACV 2024)

## Abstract
>Chest X-Ray (CXR) images play a crucial role in clinical practice, providing vital support for diagnosis and treatment. Augmenting the CXR dataset with synthetically generated CXR images annotated with radiology reports can enhance the performance of deep learning models for various tasks. However, existing studies have primarily focused on generating unimodal data of either images or reports. In this study, we propose an integrated model, \textit{CXR-IRGen}, designed specifically for generating CXR image-report pairs. Our model follows a modularized structure consisting of a vision module and a language module. Notably, we present a novel prompt design for the vision module by combining both text embedding and image embedding of a reference image. Additionally, we propose a new CXR report generation model as the language module, which effectively leverages a large language model and self-supervised learning strategy. Experimental results demonstrate that our new prompt is capable of improving the general quality (FID) and clinical efficacy (AUROC) of the generated images, with average improvements of 15.84\% and 1.84\%, respectively. Moreover, the proposed CXR report generation model outperforms baseline models in terms of clinical efficacy ($F_1$ score) and exhibits a high-level alignment of image and text, as the best $F_1$ score of our model is 6.93\% higher than the state-of-the-art CXR report generation model.

## Usage
### Training
1. Train the vision module (both Stable diffusion and U-ViT are available)
  python train_diffusion_clipSD.py
  python train_diffusion_clipUViT.py
3. Train the language module
  python train_bart.py
4. Train the language module
