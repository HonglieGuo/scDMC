# Model Download Instructions

This directory is used to store the model weight files required for the project. Due to the large size of the model files, they are not directly included in the GitHub repository.

This directory structure is divided into two parts:
*   `pretrained_models`: For storing the base pre-trained models.
*   `finetuned_models`: For storing models generated after fine-tuning with downstream task datasets.

---

## 1. Pre-trained Models

You need to download our pre-trained base model from the Hugging Face Hub and place it under the `pretrained_models/scDMC` path.

### Download Steps

1.  **Visit the Hugging Face Model Repository**:
    Please click the following link to access our official model repository:
    [https://huggingface.co/Honglie/scDMC-models](https://huggingface.co/Honglie/scDMC-models) *(Please note: If your model repository address is different, please update the link here)*

2.  **Download the Model Files**:
    On the Hugging Face page, please download all files in the repository, especially `pytorch_model.bin`, `config.json`, etc.

3.  **Place the Files**:
    Place all the downloaded model-related files **directly into the `pretrained_models/scDMC` folder**.

---

## 2. Finetuned Models

The `finetuned_models` folder is used to store the models that you produce after fine-tuning with downstream task datasets.