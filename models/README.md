# 模型下载说明 (Model Download Instructions)

此目录用于存放项目所需的模型权重文件。由于模型文件体积较大，并未直接包含在 GitHub 仓库中。

此目录结构分为两部分：
*   `pretrained_models`: 用于存放基础的预训练模型。
*   `finetuned_models`: 用于存放使用下游任务数据集微调后生成的模型。

---

## 1. 预训练模型 (Pre-trained Models)

您需要从 Hugging Face Hub 下载我们预训练好的基础模型，并将其放置在 `pretrained_models/scDMC` 路径下。

### 下载步骤

1.  **访问 Hugging Face 模型仓库**：
    请点击以下链接访问我们的官方模型仓库：
    [https://huggingface.co/Honglie/scDMC-models](https://huggingface.co/Honglie/scDMC-models)  *(请注意：如果您的模型仓库地址不同，请在此处更新链接)*

2.  **下载模型文件**：
    在 Hugging Face 页面上，请下载仓库中的所有文件，特别是 `pytorch_model.bin`, `config.json` 等。

3.  **放置文件**：
    将所有下载好的模型相关文件**直接放入 `pretrained_models/scDMC` 文件夹内**。

---

## 2. 微调后模型 (Finetuned Models)

`finetuned_models` 文件夹用于存放您自己使用下游任务数据集进行微调后所产出的模型。
