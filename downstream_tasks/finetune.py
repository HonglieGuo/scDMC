from collections import Counter
import pickle
import random
import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from transformers import BertForSequenceClassification
from transformers import Trainer
from transformers.training_args import TrainingArguments
import argparse
from scDMC_data_collator import GHLDataCollatorForCellClassification as DataCollatorForCellClassification
import os
import datetime
from scDMC_finetune_tokenizer import TranscriptomeTokenizer

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_model_path", type=str, default='../models/pretrained_models/scDMC')
parser.add_argument("--finetune_data_path", type=str, default='../datasets/finetune/zheng68k_tk60534')
parser.add_argument("--output_path", type=str, default=None)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--finetune_mode", type=str, choices=["full", "classifier_only"], default="full",
                    help="Whether to finetune the entire model or just the classification head")
parser.add_argument("--unfreeze_layers", type=int, default=0,
                    help="Number of top encoder layers to unfreeze (only used when finetune_mode='classifier_only')")
parser.add_argument("--classifier_learning_rate", type=float, default=1e-4,
                    help="Learning rate for classifier only")
parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for full")
args = parser.parse_args()

pretrained_model_path = args.pretrained_model_path
finetune_data_path = args.finetune_data_path
output_path = '../models/finetuned_models/' + pretrained_model_path.split('/')[-1] + '/' + finetune_data_path.split('/')[-1].split('.')[0] + '/'
output_path = args.output_path if args.output_path else output_path
epochs = args.epochs
test_size = args.test_size

GPU_NUMBER = [args.device]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
os.environ["NCCL_DEBUG"] = "INFO"


# Set random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(args.seed)


# Configure model training strategy (full model fine-tuning or classifier-only fine-tuning)
def configure_model_for_training(model, args):
    # By default, all parameters require gradients
    for param in model.parameters():
        param.requires_grad = True

    if args.finetune_mode == "classifier_only":
        print(f"Using classifier-only fine-tuning with {args.unfreeze_layers} unfrozen encoder layers")

        # First, freeze all BERT encoder parameters
        for param in model.bert.parameters():
            param.requires_grad = False

        # Check how many top layers we want to unfreeze
        if args.unfreeze_layers > 0:
            # Get the total number of layers
            total_layers = len(model.bert.encoder.layer)
            # Ensure not to exceed the total number of layers
            unfreeze_layers = min(args.unfreeze_layers, total_layers)

            # Unfreeze the top N layers
            for i in range(total_layers - unfreeze_layers, total_layers):
                for param in model.bert.encoder.layer[i].parameters():
                    param.requires_grad = True
                print(f"Unfrozen layer {i}")

        # The classifier is always trainable
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        print("Using full model fine-tuning")

    # Print the number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params:.2%} of total)")

    return model


# Custom optimizer
def get_optimizer_with_params(model, learning_rate, weight_decay, finetune_mode, classifier_learning_rate):
    no_decay = ["bias", "LayerNorm.weight"]

    if finetune_mode == "classifier_only":
        # Two groups of parameters: classifier parameters and other (possibly unfrozen) parameters
        classifier_params = [
            (n, p) for n, p in model.named_parameters()
            if "classifier" in n and p.requires_grad
        ]

        unfrozen_backbone_params = [
            (n, p) for n, p in model.named_parameters()
            if "classifier" not in n and p.requires_grad
        ]

        # Two groups of parameters, two learning rates
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in classifier_params if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": classifier_learning_rate,
            },
            {
                "params": [p for n, p in classifier_params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": classifier_learning_rate,
            },
            {
                "params": [p for n, p in unfrozen_backbone_params if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": learning_rate,
            },
            {
                "params": [p for n, p in unfrozen_backbone_params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": learning_rate,
            },
        ]
    else:
        # Full model fine-tuning, single learning rate
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer


# Custom Trainer
class CellClassificationTrainer(Trainer):
    def __init__(self, *args, finetune_mode="full", classifier_learning_rate=1e-4, unfreeze_layers=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.finetune_mode = finetune_mode
        self.classifier_learning_rate = classifier_learning_rate
        self.unfreeze_layers = unfreeze_layers

    def create_optimizer(self):
        """Create a custom optimizer"""
        if self.optimizer is None:
            # Pass custom parameters instead of the args object
            self.optimizer = get_optimizer_with_params(
                self.model,
                learning_rate=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                finetune_mode=self.finetune_mode,
                classifier_learning_rate=self.classifier_learning_rate
            )
        return self.optimizer


# %%
dataset = load_from_disk(finetune_data_path)

trainset_organ_shuffled = dataset.shuffle(seed=args.seed)
for label_name in ["celltype", "cell_type", "str_labels", "labels", "label"]:
    if label_name in trainset_organ_shuffled.column_names:
        break
if label_name != "label":
    trainset_organ_shuffled = trainset_organ_shuffled.rename_column(label_name, "label")
target_names = list(Counter(trainset_organ_shuffled["label"]).keys())
target_name_id_dict = dict(zip(target_names, [i for i in range(len(target_names))]))

# Save label mapping
os.makedirs(output_path, exist_ok=True)
with open(os.path.join(output_path, "label_map.txt"), "w") as f:
    for label, idx in target_name_id_dict.items():
        f.write(f"{idx}\t{label}\n")


# change labels to numerical ids
def classes_to_ids(example):
    example["label"] = target_name_id_dict[example["label"]]
    return example

labeled_trainset = trainset_organ_shuffled.map(classes_to_ids, num_proc=16)
train_size = round(len(labeled_trainset) * (1 - test_size))
labeled_train_split = labeled_trainset.select([i for i in range(0, train_size)])
labeled_eval_split = labeled_trainset.select([i for i in range(train_size, len(labeled_trainset))])
trained_labels = list(Counter(labeled_train_split["label"]).keys())


def if_trained_label(example):
    return example["label"] in trained_labels

labeled_eval_split_subset = labeled_eval_split.filter(if_trained_label, num_proc=16)

train_set = labeled_train_split
eval_set = labeled_eval_split_subset
label_name_id = target_name_id_dict


# %%
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1) if not isinstance(pred.predictions, tuple) else pred.predictions[0].argmax(-1)

    # Print prediction distribution information
    unique_preds, pred_counts = np.unique(preds, return_counts=True)
    unique_labels, label_counts = np.unique(labels, return_counts=True)

    print("\n===== PREDICTION DISTRIBUTION =====")
    print(f"Unique predictions: {unique_preds}")
    print(f"Prediction counts: {pred_counts}")
    print(f"Unique labels: {unique_labels}")
    print(f"Label counts: {label_counts}")
    print("===================================\n")

    # Calculate confusion matrix
    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:")
    print(cm)


    # Use zero_division=0 to prevent division by zero errors/warnings
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)

    # **** Use the correct functions to calculate precision and recall ****
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    # ******************************************************

    # Calculate weighted average F1 score (this can be kept, or removed if not needed)
    weighted_f1 = f1_score(labels, preds, average='weighted', zero_division=0)

    print(f"\nCalculating metrics:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    print("\n")


    # Return a dictionary containing the correct metrics
    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'precision': precision,
        'recall': recall,
        'weighted_f1': weighted_f1     # Optionally include weighted F1
    }
# %%
organ_trainset = train_set
organ_evalset = eval_set
organ_label_dict = label_name_id

# set logging steps
logging_steps = round(len(organ_trainset) / args.batch_size / 10)

# reload pretrained model
model = BertForSequenceClassification.from_pretrained(pretrained_model_path,
                                                      num_labels=len(organ_label_dict.keys()),
                                                      output_attentions=False,
                                                      output_hidden_states=False)
# Print model structure
print("\n===== MODEL STRUCTURE =====")
print(model)

# Handle vocabulary size mismatch
tk = TranscriptomeTokenizer()
config = model.bert.config
print(f"vocab_size from config: {config.vocab_size}")
print(f"len(tk.gene_token_dict): {len(tk.gene_token_dict)}")
# if config.vocab_size == len(tk.gene_token_dict) - 1:
#     embedding_layer = nn.Embedding(config.vocab_size + 1, config.hidden_size, padding_idx=config.pad_token_id)
#     # Add 1 to reserve an extra index for the special padding symbol. For example, padding_idx is used in model training to mark padded words, so an extra index position is needed.
#     for param, param_pretrain in zip(embedding_layer.parameters(), model.bert.embeddings.word_embeddings.parameters()):
#         param.adata[:-1] = param_pretrain.adata
#     model.bert.embeddings.word_embeddings = embedding_layer
# elif config.vocab_size != len(tk.gene_token_dict):
#     raise Exception("Vocab size does not match.")

# Configure model training strategy
model = configure_model_for_training(model, args)

# Set the current time as part of the model name
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# Build the model directory name including experiment parameters
model_name = f"mlm_{args.finetune_mode}"
if args.finetune_mode == "classifier_only":
    model_name += f"_unfreeze{args.unfreeze_layers}"
model_name += f"_{current_time}"

output_dir = os.path.join(output_path, model_name)
os.makedirs(output_dir, exist_ok=True)

# ensure not overwriting previously saved model
saved_model_test = os.path.join(output_dir, f"pytorch_model.bin")
if os.path.isfile(saved_model_test) == True:
    raise Exception("Model already saved to this directory.")

# set training arguments
training_args = {
    "learning_rate": args.learning_rate,
    "do_train": True,
    "do_eval": True,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",  # Ensure save strategy matches evaluation strategy
    "save_total_limit": 1,
    "logging_steps": logging_steps,
    "group_by_length": True,
    "length_column_name": "length",
    "disable_tqdm": False,
    "lr_scheduler_type": "linear",
    "warmup_steps": 500,
    # "warmup_ratio": 0.05,
    "weight_decay": 0.075,
    "per_device_train_batch_size": args.batch_size,
    "per_device_eval_batch_size": args.batch_size,
    "num_train_epochs": epochs,
    "load_best_model_at_end": True,
    "metric_for_best_model": "macro_f1",
    "output_dir": output_dir,
    "report_to": "tensorboard",
}

training_args_init = TrainingArguments(**training_args)

# Create custom Trainer
trainer = CellClassificationTrainer(
    model=model,
    args=training_args_init,
    data_collator=DataCollatorForCellClassification(tokenizer=tk),
    tokenizer=tk,
    train_dataset=organ_trainset,
    eval_dataset=organ_evalset,
    compute_metrics=compute_metrics,
    finetune_mode=args.finetune_mode,
    classifier_learning_rate=args.classifier_learning_rate,
    unfreeze_layers=args.unfreeze_layers,
)

# Print experiment settings
print("=" * 50)
print("Experiment Configuration:")
print(f"Fine-tuning mode: {args.finetune_mode}")
if args.finetune_mode == "classifier_only":
    print(f"Unfreeze top {args.unfreeze_layers} layers")
    print(f"Classifier learning rate: {args.classifier_learning_rate}")
print(f"Main learning rate: {args.learning_rate}")
print("=" * 50)

# Train the model
print("Starting fine-tuning...")
trainer.train()

# Evaluate the model
print("Evaluating model...")
predictions = trainer.predict(organ_evalset)
with open(f"{output_dir}/predictions.pickle", "wb") as fp:
    pickle.dump(predictions, fp)
trainer.save_metrics("eval", predictions.metrics)
trainer.save_model(output_dir)

# Save training parameters and evaluation results
with open(os.path.join(output_dir, "training_info.txt"), "w") as f:
    f.write(f"Fine-tuning mode: {args.finetune_mode}\n")
    f.write(f"Unfrozen layers: {args.unfreeze_layers if args.finetune_mode == 'classifier_only' else 'all'}\n")
    f.write(f"Learning rate: {args.learning_rate}\n")
    if args.finetune_mode == "classifier_only":
        f.write(f"Classifier learning rate: {args.classifier_learning_rate}\n")
    f.write(f"Evaluation results:\n")
    for metric, value in predictions.metrics.items():
        f.write(f"  {metric}: {value}\n")