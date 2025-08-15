import argparse
import os
import datetime
import random
import glob
import json
import logging
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, TrainerCallback
from transformers.trainer import unwrap_model
from transformers.utils import WEIGHTS_NAME
from safetensors.torch import save_model
from accelerate import Accelerator
from scDMC_bert import BertForMaskedLM
from loss import masked_language_modeling_loss, contrastive_loss
from bert_config import BertConfig


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Set logging level to show only warnings/errors
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.WARNING)
os.environ["PYTORCH_JIT_LOG_LEVEL"] = "ERROR"


# Set random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Get command line arguments
def get_args():
    parser = argparse.ArgumentParser(description="Pre-train BERT with multi-task setup (Accelerate version)")
    parser.add_argument("--dataset_path", type=str, default="../datasets/pretrain/pretrain_datasets/*.parquet",
                        help="Path to dataset directory")
    parser.add_argument("--vocab_size", type=int, default=60534, help="Vocabulary size")
    parser.add_argument("--pad_token_id", type=int, default=0, help="Padding token ID")
    parser.add_argument("--mask_token_id", type=int, default=1, help="[MASK] token ID")
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden size of the model")
    parser.add_argument("--num_attention_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_hidden_layers", type=int, default=12, help="Number of hidden layers")
    parser.add_argument("--intermediate_size", type=int, default=1024, help="Intermediate size of feedforward network")
    parser.add_argument("--max_position_embeddings", type=int, default=2048,
                        help="Buffer size for token type IDs (not used for position embeddings)")
    parser.add_argument("--alibi_starting_size", type=int, default=2048, help="Starting size for ALiBi bias")
    parser.add_argument("--mask_probability", type=float, default=0.15, help="Probability of masking tokens")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for optimizer")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay for optimizer")
    parser.add_argument("--output_dir", type=str, default="../models/training", help="Base directory to save trained pretrained_models")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log metrics every `logging_steps` steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Steps for gradient accumulation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--sharded_ddp", action="store_true", help="Use sharded ddp")
    parser.add_argument("--dataloader_num_workers", type=int, default=0,
                        help="Number of workers for adata loading (0 for main process only)")

    # Staged training related parameters
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to a checkpoint to resume training from")
    parser.add_argument("--dataset_parts", type=str, nargs="+",
                        help="List of paths to dataset parts to train on sequentially")
    parser.add_argument("--current_part", type=int, default=0, help="Index of current dataset part to train on")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode for dataset loading")

    # Learning rate adjustment related parameters
    parser.add_argument("--reduce_lr_on_new_part", action="store_true",
                        help="Reduce learning rate when starting a new dataset part")
    parser.add_argument("--lr_reduction_factor", type=float, default=0.7,
                        help="Factor to reduce learning rate by when moving to a new part")
    parser.add_argument("--min_learning_rate", type=float, default=1e-7,
                        help="Minimum learning rate below which no further reduction occurs")

    return parser.parse_args()


# Validate that the dataset contains the required fields
def validate_dataset(dataset):
    required_fields = ["input_ids_rank", "input_ids_epe", "expression_values"]
    missing_fields = [field for field in required_fields if field not in dataset.column_names]
    if missing_fields:
        raise ValueError(f"Dataset missing required fields: {', '.join(missing_fields)}")
    return True


# Modify CustomDataCollator to reduce memory usage
class CustomDataCollator:
    def __init__(self, pad_token_id, mask_token_id, mlm_probability):
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.mlm_probability = mlm_probability

    def __call__(self, features):
        # Create tensors directly on the GPU during batching
        batch_size = len(features)

        # Get the maximum sequence length to avoid excessive padding
        max_len1 = max(len(f["input_ids_rank"]) for f in features)
        max_len2 = max(len(f["input_ids_epe"]) for f in features)

        # Pre-allocate tensors - more efficient than dynamic growth
        input_ids1 = torch.full((batch_size, max_len1), self.pad_token_id, dtype=torch.long)
        labels1 = torch.full((batch_size, max_len1), -100, dtype=torch.long)
        attention_mask1 = torch.zeros((batch_size, max_len1), dtype=torch.long)

        input_ids2 = torch.full((batch_size, max_len2), self.pad_token_id, dtype=torch.long)
        labels2 = torch.full((batch_size, max_len2), -100, dtype=torch.long)
        attention_mask2 = torch.zeros((batch_size, max_len2), dtype=torch.long)
        expression_values = torch.zeros((batch_size, max_len2), dtype=torch.float)

        for i, feature in enumerate(features):
            # Process the first set of inputs
            seq_len1 = len(feature["input_ids_rank"])
            ids1 = torch.tensor(feature["input_ids_rank"], dtype=torch.long)
            input_ids1[i, :seq_len1] = ids1
            attention_mask1[i, :seq_len1] = 1

            # Create and apply mask - only process the valid sequence part
            mask1 = torch.bernoulli(torch.full((seq_len1,), self.mlm_probability)).bool()
            mask1 = mask1 & (ids1 != self.pad_token_id)

            # Process valid and padding parts separately
            labels1[i, :seq_len1] = ids1.clone()  # Copy the valid part
            labels1[i, :seq_len1][~mask1] = -100  # Apply mask to the valid part
            # The padding part is already set to -100 by default

            # Apply mask to input IDs
            input_ids1[i, :seq_len1][mask1] = self.mask_token_id

            # Process the second set of inputs - same fix
            seq_len2 = len(feature["input_ids_epe"])
            ids2 = torch.tensor(feature["input_ids_epe"], dtype=torch.long)
            input_ids2[i, :seq_len2] = ids2
            attention_mask2[i, :seq_len2] = 1

            # Create and apply mask
            mask2 = torch.bernoulli(torch.full((seq_len2,), self.mlm_probability)).bool()
            mask2 = mask2 & (ids2 != self.pad_token_id)

            # Process valid and padding parts separately
            labels2[i, :seq_len2] = ids2.clone()  # Copy the valid part
            labels2[i, :seq_len2][~mask2] = -100  # Apply mask to the valid part

            # Process expression values
            exp_vals = torch.tensor(feature["expression_values"], dtype=torch.float)
            expression_values[i, :seq_len2] = exp_vals
            expression_values[i, :seq_len2][~mask2] = 0.0  # Apply mask to the valid part

        return {
            "input_ids1": input_ids1,
            "labels1": labels1,
            "attention_mask1": attention_mask1,
            "input_ids2": input_ids2,
            "labels2": labels2,
            "attention_mask2": attention_mask2,
            "expression_values": expression_values
        }


# Custom Trainer callback
class LogCallback(TrainerCallback):
    def __init__(self, logging_steps):
        self.logging_steps = logging_steps  # Store logging_steps
        self.logged_steps = set()  # Used to track logged steps

    def on_log(self, args, state, control, logs=None, **kwargs):
        pass


# Momentum update callback
class MomentumUpdateCallback(TrainerCallback):
    def __init__(self, momentum=0.995):
        self.momentum = momentum

    def on_step_end(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer")
        if trainer is not None and hasattr(trainer, "update_momentum_encoder"):
            trainer.update_momentum_encoder(self.momentum)
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        # Clean up memory fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        return control


# Improved staged training state saving callback
class StageTrackingCallback(TrainerCallback):
    def __init__(self, output_dir, current_part, total_parts, status_filename="training_status.json"):
        self.output_dir = output_dir
        self.current_part = current_part
        self.total_parts = total_parts
        self.status_filename = status_filename
        self.best_loss = float('inf')
        self.steps_without_improvement = 0
        self.last_saved_checkpoint = None  # Track the last saved checkpoint

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log the current training state, can be used to determine if training has converged or needs to be stopped early"""
        if logs and "total_loss" in logs:
            current_loss = logs["total_loss"]

            # Update best loss and steps without improvement
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.steps_without_improvement = 0
            else:
                self.steps_without_improvement += 1

            # Update status file every 50 logging_steps
            if state.global_step % (50 * args.logging_steps) == 0 and args.local_rank in [-1, 0]:
                self._save_status(state, current_checkpoint=None)

    def on_save(self, args, state, control, **kwargs):
        """Save the current training state every time the model is saved"""
        if args.local_rank in [-1, 0]:
            # Get the path of the latest saved checkpoint
            checkpoints = sorted(glob.glob(os.path.join(self.output_dir, "checkpoint-*")),
                                 key=lambda x: int(x.split("-")[-1]))
            current_checkpoint = checkpoints[-1] if checkpoints else None

            # Record the last saved checkpoint
            if current_checkpoint:
                self.last_saved_checkpoint = current_checkpoint

            self._save_status(state, current_checkpoint)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        """When training ends, update the status to completed"""
        if args.local_rank in [-1, 0]:
            # Mark the current part as completed
            self._save_status(state, current_checkpoint=None, completed=True)
        return control

    def _save_status(self, state, current_checkpoint, completed=False):
        """Save training status to a file"""
        # Ensure the directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # If current_checkpoint is None but there is a known last checkpoint, use it
        if current_checkpoint is None and self.last_saved_checkpoint:
            current_checkpoint = self.last_saved_checkpoint

        # Prepare status information
        status = {
            "current_part": self.current_part,
            "total_parts": self.total_parts,
            "global_step": state.global_step,
            "epoch": state.epoch,
            "best_loss": self.best_loss,
            "steps_without_improvement": self.steps_without_improvement,
            "last_checkpoint": current_checkpoint,
            "completed": completed,
            "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Save status
        status_path = os.path.join(self.output_dir, self.status_filename)
        with open(status_path, "w") as f:
            json.dump(status, f, indent=2)

        if completed:
            print(f"\nTraining completed for part {self.current_part + 1}/{self.total_parts}")
            print(f"Status saved to {status_path}")


# Get training status
def get_training_status(output_dir):
    """Get the current training status"""
    status_path = os.path.join(output_dir, "training_status.json")
    if os.path.exists(status_path):
        try:
            with open(status_path, "r") as f:
                return json.load(f)
        except:
            pass
    return None


# Custom Trainer
class MultiTaskTrainer(Trainer):
    def __init__(self, *args, momentum_encoder=None, temperature=0.07, alpha=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum_encoder = momentum_encoder
        self.temperature = temperature
        self.alpha = alpha

    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        if self.momentum_encoder is not None:
            self.momentum_encoder.to(self.model.device)
            self.momentum_encoder.eval()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Override compute_loss to let the built-in training_step of Trainer handle automatic gradient scaling, mixed precision, and backpropagation.
        """
        input_ids1 = inputs["input_ids1"]
        attention_mask1 = inputs["attention_mask1"]
        labels1 = inputs["labels1"]

        input_ids2 = inputs["input_ids2"]
        attention_mask2 = inputs["attention_mask2"]
        labels2 = inputs["labels2"]

        expression_values = inputs["expression_values"]

        # Note: Do not delete inputs, as these variables are still needed for subsequent calculations
        # Use pass-through instead of pop to get inputs

        outputs1 = model(input_ids=input_ids1, attention_mask=attention_mask1)
        loss1 = masked_language_modeling_loss(outputs1.logits, labels1, attention_mask1)

        outputs2 = model(input_ids=input_ids2, attention_mask=attention_mask2)
        loss2 = masked_language_modeling_loss(outputs2.logits, labels2, attention_mask2, weight=expression_values)

        # Momentum encoder output (forward pass only)
        with torch.no_grad():
            if self.momentum_encoder is not None:
                # Combine batches to reduce the number of forward passes
                combined_ids = torch.cat([input_ids1, input_ids2], dim=0)
                combined_mask = torch.cat([attention_mask1, attention_mask2], dim=0)
                combined_outputs = self.momentum_encoder(combined_ids, attention_mask=combined_mask)

                # Split the results
                batch_size = input_ids1.size(0)
                momentum_features1 = combined_outputs.hidden_states[:batch_size]
                momentum_features2 = combined_outputs.hidden_states[batch_size:]
            else:
                # If there is no momentum encoder, use None as a placeholder
                momentum_features1, momentum_features2 = None, None

        # Final layer hidden states of the current model
        features1 = outputs1.hidden_states
        features2 = outputs2.hidden_states

        # Dropout
        features1 = F.dropout(features1, p=0.1, training=model.training)
        features2 = F.dropout(features2, p=0.1, training=model.training)

        # If a momentum encoder exists, apply the same dropout
        if momentum_features1 is not None:
            momentum_features1 = F.dropout(momentum_features1, p=0.1, training=model.training)
        if momentum_features2 is not None:
            momentum_features2 = F.dropout(momentum_features2, p=0.1, training=model.training)

        # Calculate contrastive loss (only if momentum encoder exists)
        if momentum_features1 is not None and momentum_features2 is not None:
            contrastive_loss_value1 = contrastive_loss(features1[:, 0, :], momentum_features2[:, 0, :],
                                                       self.temperature)
            contrastive_loss_value2 = contrastive_loss(features2[:, 0, :], momentum_features1[:, 0, :],
                                                       self.temperature)
            total_contrastive_loss = (contrastive_loss_value1 + contrastive_loss_value2) / 2
        else:
            total_contrastive_loss = torch.tensor(0.0, device=self.model.device)

        total_loss = loss1 + loss2 + self.alpha * total_contrastive_loss

        # Log each loss value every args.logging_steps
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "total_loss": total_loss.item(),
                "loss1": loss1.item(),
                "loss2": loss2.item(),
                "contrastive_loss": total_contrastive_loss.item(),
                "batch_size": self.args.per_device_train_batch_size,  # Log the current batch size
                "learning_rate": self.optimizer.param_groups[0]['lr']  # Log the current learning rate
            })

        # Periodically clean up GPU memory
        if self.state.global_step % 500 == 0:  # Clean up every 500 steps
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return (total_loss, {"loss1": loss1, "loss2": loss2,
                             "contrastive_loss": total_contrastive_loss}) if return_outputs else total_loss

    def update_momentum_encoder(self, momentum=0.995):
        if self.momentum_encoder is None:
            return

        # Use buffer update to reduce memory peak
        with torch.no_grad():
            for param, mom_param in zip(self.model.parameters(), self.momentum_encoder.parameters()):
                mom_param.data.mul_(momentum).add_(param.data, alpha=1 - momentum)


def main():
    args = get_args()
    accelerator = Accelerator()
    set_seed(args.seed)

    # Monkey patch Trainer's _save method to use save_model instead of save_file
    def patched_save(self, output_dir=None):
        """Use safetensors.torch.save_model instead of save_file to handle shared weights"""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Save model checkpoint
        if self.args.local_rank in [-1, 0]:
            checkpoint_name = f"checkpoint-{self.state.global_step}"
            checkpoint_output_dir = os.path.join(output_dir, checkpoint_name)
            if not os.path.exists(checkpoint_output_dir):
                os.makedirs(checkpoint_output_dir, exist_ok=True)  # Create directory only if it doesn't exist

            # Unwrap the model and use save_model instead of save_file
            model_to_save = unwrap_model(self.model)

            # Use safetensors.torch.save_model to handle shared weights
            save_model(
                model_to_save,
                os.path.join(checkpoint_output_dir, WEIGHTS_NAME.replace(".bin", ".safetensors")),
                metadata={"format": "pt"}
            )

            # Save other necessary files - use processing_class instead of tokenizer
            if hasattr(self, "processing_class") and self.processing_class is not None:
                self.processing_class.save_pretrained(checkpoint_output_dir)

            # Save training arguments
            torch.save(self.args, os.path.join(checkpoint_output_dir, "training_args.bin"))

            # Subsequent original functionality
            if self.state.is_hyper_param_search:
                self._save_hp_search_to_json(checkpoint_output_dir)

            # Save optimizer and learning rate scheduler states
            if self.args.should_save:
                torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_output_dir, "optimizer.pt"))
                torch.save(self.lr_scheduler.state_dict(), os.path.join(checkpoint_output_dir, "scheduler.pt"))

            # Possible additional save steps
            self._rotate_checkpoints(use_mtime=True, output_dir=output_dir)

            return checkpoint_output_dir

    # Apply monkey patch
    Trainer._save = patched_save
    print("Applied patched save method to handle shared weights with safetensors.torch.save_model")

    # Handle staged training
    if args.dataset_parts and args.current_part < len(args.dataset_parts):
        current_dataset_path = args.dataset_parts[args.current_part]
        print(f"Training on dataset part {args.current_part + 1}/{len(args.dataset_parts)}: {current_dataset_path}")
        total_parts = len(args.dataset_parts)
    else:
        current_dataset_path = args.dataset_path
        total_parts = 1
        args.current_part = 0

    # Load and validate dataset
    try:
        # Default to standard loading method (non-streaming)
        print("Using standard mode for dataset loading")
        train_dataset = load_dataset('parquet', data_files=current_dataset_path, split="train")
        validate_dataset(train_dataset)
        # Get the accurate dataset size
        num_rows = len(train_dataset)
        print(f"Dataset size: {num_rows} examples")
    except Exception as e:
        print(f"Error loading or validating dataset: {e}")
        # Add appropriate error handling logic
        raise

    # Determine if learning rate needs to be adjusted
    should_reduce_lr = False

    # Only reduce learning rate if requested and it's a new data part
    if args.resume_from_checkpoint and args.reduce_lr_on_new_part:
        # Check if resuming from the same dataset part
        status = get_training_status(args.output_dir)
        if status and status.get("current_part", -1) < args.current_part:
            # Only reduce learning rate when switching to a new part
            should_reduce_lr = True
            print("Starting new dataset part, will reduce learning rate")
        else:
            print("Resuming same dataset part, keeping original learning rate")

    # Adjust learning rate, ensuring it doesn't fall below the minimum
    if should_reduce_lr:
        new_lr = args.learning_rate * args.lr_reduction_factor
        if new_lr >= args.min_learning_rate:
            args.learning_rate = new_lr
            print(f"Reduced learning rate to {args.learning_rate}")
        else:
            args.learning_rate = args.min_learning_rate
            print(f"Learning rate reached minimum value of {args.min_learning_rate}")

    # Initialize model configuration and main model
    config = BertConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_hidden_layers,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_position_embeddings,
        pad_token_id=args.pad_token_id,
        alibi_starting_size=args.alibi_starting_size,
        type_vocab_size=1,
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.02,
        initializer_range=0.02,
        hidden_act="gelu",
        tie_word_embeddings=True,
        output_hidden_states=True
    )

    # Main model loading
    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        print(f"Loading model from checkpoint: {checkpoint_path}")
        config = BertConfig.from_pretrained(checkpoint_path)
        if not hasattr(config, "alibi_starting_size"):
            config.alibi_starting_size = args.alibi_starting_size
        model = BertForMaskedLM.from_pretrained(checkpoint_path, config=config)

        # Do not use training state for resumption - save original checkpoint path for logging
        original_checkpoint_path = checkpoint_path
        args.resume_from_checkpoint = None

        # Log staged training information
        part_info_path = os.path.join(args.output_dir, "part_training_info.json")
        part_info = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "part": args.current_part,
            "model_loaded_from": checkpoint_path,
            "learning_rate": args.learning_rate
        }
        os.makedirs(os.path.dirname(part_info_path), exist_ok=True)

        # Append staged training information
        with open(part_info_path, "a") as f:
            f.write(json.dumps(part_info) + "\n")

        print(f"Starting new training (Part {args.current_part}) with weights from: {checkpoint_path}")
        print(f"Using learning rate: {args.learning_rate}")
    else:
        # Train from scratch
        model = BertForMaskedLM(config=config)
        original_checkpoint_path = None

    # Enable gradient checkpointing for the main model
    model.gradient_checkpointing_enable()

    # Momentum encoder always copies parameters from the main model
    momentum_encoder = BertForMaskedLM(config=config)
    momentum_encoder.load_state_dict(model.state_dict())
    print("Momentum encoder initialized directly from current main model")

    model = accelerator.prepare_model(model)
    momentum_encoder = accelerator.prepare_model(momentum_encoder)
    # Move momentum_encoder to the same device
    momentum_encoder.to(accelerator.device)
    momentum_encoder.eval()

    # Freeze momentum encoder parameters
    for param in momentum_encoder.parameters():
        param.requires_grad = False

    # Calculate the number of parameters in the model
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Convert the number of parameters to MB
    total_params_mb = total_params * 4 / (1024 ** 2)

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"scDMC_params{int(total_params_mb)}MB_hid_size{args.hidden_size}_heads{args.num_attention_heads}_layers{args.num_hidden_layers}_{current_time}"

    if args.resume_from_checkpoint:
        # If resuming training, use the original output directory
        output_dir = os.path.dirname(
            original_checkpoint_path) if "checkpoint-" in original_checkpoint_path else original_checkpoint_path
        if not os.path.exists(output_dir):
            output_dir = args.output_dir
    else:
        # Use a new directory for new training
        output_dir = f"{args.output_dir}/{model_name}"

    model_save_path = os.path.join(output_dir)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        eval_strategy="no",  # Disable evaluation
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=int(np.floor(num_rows / (args.batch_size * args.gradient_accumulation_steps) / 8)),
        save_total_limit=3,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=True,  # Let Trainer know to use mixed precision
        fp16_full_eval=False,
        remove_unused_columns=False,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",  # Use cosine decay scheduler
        dataloader_num_workers=args.dataloader_num_workers,  # Use the number of worker processes specified by the command line argument
        dataloader_pin_memory=True,  # Pin data in memory to speed up GPU transfer
        optim="adamw_torch",  # Use PyTorch's native AdamW optimizer
        report_to=["tensorboard"],  # Enable TensorBoard integration
        logging_dir=os.path.join(output_dir, "logs"),  # TensorBoard log directory
    )

    # Custom data collator
    train_data_collator = CustomDataCollator(
        pad_token_id=args.pad_token_id,
        mask_token_id=args.mask_token_id,
        mlm_probability=args.mask_probability
    )

    # Prepare callback list
    callbacks = [
        LogCallback(logging_steps=args.logging_steps),
        MomentumUpdateCallback(momentum=0.995),
        StageTrackingCallback(output_dir=output_dir,
                              current_part=args.current_part,
                              total_parts=total_parts),
    ]

    # Define Trainer
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=train_data_collator,
        callbacks=callbacks,
        momentum_encoder=momentum_encoder,
        temperature=0.07,
        alpha=1.0,
    )

    # Resume from checkpoint or start from scratch, with exception handling
    try:
        # No longer passing resume_from_checkpoint parameter, starting a new training loop from scratch
        trainer.train()
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"CUDA OOM error: {e}")
            # Reduce batch size and retry
            new_batch_size = max(1, trainer.args.per_device_train_batch_size // 2)
            print(f"Reduced batch size to {new_batch_size} and retrying...")
            trainer.args.per_device_train_batch_size = new_batch_size
            torch.cuda.empty_cache()
            trainer.train()
        else:
            raise

    torch.cuda.empty_cache()  # Clean up GPU memory
    torch.cuda.reset_peak_memory_stats()  # Reset GPU peak memory stats

    # Check if saving the model on the main process (also available in Accelerate as accelerator.is_local_main_process)
    if accelerator.is_local_main_process:
        # Unwrap the model and save (use save_model to handle shared weights)
        unwrapped_model = accelerator.unwrap_model(trainer.model)
        unwrapped_model.save_pretrained(model_save_path, safe_serialization=False)  # Use safe_serialization

        # No longer saving the momentum encoder as it is always copied from the main model
        print(f"Model saved to {model_save_path}")
        print("No need to save momentum encoder as it will be initialized from the main model.")

        # If it's part of staged training, save a marker that the current part is complete
        if args.dataset_parts:
            status_path = os.path.join(output_dir, "training_status.json")
            if os.path.exists(status_path):
                with open(status_path, "r") as f:
                    status = json.load(f)

                status["completed"] = True
                status["final_model_path"] = model_save_path
                # Remove momentum encoder path information
                status["completed_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                with open(status_path, "w") as f:
                    json.dump(status, f, indent=2)

                print(f"Updated training status at {status_path}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Only return the model save path, no longer the momentum encoder path
    return model_save_path


if __name__ == "__main__":
    # Clean up memory fragmentation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    main()