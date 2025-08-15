import argparse
import os
import pickle
import torch
from tqdm import tqdm
from datasets import load_from_disk, ClassLabel
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification

# Import necessary custom classes from your project
from scDMC_finetune_tokenizer import TranscriptomeTokenizer
from scDMC_data_collator import GHLDataCollatorForCellClassification as DataCollatorForCellClassification


# ==============================================================================
# 1. Define command-line arguments for the script
# ==============================================================================
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate cell embeddings (hidden states) from a fine-tuned model.")
    parser.add_argument("--model_path", type=str, default='../models/finetuned_models/zheng68k_tk60534',
                        help="Path to the directory of the [fine-tuned] model.")
    parser.add_argument("--data_path", type=str, default='../datasets/finetune/zheng68k_tk60534',
                        help="Path to the directory of the dataset to be processed.")
    parser.add_argument("--output_dir", type=str, default='./output/embeddings/',
                        help="[Base directory] for storing the output results.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size used for prediction. If you encounter an OOM error, try reducing this value.")
    parser.add_argument("--device", type=int, default=0,
                        help="Specify the GPU device number to use.")
    return parser.parse_args()


def main():
    """
    Main function of the script: load model and data, perform prediction using a manual loop, and save the results.
    """
    args = parse_args()

    # ==============================================================================
    # 2. Set up environment and dynamically construct output paths
    # ==============================================================================
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = os.path.basename(os.path.dirname(args.model_path.rstrip('/')))
    dataset_name = os.path.basename(args.data_path.rstrip('/'))

    final_output_dir = os.path.join(args.output_dir, model_name, dataset_name)
    os.makedirs(final_output_dir, exist_ok=True)

    output_file_path = os.path.join(final_output_dir, 'embeddings.pickle')

    print("=" * 60)
    print("Starting to generate embeddings from the fine-tuned model (memory-optimized mode)...")
    print(f"  - Model Path: {args.model_path}")
    print(f"  - Data Path: {args.data_path}")
    print(f"  - Output Directory: {final_output_dir}")
    print(f"  - Device: {device}")
    print("=" * 60)

    # ==============================================================================
    # 3. Load the model and move it to the device
    # ==============================================================================
    print("\nLoading model...")
    model = BertForSequenceClassification.from_pretrained(
        args.model_path,
        output_hidden_states=True,
        output_attentions=False
    ).to(device)
    model.eval()  # Switch to evaluation mode

    # ==============================================================================
    # 4. Load the dataset and process labels
    # ==============================================================================
    print("\nLoading dataset...")
    dataset = load_from_disk(args.data_path)

    label_column_found = False
    for label_name in ["celltype", "cell_type", "str_labels", "labels", "label"]:
        if label_name in dataset.column_names:
            if label_name != "label":
                print(f"  - Renaming column '{label_name}' to 'label'.")
                dataset = dataset.rename_column(label_name, "label")
            label_column_found = True
            break

    if not label_column_found:
        raise ValueError("No possible label column found in the dataset.")

    if not isinstance(dataset.features['label'], ClassLabel):
        if isinstance(dataset[0]['label'], str):
            print("  - Detected labels in string format, starting auto-conversion to integer IDs...")
            dataset = dataset.class_encode_column("label")
            print("  - Label conversion complete.")

    # Save the label mapping
    label_map = dataset.features['label'].names
    label_map_path = os.path.join(final_output_dir, 'label_map.txt')
    with open(label_map_path, 'w') as f:
        for i, label_str in enumerate(label_map):
            f.write(f"{i}\t{label_str}\n")
    print(f"  - Label map saved to: {label_map_path}")

    # ==============================================================================
    # 5. Manually create DataLoader (using the correct custom DataCollator)
    # ==============================================================================
    print("\nInitializing data loader...")

    tokenizer = TranscriptomeTokenizer()

    # Use the exact same custom DataCollator as in your fine-tuning
    data_collator = DataCollatorForCellClassification(tokenizer=tokenizer)

    # We only need input_ids and label for the DataLoader
    dataset.set_format(type='torch', columns=['input_ids', 'label'])

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,  # Use the custom collate_fn
        shuffle=False,
        num_workers=4  # Can speed up data loading
    )

    # ==============================================================================
    # 6. Manually loop for prediction and progressively collect results
    # ==============================================================================
    print(f"\nStarting prediction on the dataset (total {len(dataset)} samples)...")

    all_cls_embeddings = []
    all_labels = []

    with torch.no_grad():  # Crucial: disable gradient calculation to save a lot of memory and computation
        for batch in tqdm(dataloader, desc="Processing Batches"):
            # The DataCollator already returns tensors, just move them to the device
            inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}

            # Model forward pass
            outputs = model(**inputs)

            # Extract hidden_states from the last layer
            last_hidden_state = outputs.hidden_states[-1]  # On GPU

            # Extract the embedding of the [CLS] token (index 0)
            cls_embeddings_batch = last_hidden_state[:, 0, :]  # Still on GPU

            # **Core step**: Move results to CPU and append to lists
            all_cls_embeddings.append(cls_embeddings_batch.cpu())
            all_labels.append(batch['labels'].cpu())  # The label key returned by DataCollator is 'labels'

    # Concatenate results from all batches into a single large tensor
    final_embeddings = torch.cat(all_cls_embeddings, dim=0).numpy()
    final_labels = torch.cat(all_labels, dim=0).numpy()

    print("  - Prediction complete.")
    print(f"  - Successfully extracted embeddings with shape: {final_embeddings.shape}")

    # ==============================================================================
    # 7. Save the results
    # ==============================================================================
    print(f"\nSaving results to: {output_file_path}")

    output_to_save = {
        'embeddings': final_embeddings,
        'labels': final_labels,
        'info': {
            'model_path': args.model_path,
            'data_path': args.data_path,
            'description': 'Embeddings are from the [CLS] token of the last hidden layer.'
        }
    }

    with open(output_file_path, 'wb') as f:
        pickle.dump(output_to_save, f)

    print("\n🎉 Success! Cell embeddings have been saved.")
    print(f"  - Output file: {output_file_path}")


if __name__ == "__main__":
    main()