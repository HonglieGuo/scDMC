import argparse
import os
import random
import datetime
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.utils.rnn as rnn_utils
from datasets import load_from_disk
from transformers import BertModel
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt

try:
    from scDMC_finetune_tokenizer import TranscriptomeTokenizer
except ImportError:
    print("Error: Unable to import TranscriptomeTokenizer.")
    print("Please ensure scDMC_finetune_tokenizer.py exists and is in the Python path.")
    exit(1)

parser = argparse.ArgumentParser(description="Generate embeddings using a fine-tuned model for clustering evaluation and visualization.")
parser.add_argument("--finetuned_model_path", type=str, default='../models/finetuned_models/pbmc10k_tk60534',
                    help="Path to the fine-tuned base BERT model for embedding extraction (e.g., the directory containing pytorch_model.bin).")
parser.add_argument("--dataset_path", type=str, default='../datasets/finetune/pbmc10k_tk60534',
                    help="Path to the complete, preprocessed, and tokenized .dataset file (should contain an 'input_ids' column).")
parser.add_argument("--output_dir", type=str, default='./output/cluster/',
                    help="Base output directory where all evaluation results will be saved in a specific subdirectory.")
parser.add_argument("--label_col_name", type=str, default="cell_type",
                    help="Column name in the dataset containing the true cell type labels. If 'None' or the column does not exist, evaluation based on true labels will not be performed.")
parser.add_argument("--device", type=int, default=0, help="GPU device number (if available).")
parser.add_argument("--embedding_batch_size", type=int, default=32, help="Batch size for generating embeddings.")
parser.add_argument("--seed", type=int, default=1, help="Random seed.")
parser.add_argument("--leiden_resolution", type=float, default=0.2, help="Resolution parameter for Leiden clustering.")

args = parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")


def log_message(message, log_file_path):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(full_message + "\n")


def get_cell_embeddings(
        tokenized_data,
        model: torch.nn.Module,
        pad_token_id: int,
        batch_size: int = 32,
        device: str = "cpu",
        log_func=print
):
    model.eval()
    all_embeddings_list = []
    num_cells_to_process = len(tokenized_data)

    if 'input_ids' not in tokenized_data.features:
        log_func("Error: 'input_ids' column not found in tokenized_data.features.")
        return None

    log_func(f"Starting to generate embeddings for {num_cells_to_process} cells...")
    for i in range(0, num_cells_to_process, batch_size):
        batch_end = min(i + batch_size, num_cells_to_process)
        current_batch_records = tokenized_data[i:batch_end]

        batch_input_ids_unpadded = []
        for cell_idx_in_batch in range(len(current_batch_records['input_ids'])):
            token_sequence_from_dataset = current_batch_records['input_ids'][cell_idx_in_batch]
            if not isinstance(token_sequence_from_dataset, list):
                try:
                    token_sequence_from_dataset = list(token_sequence_from_dataset)
                except TypeError:
                    log_func(
                        f"Warning: Batch {i // batch_size}, record {cell_idx_in_batch}: 'input_ids' is not a list and cannot be converted. Skipping.")
                    continue
            batch_input_ids_unpadded.append(torch.tensor(token_sequence_from_dataset, dtype=torch.long))

        if not batch_input_ids_unpadded:
            log_func(f"Warning: Batch {i // batch_size} did not generate any valid input sequences.")
            continue

        padded_input_ids = rnn_utils.pad_sequence(batch_input_ids_unpadded, batch_first=True,
                                                  padding_value=pad_token_id)
        padded_input_ids = padded_input_ids.to(device)
        attention_mask_tensor = (padded_input_ids != pad_token_id).long().to(device)

        with torch.no_grad():
            outputs = model(padded_input_ids, attention_mask=attention_mask_tensor)
            batch_cell_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings_list.append(batch_cell_embeddings)

        if (i // batch_size + 1) % 100 == 0 or batch_end == num_cells_to_process:
            log_func(f"  Processed cells {batch_end}/{num_cells_to_process}")

    if not all_embeddings_list:
        log_func("Error: Failed to generate any cell embeddings.")
        return None

    final_embeddings = np.concatenate(all_embeddings_list, axis=0)
    log_func(f"Successfully generated embeddings for all cells. Shape: {final_embeddings.shape}")
    return final_embeddings


def main(cli_args):
    # model_name_part = os.path.basename(os.path.normpath(cli_args.finetuned_model_path))
    model_name_part = os.path.normpath(cli_args.finetuned_model_path).split(os.sep)[-2]  # Take the third to last part
    dataset_name_part = os.path.basename(os.path.normpath(cli_args.dataset_path))
    current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_specific_name = f"clustering_eval_leiden_res_{cli_args.leiden_resolution}_seed_{cli_args.seed}_{current_time_str}"

    specific_output_dir = os.path.join(
        cli_args.output_dir,
        model_name_part,
        dataset_name_part,
        run_specific_name
    )
    os.makedirs(specific_output_dir, exist_ok=True)

    log_file = os.path.join(specific_output_dir, "clustering_evaluation_log.txt")

    def current_log(message):
        log_message(message, log_file)

    current_log(f"Script started at: {datetime.datetime.now()}")
    current_log(f"Arguments: {cli_args}")
    current_log(f"All clustering evaluation outputs will be saved to: {specific_output_dir}")

    set_seed(cli_args.seed)

    if torch.cuda.is_available() and cli_args.device >= 0:
        device = torch.device(f"cuda:{cli_args.device}")
        torch.cuda.set_device(cli_args.device)
    else:
        device = torch.device("cpu")
    current_log(f"Using device: {device}")

    current_log("Loading TranscriptomeTokenizer...")
    try:
        tk = TranscriptomeTokenizer(special_token=False)
        PAD_ID = tk.gene_token_dict.get("<pad>")
        if PAD_ID is None:
            current_log("Error: Could not get <pad> ID from Tokenizer. Please check the Tokenizer implementation.")
            return
        current_log(f"Tokenizer loaded successfully. PAD ID: {PAD_ID}.")
    except Exception as e:
        current_log(f"Error: Failed to load TranscriptomeTokenizer: {e}")
        return

    current_log(f"Loading full dataset from: {cli_args.dataset_path}")
    try:
        full_dataset = load_from_disk(cli_args.dataset_path)
        current_log(f"Successfully loaded dataset with {len(full_dataset)} cells.")
        current_log(f"Dataset features: {full_dataset.features}")
        if 'input_ids' not in full_dataset.features:
            current_log("Error: Dataset must contain an 'input_ids' column. Please provide a tokenized dataset.")
            return
    except Exception as e:
        current_log(f"Error: Failed to load dataset: {e}")
        return

    current_log(f"Loading fine-tuned model (BertModel for embeddings) from: {cli_args.finetuned_model_path}")
    try:
        embedding_model = BertModel.from_pretrained(cli_args.finetuned_model_path)
        embedding_model.to(device)
        embedding_model.eval()
        current_log("Successfully loaded embedding model.")
    except Exception as e:
        current_log(f"Error: Failed to load embedding model: {e}")
        return

    cell_embeddings = get_cell_embeddings(
        tokenized_data=full_dataset,
        model=embedding_model,
        pad_token_id=PAD_ID,
        batch_size=cli_args.embedding_batch_size,
        device=device,
        log_func=current_log
    )
    if cell_embeddings is None:
        current_log("Script terminated due to embedding generation failure.")
        return

    current_log("Creating AnnData object...")
    obs_data = {}
    adata_obs_index = [f"cell_{i}" for i in range(len(full_dataset))]

    has_true_labels = False
    if cli_args.label_col_name and cli_args.label_col_name.lower() != 'none' and cli_args.label_col_name in full_dataset.features:
        obs_data['true_labels'] = list(full_dataset[cli_args.label_col_name])
        has_true_labels = True
        current_log(f"Added true labels from column: '{cli_args.label_col_name}' to AnnData.obs['true_labels'].")
    else:
        current_log(
            f"Warning: Specified true label column '{cli_args.label_col_name}' not found in dataset or not provided. Will not perform evaluation based on true labels.")

    adata_obs_df = pd.DataFrame(obs_data, index=adata_obs_index)

    adata = sc.AnnData(X=cell_embeddings, obs=adata_obs_df)
    adata.obsm["X_generated_embedding"] = cell_embeddings
    current_log(f"AnnData object created successfully. Shape: {adata.shape}")
    current_log(f"Columns in obs: {list(adata.obs.columns)}")

    current_log("Starting clustering (computing neighbor graph and Leiden)...")
    try:
        sc.pp.neighbors(adata, use_rep="X_generated_embedding", n_neighbors=15)
        sc.tl.leiden(adata, resolution=cli_args.leiden_resolution, key_added="leiden_clusters", flavor="igraph",
                     n_iterations=2)
        current_log(f"Leiden clustering completed, results stored in AnnData.obs['leiden_clusters']. Resolution: {cli_args.leiden_resolution}")
        adata.obs['leiden_clusters'] = adata.obs['leiden_clusters'].astype('category')
    except Exception as e:
        current_log(f"Error: Clustering execution failed: {e}")
        return

    metrics_results = {}
    if has_true_labels:
        current_log("Starting clustering performance evaluation (ARI, NMI)...")
        true_labels_for_eval = adata.obs['true_labels']
        predicted_clusters = adata.obs['leiden_clusters']

        try:
            ari_score = adjusted_rand_score(true_labels_for_eval, predicted_clusters)
            nmi_score = normalized_mutual_info_score(true_labels_for_eval, predicted_clusters)
            metrics_results['ARI'] = ari_score
            metrics_results['NMI'] = nmi_score
            current_log(f"  Adjusted Rand Index (ARI): {ari_score:.4f}")
            current_log(f"  Normalized Mutual Information (NMI): {nmi_score:.4f}")

            metrics_file_path = os.path.join(specific_output_dir, "clustering_metrics.txt")
            with open(metrics_file_path, "w", encoding="utf-8") as f:
                f.write(f"Clustering Evaluation Metrics (Leiden resolution: {cli_args.leiden_resolution}):\n")
                f.write(f"  Adjusted Rand Index (ARI): {ari_score:.4f}\n")
                f.write(f"  Normalized Mutual Information (NMI): {nmi_score:.4f}\n")
            current_log(f"Clustering evaluation metrics saved to: {metrics_file_path}")

        except Exception as e:
            current_log(f"Error: Failed to calculate clustering metrics: {e}")
    else:
        current_log("Skipping clustering evaluation based on true labels as they were not provided.")

    if has_true_labels:
        current_log("Attempting to map Leiden clusters to cell type names...")
        try:
            if not isinstance(adata.obs['true_labels'].dtype, pd.CategoricalDtype):
                adata.obs['true_labels'] = adata.obs['true_labels'].astype('category')

            cluster_to_label_map = {}
            leiden_categories = adata.obs['leiden_clusters'].cat.categories

            for cluster_id in leiden_categories:
                cells_in_cluster_mask = (adata.obs['leiden_clusters'] == cluster_id)
                most_frequent_label_series = adata.obs['true_labels'][cells_in_cluster_mask].mode()

                if not most_frequent_label_series.empty:
                    label_name = most_frequent_label_series.iloc[0]
                    cluster_to_label_map[cluster_id] = label_name
                else:
                    cluster_to_label_map[cluster_id] = f"Cluster {cluster_id}"

            mapped_labels = adata.obs['leiden_clusters'].map(cluster_to_label_map)
            unique_mapped_labels = [lbl for lbl in pd.Series(mapped_labels).unique() if pd.notna(lbl)]
            true_label_cat_order = list(adata.obs['true_labels'].cat.categories)

            final_cat_order_for_mapped = [lbl for lbl in true_label_cat_order if lbl in unique_mapped_labels]
            for lbl in unique_mapped_labels:
                if lbl not in final_cat_order_for_mapped:
                    final_cat_order_for_mapped.append(lbl)

            adata.obs['leiden_cluster_annotated'] = pd.Categorical(
                mapped_labels,
                categories=final_cat_order_for_mapped,
                ordered=False
            )
            current_log("Leiden clusters successfully mapped to cell type names and stored in 'leiden_cluster_annotated'.")
        except Exception as e:
            current_log(f"Error: Failed to map Leiden clusters to cell type names: {e}")
            import traceback
            current_log(traceback.format_exc())

    current_log("Starting UMAP dimensionality reduction...")
    try:
        sc.tl.umap(adata, min_dist=0.3)
        current_log("UMAP dimensionality reduction complete.")

        current_log("Generating UMAP visualization plots...")

        plot_configs = []
        if has_true_labels:
            if not isinstance(adata.obs['true_labels'].dtype, pd.CategoricalDtype):
                adata.obs['true_labels'] = adata.obs['true_labels'].astype('category')
            plot_configs.append(
                {'color_by': 'true_labels',
                 'title': 'True cell type'})

        predictions_plot_config = {
            'color_by': 'leiden_clusters',
            'title': 'Predictions',
        }
        if has_true_labels and 'leiden_cluster_annotated' in adata.obs:
            predictions_plot_config['color_by'] = 'leiden_cluster_annotated'

        plot_configs.append(predictions_plot_config)

        if not plot_configs:
            current_log("No data columns available for UMAP coloring (true labels or Leiden clusters).")
        else:
            num_plots = len(plot_configs)
            base_plot_width = 6
            legend_allowance = 2.5
            fig_width = (base_plot_width + legend_allowance) * num_plots
            fig_height = base_plot_width + 1
            legend_pos = 'right margin'
            legend_font_size = 'small'

            try:
                plt.rcParams['font.family'] = 'Times New Roman'
                plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Sans', 'Verdana', 'Arial']
                plt.rcParams['axes.unicode_minus'] = False
            except Exception as font_e:
                current_log(f"Warning: Failed to set Times New Roman font: {font_e}")

            with plt.rc_context({"figure.figsize": (fig_width, fig_height), "figure.dpi": 100}):
                fig, axes = plt.subplots(1, num_plots, squeeze=False)

                for i, config in enumerate(plot_configs):
                    ax_curr = axes[0, i]
                    sc.pl.umap(adata,
                               color=config['color_by'],
                               ax=ax_curr,
                               show=False,
                               title=config['title'],
                               legend_loc=legend_pos,
                               legend_fontsize=legend_font_size,
                               legend_fontoutline=None,
                               frameon=True)
                    ax_curr.set_xlabel("UMAP1")
                    ax_curr.set_ylabel("UMAP2" if i == 0 else "")
                    if i > 0: ax_curr.set_yticklabels([])

                fig.subplots_adjust(wspace=0.27)

                # --- Modification starts here ---
                # Define a unified base filename for subsequent layout and multiple uses
                base_filename = os.path.join(specific_output_dir,
                                             f"umap_comparison_{model_name_part}_{dataset_name_part}")

                # Define a list of formats to save
                formats_to_save = {
                    'png': {'dpi': 300},
                    'pdf': {},
                    'svg': {}
                }

                current_log("Starting to save UMAP plots to multiple formats...")
                for fmt, options in formats_to_save.items():
                    filepath = f"{base_filename}.{fmt}"
                    try:
                        fig.savefig(filepath, format=fmt, bbox_inches="tight", **options)
                        current_log(f"  Successfully saved UMAP plot to: {filepath}")
                    except Exception as fig_e:
                        current_log(f"  Error: Failed to save as {fmt.upper()} format: {fig_e}")

                plt.close(fig)
                # --- Modification ends here ---

    except Exception as e:
        current_log(f"Error: UMAP dimensionality reduction or visualization failed: {e}")
        import traceback
        current_log(traceback.format_exc())

    adata_output_path = os.path.join(specific_output_dir, "adata_with_embeddings_and_clusters.h5ad")
    current_log(f"Attempting to save AnnData object to: {adata_output_path}")
    try:
        adata.write_h5ad(adata_output_path)
        current_log(f"AnnData object saved successfully.")
    except Exception as e:
        current_log(f"Error: Failed to save AnnData object: {e}")

    current_log(f"Script finished at: {datetime.datetime.now()}")
    current_log("All operations completed!")


if __name__ == "__main__":
    main(args)