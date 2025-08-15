import random
import numpy as np
import pandas as pd
import torch
import torch.nn.utils.rnn as rnn_utils
from transformers import BertModel
import argparse
import os
import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import scanpy as sc
from datasets import load_from_disk
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

try:
    from scDMC_finetune_tokenizer import TranscriptomeTokenizer
except ImportError:
    print("Error: Unable to import TranscriptomeTokenizer. Please ensure scDMC_finetune_tokenizer.py exists and is in the Python path.")
    exit()

should_use_special_tokens = False

parser = argparse.ArgumentParser(description="End-to-end batch integration evaluation script (final simplified version)")
parser.add_argument("--model_path", type=str,
                    default='../models/finetuned_models/pbmc10k_tk60534')
parser.add_argument("--tokenized_data_path", type=str, default='../datasets/finetune/pbmc10k_tk60534',
                    help="Path to the preprocessed and tokenized dataset file")
parser.add_argument("--integration_output_path", type=str, default='./output/integration/',
                    help="Output path for integration results")
parser.add_argument("--device", type=int, default=0, help="GPU device number")
parser.add_argument("--seed", type=int, default=1, help="Random seed")
parser.add_argument("--embedding_batch_size", type=int, default=64, help="Batch size for generating embeddings")
parser.add_argument("--leiden_resolution", type=float, default=0.2, help="Resolution parameter for Leiden clustering.")
args = parser.parse_args()

GPU_NUMBER = [args.device]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(args.seed)

path_parts = Path(args.model_path).parts
base_model_name = path_parts[-2] if len(path_parts) > 1 else "unknown_model"
task_dataset_name = path_parts[-1]
dataset_short_name = task_dataset_name.split('_')[0]
integration_output_path = Path(args.integration_output_path)
integration_output_path.mkdir(parents=True, exist_ok=True)
current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_output_dir = integration_output_path / base_model_name / task_dataset_name / f"run_{current_time_str}"
run_output_dir.mkdir(parents=True, exist_ok=True)


def log_message(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    with open(run_output_dir / "integration_log.txt", "a", encoding="utf-8") as f:
        f.write(full_message + "\n")


log_message(f"Integration experiment started, output directory: {run_output_dir}")
log_message(f"Parameters: {args}")
log_message(f"Actually use special tokens (e.g., <cls>, <eos>) and add them to the sequence: {should_use_special_tokens} (hardcoded)")

log_message(f"Step 1: Loading preprocessed tokenized data from {args.tokenized_data_path}...")
try:
    tokenized_dataset = load_from_disk(args.tokenized_data_path)
    num_total_cells = len(tokenized_dataset)
    log_message(f"Successfully loaded data: {num_total_cells} cells. Features: {tokenized_dataset.features}")
except Exception as e:
    log_message(f"Error: Failed to load .dataset file: {e}")
    exit()

if "batch" not in tokenized_dataset.features:
    log_message("Error: 'batch' column not found in tokenized_dataset.features. Please ensure the data contains batch information.")
    exit()

label_col_options = ["cell_type", "str_labels", "labels"]
found_label_col = None
for col in label_col_options:
    if col in tokenized_dataset.features:
        found_label_col = col
        break
if not found_label_col:
    log_message(
        f"Warning: No specified cell type label column ({', '.join(label_col_options)}) found in tokenized_dataset.features. scib biological conservation metrics may be affected.")

if 'n_counts' not in tokenized_dataset.features and 'total_counts' not in tokenized_dataset.features:
    log_message("Warning: 'n_counts' or 'total_counts' not found in tokenized_dataset.features.")

log_message(f"Step 2: Loading fine-tuned model from {args.model_path} (for getting embeddings)...")
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
log_message(f"Using device: {device}")
try:
    model_for_embedding = BertModel.from_pretrained(args.model_path, output_hidden_states=True).to(device)
    model_for_embedding.eval()
    log_message("Successfully loaded embedding model.")
except Exception as e:
    log_message(f"Error: Failed to load embedding model: {e}")
    exit()

log_message("Instantiating TranscriptomeTokenizer (using its default configuration)...")
PAD_ID = None
try:
    tk = TranscriptomeTokenizer(special_token=should_use_special_tokens,
                                model_input_size=model_for_embedding.config.max_position_embeddings if hasattr(
                                    model_for_embedding.config, 'max_position_embeddings') else 2048)
    if not hasattr(tk, 'gene_token_dict') or tk.gene_token_dict is None:
        log_message("Error: TranscriptomeTokenizer instance failed to correctly initialize 'gene_token_dict'. Please check its default configuration.")
        exit()
    PAD_ID = tk.gene_token_dict.get("<pad>")
    if PAD_ID is None:
        log_message("Error: Unable to get '<pad>' ID from Tokenizer's gene_token_dict. Please check its default token dictionary.")
        exit()
    if should_use_special_tokens:
        CLS_ID = tk.gene_token_dict.get("<cls>")
        EOS_ID = tk.gene_token_dict.get("<eos>")
        if CLS_ID is None or EOS_ID is None:
            log_message("Error: Tokenizer was asked to use special tokens, but <cls> or <eos> were not found in its default dictionary.")
            exit()
        log_message(f"Tokenizer loaded successfully. PAD ID: {PAD_ID}, CLS ID: {CLS_ID}, EOS ID: {EOS_ID}")
    else:
        log_message(f"Tokenizer loaded successfully. PAD ID: {PAD_ID}. Not actively adding special tokens (<cls>, <eos>) to sequence ends.")
except Exception as e:
    log_message(f"Error: Failed to load or instantiate TranscriptomeTokenizer: {e}")
    import traceback

    log_message(traceback.format_exc())
    exit()

log_message("Step 3: Generating cell embeddings from pre-tokenized data...")


def get_cell_embeddings_from_tokenized_data(tokenized_data, model: torch.nn.Module, pad_token_id_for_batching: int,
                                            batch_size: int = 32, device: str = "cpu",
                                            use_special_tokens_in_sequence: bool = False,
                                            tokenizer_instance_for_special_tokens: TranscriptomeTokenizer = None):
    model.eval()
    all_embeddings_list = []
    num_cells_to_process = len(tokenized_data)
    cls_token_id = None
    eos_token_id = None
    if use_special_tokens_in_sequence:
        if tokenizer_instance_for_special_tokens is None or not hasattr(tokenizer_instance_for_special_tokens,
                                                                        'gene_token_dict'):
            log_message("Error: Requested to use special tokens but a valid tokenizer instance was not provided to get their IDs.")
            return None
        cls_token_id = tokenizer_instance_for_special_tokens.gene_token_dict.get("<cls>")
        eos_token_id = tokenizer_instance_for_special_tokens.gene_token_dict.get("<eos>")
        if cls_token_id is None or eos_token_id is None:
            log_message("Error: Requested to use special tokens but could not get <cls> or <eos> IDs from the provided tokenizer instance.")
            return None
    if 'input_ids' not in tokenized_data.features:
        log_message("Error: 'input_ids' column not found in tokenized_data.features.")
        return None
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
                    log_message(
                        f"Warning: Batch {i // batch_size}, record {cell_idx_in_batch}: 'input_ids' is not a list and cannot be converted. Skipping.")
                    continue
            final_token_sequence = []
            if use_special_tokens_in_sequence:
                final_token_sequence = [cls_token_id] + token_sequence_from_dataset + [eos_token_id]
            else:
                final_token_sequence = token_sequence_from_dataset
            batch_input_ids_unpadded.append(torch.tensor(final_token_sequence, dtype=torch.long))
        if not batch_input_ids_unpadded:
            log_message(f"Warning: Batch {i // batch_size} did not generate any valid input sequences.")
            continue
        padded_input_ids = rnn_utils.pad_sequence(batch_input_ids_unpadded, batch_first=True,
                                                  padding_value=pad_token_id_for_batching)
        padded_input_ids = padded_input_ids.to(device)
        attention_mask_tensor = (padded_input_ids != pad_token_id_for_batching).long().to(device)
        with torch.no_grad():
            outputs = model(padded_input_ids, attention_mask=attention_mask_tensor)
            batch_cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings_list.append(batch_cls_embeddings)
        if (i // batch_size + 1) % 100 == 0 or batch_end == num_cells_to_process:
            log_message(f"  Processed cells {batch_end}/{num_cells_to_process}")
    if not all_embeddings_list:
        log_message("No embeddings were generated.")
        return None
    final_embeddings = np.concatenate(all_embeddings_list, axis=0)
    log_message(f"Generated embeddings for all cells. Shape: {final_embeddings.shape}")
    return final_embeddings


cell_embeddings = get_cell_embeddings_from_tokenized_data(tokenized_data=tokenized_dataset, model=model_for_embedding,
                                                          pad_token_id_for_batching=PAD_ID,
                                                          batch_size=args.embedding_batch_size, device=device,
                                                          use_special_tokens_in_sequence=should_use_special_tokens,
                                                          tokenizer_instance_for_special_tokens=tk if should_use_special_tokens else None)
if cell_embeddings is None:
    log_message("Failed to generate cell embeddings. Exiting.")
    exit()

log_message("Step 3.5: Creating AnnData object to store embeddings and metadata...")
obs_data = {}
obs_data['batch'] = tokenized_dataset['batch']
final_label_col_in_obs = 'cell_type'
if found_label_col:
    obs_data[final_label_col_in_obs] = tokenized_dataset[found_label_col]
    log_message(f"Using '{found_label_col}' as cell type labels and storing in AnnData.obs['{final_label_col_in_obs}'].")
    if 'labels' not in tokenized_dataset.features and found_label_col != 'labels':
        obs_data['labels'] = tokenized_dataset[found_label_col]
        log_message(f"Also copying '{found_label_col}' to AnnData.obs['labels'] for scib usage.")
    elif 'labels' in tokenized_dataset.features:
        obs_data['labels'] = tokenized_dataset['labels']
else:
    log_message(f"Warning: Cell type label column not found, AnnData.obs['{final_label_col_in_obs}'] will not be created.")

if 'n_counts' in tokenized_dataset.features:
    obs_data['n_counts'] = tokenized_dataset['n_counts']
elif 'total_counts' in tokenized_dataset.features:
    obs_data['n_counts'] = tokenized_dataset['total_counts']
    log_message("'total_counts' column found and used as 'n_counts'.")
else:
    log_message(
        f"Warning: 'n_counts' or 'total_counts' not found in tokenized_dataset. AnnData.obs['n_counts'] will not be created.")

keys_to_delete = []
for key, value_list in obs_data.items():
    if len(value_list) != num_total_cells:
        log_message(f"Error: Metadata column '{key}' length ({len(value_list)}) does not match total number of cells ({num_total_cells}).")
        keys_to_delete.append(key)
for key in keys_to_delete:
    del obs_data[key]
    log_message(f"Removed problematic column '{key}'.")

cell_indices = [f"cell_{i}" for i in range(num_total_cells)]
adata_obs_df = pd.DataFrame(obs_data, index=cell_indices)

embedding_key = "X_bert"
adata = sc.AnnData(obs=adata_obs_df, obsm={embedding_key: cell_embeddings})
log_message(f"Created AnnData object. Shape: {adata.shape}")
log_message(f"Columns in obs: {list(adata.obs.columns)}")
log_message(f"Added embeddings to adata.obsm['{embedding_key}']")

log_message("Step 4: Evaluating integration using scib (Luecken et al. 2022 metrics)...")
final_metrics = {}
cluster_key_scib = 'leiden_clusters_for_scib'  # Define here so it can be used in later steps
try:
    import scib

    scib_available = True
except ImportError:
    log_message("Warning: scib library not installed. Skipping scib quantitative evaluation step.")
    scib_available = False

if scib_available:
    batch_key_scib = 'batch'
    label_key_scib = 'labels'

    valid_scib_run = True
    if batch_key_scib not in adata.obs:
        log_message(f"Warning: Required batch key '{batch_key_scib}' for scib is missing in adata.obs. scib evaluation will be incomplete or fail.")
        valid_scib_run = False
    else:
        if not isinstance(adata.obs[batch_key_scib].dtype, pd.CategoricalDtype):
            log_message(f"Converting batch key '{batch_key_scib}' column to category type for scib.")
            adata.obs[batch_key_scib] = adata.obs[batch_key_scib].astype('category')
    if label_key_scib not in adata.obs:
        if final_label_col_in_obs in adata.obs:
            log_message(
                f"Required label key '{label_key_scib}' for scib is missing, creating from '{final_label_col_in_obs}' column and converting to category type.")
            adata.obs[label_key_scib] = adata.obs[final_label_col_in_obs].astype('category')
        else:
            log_message(
                f"Warning: Required label key '{label_key_scib}' and alternative '{final_label_col_in_obs}' column are both missing in adata.obs. scib biological metrics evaluation will be affected.")
            valid_scib_run = False
    else:
        if not isinstance(adata.obs[label_key_scib].dtype, pd.CategoricalDtype):
            log_message(f"Converting existing label key '{label_key_scib}' column to category type for scib.")
            adata.obs[label_key_scib] = adata.obs[label_key_scib].astype('category')

    if valid_scib_run:
        try:
            log_message(f"Calculating neighbor graph for evaluation metrics (based on {embedding_key})...")
            sc.pp.neighbors(adata, use_rep=embedding_key, n_neighbors=15)
            log_message(f"Running Leiden clustering, resolution: {args.leiden_resolution}")
            sc.tl.leiden(adata, resolution=args.leiden_resolution, key_added=cluster_key_scib, flavor="igraph",
                         n_iterations=2)
            log_message("Calculating core metrics...")

            # 1. Calculate biological conservation metrics
            final_metrics['ARI'] = adjusted_rand_score(adata.obs[label_key_scib], adata.obs[cluster_key_scib])
            final_metrics['NMI'] = normalized_mutual_info_score(adata.obs[label_key_scib], adata.obs[cluster_key_scib])
            final_metrics['ASW_cell'] = scib.metrics.silhouette(adata, label_key=label_key_scib, embed=embedding_key)

            bio_score_components = [final_metrics.get('ARI'), final_metrics.get('NMI'), final_metrics.get('ASW_cell')]
            valid_bio_scores = [s for s in bio_score_components if s is not None and not np.isnan(s)]
            final_metrics['Avg_Bio'] = np.mean(valid_bio_scores) if valid_bio_scores else np.nan

            # 2. Calculate batch correction metrics
            final_metrics['ASW_batch'] = scib.metrics.silhouette_batch(
                adata,
                batch_key=batch_key_scib,
                label_key=label_key_scib,
                embed=embedding_key,
                verbose=False
            )
            final_metrics['GraphConn'] = scib.metrics.graph_connectivity(adata, label_key=label_key_scib)

            batch_score_components = [final_metrics.get('ASW_batch'), final_metrics.get('GraphConn')]
            valid_batch_scores = [s for s in batch_score_components if s is not None and not np.isnan(s)]
            final_metrics['Avg_batch'] = np.mean(valid_batch_scores) if valid_batch_scores else np.nan

            # 3. Calculate final overall score
            avg_bio_score = final_metrics.get('Avg_Bio')
            avg_batch_score = final_metrics.get('Avg_batch')
            overall_score = np.nan
            if avg_bio_score is not None and not np.isnan(
                    avg_bio_score) and avg_batch_score is not None and not np.isnan(avg_batch_score):
                overall_score = 0.6 * avg_bio_score + 0.4 * avg_batch_score
            final_metrics['overall_score'] = overall_score

            log_message("\n--- Final Metrics Summary ---")
            log_message(f"  ARI: {final_metrics.get('ARI', np.nan):.4f}")
            log_message(f"  NMI: {final_metrics.get('NMI', np.nan):.4f}")
            log_message(f"  ASW_cell (Biology): {final_metrics.get('ASW_cell', np.nan):.4f}")
            log_message(f"  Avg_Bio: {final_metrics.get('Avg_Bio', np.nan):.4f}  (Average score for biological conservation)")
            log_message("-" * 20)
            log_message(f"  ASW_batch (Batch): {final_metrics.get('ASW_batch', np.nan):.4f}")
            log_message(f"  GraphConn (Batch): {final_metrics.get('GraphConn', np.nan):.4f}")
            log_message(f"  Avg_batch: {final_metrics.get('Avg_batch', np.nan):.4f}  (Average score for batch integration)")
            log_message("-" * 20)
            log_message(
                f"  overall_score: {final_metrics.get('overall_score', np.nan):.4f}  (0.6*Avg_Bio + 0.4*Avg_batch)")
            log_message("--- End ---")

            # Save results
            summary_df = pd.DataFrame.from_dict(final_metrics, orient='index', columns=['Score'])
            summary_df.to_csv(run_output_dir / "metrics_summary.csv")
            log_message(f"Metrics summary saved to {run_output_dir / 'metrics_summary.csv'}")

        except Exception as e:
            log_message(f"Error during scib evaluation or custom metric calculation: {e}")
            import traceback

            log_message(traceback.format_exc())
    else:
        log_message("Skipping scib evaluation due to missing required 'batch' or 'labels' columns (or failure to convert to category type).")

log_message("Step 5: Generating UMAP visualizations...")
try:
    if embedding_key in adata.obsm:
        if "neighbors" not in adata.uns:
            log_message(f"Calculating neighbor graph (based on {embedding_key})...")
            sc.pp.neighbors(adata, use_rep=embedding_key, n_neighbors=15)
        else:
            log_message("Neighbor graph already exists.")
        log_message("Calculating UMAP...")
        sc.tl.umap(adata)

        plot_colors = []
        if 'cell_type' in adata.obs: plot_colors.append('cell_type')
        if cluster_key_scib in adata.obs: plot_colors.append(cluster_key_scib)
        if 'batch' in adata.obs: plot_colors.append('batch')

        if not plot_colors:
            log_message("Warning: No columns found in adata.obs suitable for plotting. Cannot draw UMAP plots.")
        else:
            title_map = {'cell_type': f'True cell type ({dataset_short_name})',
                         cluster_key_scib: f'{base_model_name} Predictions', 'batch': 'Batch'}

            log_message("Generating separate UMAP plots (legend on the right)...")
            for color_key in plot_colors:
                log_message(f"  Plotting UMAP colored by '{color_key}'...")

                if not isinstance(adata.obs[color_key].dtype, pd.CategoricalDtype):
                    log_message(f"Converting plotting column '{color_key}' to category type.")
                    adata.obs[color_key] = adata.obs[color_key].astype('category')

                num_categories = len(adata.obs[color_key].cat.categories)
                legend_loc = 'right margin'

                if num_categories > 30:
                    fig_width, fig_height = 12, 8
                    legend_fs = 6
                else:
                    fig_width, fig_height = 10, 7
                    legend_fs = 'small'

                with plt.rc_context({"figure.figsize": (fig_width, fig_height), "figure.dpi": 150}):
                    fig, ax = plt.subplots(1, 1)

                    sc.pl.umap(adata, color=color_key, ax=ax, show=False,
                               title=title_map.get(color_key, color_key.replace('_', ' ').capitalize()),
                               legend_loc=legend_loc,
                               legend_fontsize=legend_fs,
                               frameon=True)

                    ax.set_xlabel("UMAP1")
                    ax.set_ylabel("UMAP2")

                    base_filename = run_output_dir / f"umap_{color_key}"
                    formats_to_save = {'png': {'dpi': 300}, 'pdf': {}, 'svg': {}}
                    log_message(f"  Starting to save UMAP plot ('{color_key}') to multiple formats...")
                    for fmt, options in formats_to_save.items():
                        filepath = f"{base_filename}.{fmt}"
                        try:
                            fig.savefig(filepath, format=fmt, bbox_inches="tight", **options)
                            log_message(f"    Successfully saved UMAP plot to: {filepath}")
                        except Exception as fig_e:
                            log_message(f"    Error: Failed to save as {fmt.upper()} format: {fig_e}")
                    plt.close(fig)
except Exception as e:
    log_message(f"Error during UMAP visualization: {e}")
    import traceback

    log_message(traceback.format_exc())

log_message("\nAppending results to the main Excel file...")
excel_file_path = integration_output_path / "integration_results.xlsx"
new_row_data = {
    'model': base_model_name,
    'task_dataset': task_dataset_name,
    'ARI': final_metrics.get('ARI'),
    'NMI': final_metrics.get('NMI'),
    'ASW_cell': final_metrics.get('ASW_cell'),
    'Avg_Bio': final_metrics.get('Avg_Bio'),
    'ASW_batch': final_metrics.get('ASW_batch'),
    'GraphConn': final_metrics.get('GraphConn'),
    'Avg_batch': final_metrics.get('Avg_batch'),
    'overall_score': final_metrics.get('overall_score'),
    'leiden_resolution': args.leiden_resolution,
    'seed': args.seed,
    'date': datetime.datetime.now().strftime("%Y-%m-%d")
}
df_new_row = pd.DataFrame([new_row_data])
try:
    if os.path.exists(excel_file_path):
        df_existing = pd.read_excel(excel_file_path)
        df_combined = pd.concat([df_existing, df_new_row], ignore_index=True)
    else:
        df_combined = df_new_row
    df_combined.to_excel(excel_file_path, index=False)
    log_message(f"Successfully saved/appended results to: {excel_file_path}")
except Exception as e:
    log_message(f"Error: Failed to save to Excel file: {e}")

log_message(f"Integration experiment finished. Results are in {run_output_dir}")
