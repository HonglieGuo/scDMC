import json
import logging
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from scipy.stats import zscore
from scipy.sparse import csr_matrix
from tqdm import tqdm

# --- Hugging Face Transformers ---
from datasets import load_from_disk
from transformers import BertConfig, BertForSequenceClassification

# --- Visualization ---
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

try:
    import scanpy as sc
except ImportError:
    print("Error: Please install `scanpy` and `leidenalg` libraries to run gene program analysis.")
    print("Run: pip install scanpy leidenalg")
    exit(1)

# --- Check and import custom modules ---
try:
    from scDMC_finetune_tokenizer import TranscriptomeTokenizer
except ImportError as e:
    print(f"Error: Failed to import TranscriptomeTokenizer. Please ensure scDMC_finetune_tokenizer.py exists.")
    exit(1)

# ==============================================================================
# 【Core Configuration Area】--- Modify your parameters here ---
# ==============================================================================
FINETUNED_MODEL_PATH = "../models/finetuned_models/ms_tk60534"
DATA_PATH = "../datasets/finetune/ms_tk60534"
GENE_MAP_PATH = "../tokenizer/token_files/gene_to_ensembl_mapping_95m.json"
OUTPUT_PATH = "./output/attention_analysis"
ANALYZE_CELL_TYPES = "all"
DEVICE_ID = 0
TOP_N_GENES_FOR_HEATMAP = 5
# --- Sampling Configuration ---
N_SAMPLES_PER_CELL_TYPE = 100  # Sample 100 cells per cell type for the heatmap
# --- [New] Gene Program Clustering Configuration ---
LEIDEN_RESOLUTION = 1.0  # Clustering resolution, a larger value results in more gene programs (clusters)
# ==============================================================================

# --- Global and Font Settings (Beautified) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
try:
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    # Increased font size and added a medium font size for better readability
    tnr_font_large = FontProperties(family='Times New Roman', size=22)
    tnr_font_medium = FontProperties(family='Times New Roman', size=20)  # New medium font size
    tnr_font_small = FontProperties(family='Times New Roman', size=16)  # For gene program labels
    tnr_font_title = FontProperties(family='Times New Roman', size=24, weight='bold')
    logging.info("Successfully set global font to Times New Roman.")
except Exception as e:
    logging.warning(f"Failed to set Times New Roman font: {e}")
    # Provide fallback font sizes
    tnr_font_large = FontProperties(size=14)
    tnr_font_medium = FontProperties(size=12)
    tnr_font_small = FontProperties(size=10)
    tnr_font_title = FontProperties(size=18, weight='bold')


# --- Utility Functions ---
def load_and_invert_gene_map(filepath):
    """Load and intelligently invert the gene mapping file."""
    logging.info(f"Loading gene map from '{filepath}'...")
    try:
        with open(filepath, 'r') as f:
            symbol_to_ensebl = json.load(f)
        ensembl_to_symbol_map = {}
        for symbol, ensembl_id in symbol_to_ensebl.items():
            if ensembl_id not in ensembl_to_symbol_map or (
                    symbol.isupper() and not ensembl_to_symbol_map[ensembl_id].isupper()) or (
                    len(symbol) < len(ensembl_to_symbol_map[ensembl_id])):
                ensembl_to_symbol_map[ensembl_id] = symbol
        logging.info(f"Successfully loaded and inverted {len(ensembl_to_symbol_map)} gene mappings.")
        return ensembl_to_symbol_map
    except Exception as e:
        logging.error(f"Error loading gene map file: {e}. Charts will only show Ensembl IDs.")
        return {}


def get_average_attention_scores(model, dataset, tokenizer, label_text_map, ensembl_to_symbol_map, device, output_dir,
                                 analyze_cell_types_str="all"):
    """Calculate average attention scores for each cell type and generate bar plots."""
    logging.info("\n" + "=" * 60 + "\n========== Stage 1: Calculate Average Attention & Generate Bar Plots ==========\n" + "=" * 60)
    model.eval().to(device)
    id_to_ensembl = {v: k for k, v in tokenizer.gene_token_dict.items()}
    all_cell_type_scores = {}
    cell_types_to_process = list(label_text_map.values())
    if analyze_cell_types_str.lower() != 'all':
        specified_types = {ct.strip() for ct in analyze_cell_types_str.split(',')}
        cell_types_to_process = [ct for ct in cell_types_to_process if ct in specified_types]

    for cell_type_name in sorted(cell_types_to_process):
        cell_type_dataset = dataset.filter(lambda ex: ex['label_text'] == cell_type_name, num_proc=os.cpu_count() or 1)
        if len(cell_type_dataset) == 0: continue
        logging.info(f"\n--- Processing cell type: '{cell_type_name}' ({len(cell_type_dataset)} cells) ---")

        gene_attention_sum = defaultdict(float)
        gene_counts = defaultdict(int)
        with torch.no_grad():
            for cell_data in tqdm(cell_type_dataset, desc=f"Processing {cell_type_name}"):
                input_ids = cell_data['input_ids']
                input_tensor = torch.tensor([input_ids], device=device)
                outputs = model(input_ids=input_tensor, output_attentions=True)
                attentions = outputs.attentions
                aggregated_attention = torch.stack(attentions).mean(dim=[0, 1, 2]).squeeze(0)
                attention_scores_per_gene = aggregated_attention.sum(dim=0)
                special_ids = {tokenizer.gene_token_dict.get(token, -1) for token in
                               ["<cls>", "<pad>", "<eos>", "<unk>", "<mask>"]}
                for j, token_id in enumerate(input_ids):
                    if token_id not in special_ids:
                        gene_attention_sum[token_id] += attention_scores_per_gene[j].item()
                        gene_counts[token_id] += 1

        avg_scores_by_id = {tid: gene_attention_sum[tid] / gene_counts[tid] for tid in gene_attention_sum if
                            gene_counts[tid] > 0}
        avg_scores_by_ensembl = {id_to_ensebl.get(tid, f"ID_{tid}"): score for tid, score in avg_scores_by_id.items()}
        all_cell_type_scores[cell_type_name] = avg_scores_by_ensembl

        safe_cell_type_name = cell_type_name.replace(" ", "_").replace("/", "_")
        cell_type_output_dir = Path(output_dir) / safe_cell_type_name
        cell_type_output_dir.mkdir(parents=True, exist_ok=True)
        df_scores = pd.DataFrame(list(avg_scores_by_ensebl.items()),
                                 columns=['EnsemblID', 'AttentionScore']).sort_values('AttentionScore', ascending=False)
        df_scores['GeneSymbol'] = df_scores['EnsemblID'].map(ensembl_to_symbol_map).fillna(df_scores['EnsemblID'])
        df_scores.to_csv(cell_type_output_dir / 'attention_scores.csv', index=False)
        plot_top_attention_barplot(df_scores, cell_type_name, cell_type_output_dir)

    logging.info("\nAverage score calculation and bar plot generation complete!")
    return all_cell_type_scores


def plot_top_attention_barplot(scores_df, cell_type_name, output_dir, top_n=10):
    """Plot a bar chart for the Top-N genes."""
    if scores_df.empty: return
    top_genes_df = scores_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 12))
    sns.barplot(x='AttentionScore', y='GeneSymbol', data=top_genes_df, palette='viridis', hue='GeneSymbol',
                legend=False, ax=ax)
    ax.set_title(f'Top {top_n} Attention Genes for\n{cell_type_name}', fontproperties=tnr_font_title)
    ax.set_xlabel('Average Attention Score', fontproperties=tnr_font_large)
    ax.set_ylabel('Gene Symbol', fontproperties=tnr_font_large)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()): label.set_fontproperties(tnr_font_large)
    plt.tight_layout()
    output_basename = Path(output_dir) / 'barplot_top_attention_genes'
    plt.savefig(f"{output_basename}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_basename}.pdf", bbox_inches='tight')
    plt.close(fig)
    logging.info(f"  > Saved Top-{top_n} gene bar plot for '{cell_type_name}'.")


def generate_sampled_heatmap(model, dataset, tokenizer, label_text_map, ensembl_to_symbol_map, device, output_dir,
                             avg_scores_per_type, top_n_per_type=5, n_samples=100, dataset_name=""):
    """Enhance visualization of sparse signals by applying a log transform to the raw scores."""
    logging.info(
        "\n" + "=" * 60 + f"\n========== Stage 2: Generate Log-transformed Sampled Heatmap (N={n_samples}) ==========\n" + "=" * 60)

    logging.info(f"Sampling up to {n_samples} cells from each cell type...")
    sampled_indices = []
    cell_types_to_process = sorted(label_text_map.values())
    for cell_type in cell_types_to_process:
        all_cell_indices = np.where(np.array(dataset['label_text']) == cell_type)[0]
        chosen_indices = np.random.choice(all_cell_indices, min(len(all_cell_indices), n_samples), replace=False)
        sampled_indices.extend(chosen_indices)
        logging.info(f"  > Selected {len(chosen_indices)} cells for '{cell_type}'.")

    sampled_dataset = dataset.select(sampled_indices).sort("label_text")
    logging.info(f"Sampling complete. Total of {len(sampled_dataset)} cells for heatmap generation.")

    final_genes_for_heatmap_set = set()
    for cell_type in cell_types_to_process:
        sorted_genes = sorted(avg_scores_per_type[cell_type].items(), key=lambda item: item[1], reverse=True)
        for ensembl_id, score in sorted_genes[:top_n_per_type]:
            if ensembl_id and not ensembl_id.startswith("ID_"): final_genes_for_heatmap_set.add(ensembl_id)
    final_genes_for_heatmap = sorted(list(final_genes_for_heatmap_set))
    if not final_genes_for_heatmap: logging.warning("Could not determine any top genes for the heatmap."); return

    heatmap_gene_symbols = [ensembl_to_symbol_map.get(eid, eid) for eid in final_genes_for_heatmap]
    ensembl_to_token_id = tokenizer.gene_token_dict

    logging.info("Constructing raw attention score matrix...")
    cell_type_annot = sampled_dataset['label_text']
    heatmap_matrix = np.zeros((len(final_genes_for_heatmap), len(sampled_dataset)))

    model.eval().to(device)
    with torch.no_grad():
        for col_idx, cell_data in enumerate(tqdm(sampled_dataset, desc="Populating heatmap matrix")):
            input_ids = cell_data['input_ids']
            input_tensor = torch.tensor([input_ids], device=device)
            outputs = model(input_ids=input_tensor, output_attentions=True)
            attentions = outputs.attentions
            aggregated_attention = torch.stack(attentions).mean(dim=[0, 1, 2]).squeeze(0)
            attention_scores_per_gene = aggregated_attention.sum(dim=0)
            scores_dict = {tid: score.item() for tid, score in zip(input_ids, attention_scores_per_gene)}
            for row_idx, ensembl_id in enumerate(final_genes_for_heatmap):
                token_id = ensembl_to_token_id.get(ensembl_id)
                if token_id in scores_dict: heatmap_matrix[row_idx, col_idx] = scores_dict[token_id]

    logging.info("Applying log(1+x) transform to raw scores...")
    log_transformed_matrix = np.log1p(heatmap_matrix)
    heatmap_df = pd.DataFrame(log_transformed_matrix, index=heatmap_gene_symbols)
    logging.info("Log transformation complete, plotting heatmap...")

    lut = dict(zip(cell_types_to_process, sns.color_palette("husl", len(cell_types_to_process))))
    col_colors = pd.Series(cell_type_annot, name="Cell Type").map(lut)

    g = sns.clustermap(heatmap_df, cmap="Reds", col_cluster=False, row_cluster=True, col_colors=col_colors,
                       dendrogram_ratio=(0.05, 0.05),
                       # 1. Fine-tune colorbar position to the top right
                       cbar_pos=(1.07, 0.7, 0.03, 0.2),
                       figsize=(15, 20),
                       xticklabels=False, yticklabels=True, linewidths=0)

    g.ax_col_colors.yaxis.label.set_fontproperties(tnr_font_large)
    g.ax_heatmap.set_xlabel(f"Cells (Sampled N={n_samples} per type)", fontproperties=tnr_font_large)
    g.ax_heatmap.set_ylabel("Top Attention Genes", fontproperties=tnr_font_large, fontsize=26)
    for tick_label in g.ax_heatmap.get_yticklabels(): tick_label.set_fontproperties(tnr_font_large)

    # Set colorbar title and font
    g.cax.set_title('Attention\nlog(1+Score)', fontproperties=tnr_font_large, loc='left')
    for label in g.cax.get_yticklabels(): label.set_fontproperties(tnr_font_large)

    # 2. Create and place the cell type legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=lut[name]) for name in cell_types_to_process]

    # Attach the legend to the main heatmap axis and place it in the bottom right
    g.ax_heatmap.legend(handles, cell_types_to_process,
                        bbox_to_anchor=(1.1, 0),  # Anchor point is to the right and bottom of the heatmap
                        loc='lower left',  # Align the bottom-left of the legend to the anchor
                        prop=tnr_font_large,
                        title='Cell Type',
                        title_fontproperties=tnr_font_large)

    title = 'Attention Weights Across Sampled Cells'
    if dataset_name:
        title += f' ({dataset_name})'
    g.fig.suptitle(title, fontproperties=tnr_font_title, y=0.98)

    output_basename = Path(output_dir) / f'heatmap_sampled_log_transformed_n{n_samples}'
    g.savefig(f"{output_basename}.png", dpi=300, bbox_inches='tight')
    g.savefig(f"{output_basename}.pdf", bbox_inches='tight')
    plt.close()
    logging.info(f"Log-transformed heatmap saved to: {output_dir.resolve()}")


def generate_gene_program_heatmap(model, dataset_df, tokenizer, ensembl_to_symbol_map, output_dir, resolution=0.8,
                                  dataset_name=""):
    """Extract gene programs by clustering gene embeddings and visualize their average expression across cell types. (Beautified)"""
    logging.info("\n" + "=" * 60 + "\n========== Stage 3: Generate Gene Program Heatmap ==========\n" + "=" * 60)

    # --- Step 1: Extract Gene Embeddings ---
    logging.info("Step 1/4: Extracting gene embeddings from the fine-tuned model...")
    device = model.device
    gene_token_dict = tokenizer.gene_token_dict
    vocab_tokens = {v: k for k, v in gene_token_dict.items()}

    ensembl_list_for_adata = []
    token_ids = []
    for token_id, gene_ensembl in vocab_tokens.items():
        if not gene_ensembl.startswith('<'):
            ensembl_list_for_adata.append(gene_ensembl)
            token_ids.append(token_id)

    with torch.no_grad():
        token_ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
        gene_embeddings = model.bert.get_input_embeddings()(token_ids_tensor).cpu().numpy()
    logging.info(f"Successfully extracted embeddings for {len(ensembl_list_for_adata)} genes.")

    # --- Step 2: Cluster Genes ---
    logging.info(f"Step 2/4: Clustering genes using Leiden algorithm (resolution={resolution})...")
    adata_genes = sc.AnnData(gene_embeddings)
    adata_genes.obs_names = ensembl_list_for_adata
    sc.pp.neighbors(adata_genes, use_rep='X', n_neighbors=30)
    sc.tl.leiden(adata_genes, resolution=resolution, key_added='gene_programs', flavor="igraph", n_iterations=2)
    gene_programs = adata_genes.obs.groupby('gene_programs', observed=True).apply(lambda x: x.index.tolist(),
                                                                                  include_groups=False).to_dict()
    logging.info(f"Successfully clustered genes into {len(gene_programs)} programs.")

    # --- Step 3: Calculate Average Expression Matrix ---
    logging.info("Step 3/4: Calculating average expression of each gene program in each cell type...")
    cell_types = sorted(dataset_df['label_text'].unique())
    program_ids = sorted(gene_programs.keys())
    heatmap_df = pd.DataFrame(index=program_ids, columns=cell_types, dtype=float)
    all_genes_in_data = [g for g in dataset_df.columns if g in ensembl_list_for_adata]
    mean_expr_by_celltype = dataset_df.groupby('label_text')[all_genes_in_data].mean()
    for prog_id, genes_in_prog in tqdm(gene_programs.items(), desc="Calculating program expression"):
        valid_genes = [g for g in genes_in_prog if g in mean_expr_by_celltype.columns]
        if not valid_genes: continue
        heatmap_df.loc[prog_id] = mean_expr_by_celltype[valid_genes].mean(axis=1)
    heatmap_df.dropna(inplace=True)

    # --- Step 4: Standardize and Visualize ---
    logging.info("Step 4/4: Standardizing matrix and plotting heatmap...")
    heatmap_zscored = pd.DataFrame(zscore(heatmap_df, axis=1, nan_policy='omit'), index=heatmap_df.index,
                                   columns=heatmap_df.columns).fillna(0)

    yticklabels = []
    for prog_id in heatmap_zscored.index:
        genes_symbols = [ensembl_to_symbol_map.get(g, g) for g in gene_programs[prog_id]]
        label = ", ".join(genes_symbols[:5])
        if len(genes_symbols) > 5: label += ", ..."
        yticklabels.append(f"Prog {prog_id}: {label}")

    # ==================== Beautified Plotting Area ====================
    g = sns.clustermap(
        heatmap_zscored,
        cmap="viridis", row_cluster=True, col_cluster=True,
        dendrogram_ratio=(0.05, 0.03),

        cbar_pos=(1.2, 0, 0.02, 0.15),

        figsize=(10, max(12, len(heatmap_zscored) * 0.4)),
        linewidths=0.5, yticklabels=yticklabels
    )

    # Adjust title position and font
    title = 'Cell-Type Specific Activation of Gene Programs'
    if dataset_name:
        title += f' ({dataset_name})'
    g.fig.suptitle(title, fontproperties=tnr_font_title, y=1.02)

    # Adjust axis labels and tick labels font
    g.ax_heatmap.set_xlabel("Cell Type", fontproperties=tnr_font_large)
    g.ax_heatmap.set_ylabel("Gene Programs (clustered by function)", fontproperties=tnr_font_large)

    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor",
             fontproperties=tnr_font_medium)
    plt.setp(g.ax_heatmap.get_yticklabels(), fontproperties=tnr_font_small)

    # Adjust colorbar title and tick font
    g.cax.set_title('Mean Expression\n(Z-score)', fontproperties=tnr_font_medium, loc='center')
    for label in g.cax.get_yticklabels():
        label.set_fontproperties(tnr_font_small)

    output_basename = Path(output_dir) / f'heatmap_gene_programs_res{resolution}'
    g.savefig(f"{output_basename}.png", dpi=300, bbox_inches='tight')
    g.savefig(f"{output_basename}.pdf", bbox_inches='tight')
    plt.close()

    logging.info(f"Beautified gene program heatmap saved to: {output_dir.resolve()}")


def load_label_map_from_file(model_path):
    """Load label mapping from label_map.txt."""
    label_map_file = Path(model_path) / "label_map.txt"
    if not label_map_file.exists(): return None, None
    id2label, label2id = {}, {}
    try:
        with open(label_map_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t');
                if len(parts) == 2:
                    label_id, label_name = int(parts[0]), parts[1]
                    id2label[label_id] = label_name
                    label2id[label_name] = label_id
        return id2label, label2id
    except Exception as e:
        logging.error(f"Failed to read label_map.txt: {e}");
        return None, None


def main():
    finetuned_model_path = Path(FINETUNED_MODEL_PATH)
    if not (finetuned_model_path.is_dir() and ((finetuned_model_path / "pytorch_model.bin").exists() or (
            finetuned_model_path / "model.safetensors").exists())):
        logging.error(f"Error: The specified model path '{finetuned_model_path}' is invalid.");
        return

    output_dir = Path(OUTPUT_PATH) / finetuned_model_path.name
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Analysis results will be saved to: {output_dir}")

    dataset_name_for_title = finetuned_model_path.name.split('_')[0]

    device = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() and DEVICE_ID >= 0 else "cpu")
    logging.info(f"Using device: {device}")

    ensembl_to_symbol_map = load_and_invert_gene_map(GENE_MAP_PATH)

    logging.info(f"Loading model from '{finetuned_model_path}'...")
    try:
        config_analysis = BertConfig.from_pretrained(finetuned_model_path, output_attentions=True,
                                                     attn_implementation="eager")
        model_analysis = BertForSequenceClassification.from_pretrained(finetuned_model_path, config=config_analysis)
        model_analysis.to(device)
    except Exception as e:
        logging.error(f"Failed to load model: {e}");
        return

    logging.info(f"Loading dataset from: {DATA_PATH}")
    try:
        dataset = load_from_disk(DATA_PATH)
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}");
        return
    tk = TranscriptomeTokenizer()

    logging.info("Getting authoritative label mapping...")
    id2label, label2id = load_label_map_from_file(finetuned_model_path)
    if not id2label:
        logging.warning("label_map.txt not found, will build label map dynamically from the dataset.")
        source_column = next((c for c in ['cell_type', 'celltype', 'labels'] if c in dataset.column_names), None)
        if not source_column: logging.error("Error: Could not find any label column."); return
        unique_labels = sorted(list(set(dataset[source_column])))
        label2id = {label: i for i, label in enumerate(unique_labels)}
        id2label = {i: label for i, label in enumerate(unique_labels)}
    model_analysis.config.id2label, model_analysis.config.label2id = id2label, label2id
    logging.info(f"Updated model config with authoritative label map. Total {len(id2label)} labels.")

    source_column = next((c for c in ['cell_type', 'celltype', 'labels'] if c in dataset.column_names), None)

    labeled_dataset = dataset.map(
        lambda ex: {'label': label2id.get(ex[source_column], -1), 'label_text': ex[source_column]},
        num_proc=os.cpu_count() or 1)

    # --- [Modification 2] Replace memory-intensive code with efficient sparse matrix construction ---
    # --- [Final Fix & Memory Optimization] Prepare a DataFrame with raw expression values for gene program analysis ---
    logging.info("Efficiently converting dataset to a sparse matrix for subsequent analysis...")

    # 1. Create a mapping from gene to column index
    all_genes_list = [g for g in tk.gene_token_dict.keys() if not g.startswith('<')]
    gene_to_col_idx = {gene: i for i, gene in enumerate(all_genes_list)}
    id_to_gene_map = {v: k for k, v in tk.gene_token_dict.items()}

    # 2. Efficiently build a sparse matrix (CSR format)
    data, indices, indptr = [], [], [0]
    num_cells = len(dataset)
    for cell in tqdm(dataset, desc="Extracting expression data (sparse mode)"):
        # For genes present in each cell, record their column indices
        cell_gene_indices = []
        for gene_id in cell['input_ids']:
            gene_ensembl = id_to_gene_map.get(gene_id)
            if gene_ensembl in gene_to_col_idx:
                cell_gene_indices.append(gene_to_col_idx[gene_ensembl])

        # Sort and remove duplicates, just in case
        cell_gene_indices = sorted(list(set(cell_gene_indices)))

        # Add data to lists
        data.extend([1] * len(cell_gene_indices))  # Expression values are all 1
        indices.extend(cell_gene_indices)  # Column indices of non-zero elements
        indptr.append(len(indices))  # Record the end position of each row

    # 3. Create the sparse matrix
    sparse_matrix = csr_matrix((data, indices, indptr), shape=(num_cells, len(all_genes_list)), dtype=np.float32)
    logging.info(f"Sparse matrix created, significantly reducing memory usage. Matrix dimensions: {sparse_matrix.shape}")

    # 4. Create a Pandas DataFrame using a sparse data type (memory-friendly)
    df_for_analysis = pd.DataFrame.sparse.from_spmatrix(sparse_matrix, columns=all_genes_list)
    df_for_analysis['label_text'] = list(dataset[source_column])
    logging.info("Sparse DataFrame conversion complete.")
    # --- End of modification ---

    # Stage 1: Calculate average scores and generate bar plots
    avg_scores_per_type = get_average_attention_scores(model_analysis, labeled_dataset, tk, id2label,
                                                       ensembl_to_symbol_map, device, output_dir, ANALYZE_CELL_TYPES)

    # Stage 2: Generate log-transformed sampled heatmap
    if avg_scores_per_type:
        generate_sampled_heatmap(model_analysis, labeled_dataset, tk, id2label, ensembl_to_symbol_map, device,
                                 output_dir, avg_scores_per_type,
                                 top_n_per_type=TOP_N_GENES_FOR_HEATMAP,
                                 n_samples=N_SAMPLES_PER_CELL_TYPE,
                                 dataset_name=dataset_name_for_title)

    # Stage 3: Generate gene program heatmap
    generate_gene_program_heatmap(model_analysis, df_for_analysis, tk, ensembl_to_symbol_map,
                                  output_dir, resolution=LEIDEN_RESOLUTION,
                                  dataset_name=dataset_name_for_title)

    logging.info(f"All analysis scripts have completed! Results are saved in {output_dir.resolve()}")


if __name__ == "__main__":
    main()