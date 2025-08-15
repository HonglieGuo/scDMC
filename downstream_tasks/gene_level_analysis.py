import argparse
import datetime
import os
import random
from pathlib import Path
import logging
import warnings
from collections import defaultdict
import json

import numpy as np
import pandas as pd
import torch
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from datasets import load_from_disk
from transformers import BertModel
import torch.nn.utils.rnn as rnn_utils
import gseapy as gp
from scipy.stats import ttest_rel
from sklearn.metrics.pairwise import cosine_similarity

try:
    from scDMC_finetune_tokenizer import TranscriptomeTokenizer
except ImportError:
    print("Error: Unable to import TranscriptomeTokenizer. Please ensure scDMC_finetune_tokenizer.py exists and is in the Python path.")
    exit()

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['axes.labelsize'] = 22
except Exception as e:
    logging.warning(f"Failed to set font: {e}")

# --- Task Control Switches ---
RUN_GENE_MODULE_DISCOVERY = True
RUN_GRN_INFERENCE = True
RUN_GRN_TOPOLOGY_SCAN = True
RUN_MODULE_QUANTITATIVE_EVAL = False


# --- 1. Argument Definitions ---
def get_args():
    parser = argparse.ArgumentParser(description="Gene-level analysis script")
    parser.add_argument("--model_path", type=str, default='../models/finetuned_models/myeloid_tk60534')
    parser.add_argument("--tokenized_data_path", type=str, default='../datasets/finetune/myeloid_tk60534')
    parser.add_argument("--raw_adata_path", type=str, default='../datasets/adata/myeloid.h5ad',
                        help="Path to the raw AnnData file for calculating Pearson correlation")
    parser.add_argument("--output_path", type=str, default='./output/gene_analysis/')
    parser.add_argument("--target_cell_type", type=str, default='Macro_IL1B')
    parser.add_argument("--cell_type_column", type=str, default='cell_type')
    parser.add_argument("--gene_map_path", type=str, default="../tokenizer/token_files/gene_to_ensembl_mapping_95m.json")
    parser.add_argument("--leiden_resolutions", type=str, default="0.2, 0.5, 0.7")
    parser.add_argument("--grn_thresholds", type=str, default="0.5, 0.6, 0.7, 0.8")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding_batch_size", type=int, default=32)
    return parser.parse_args()


def set_seed(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(
    seed) if torch.cuda.is_available() else None


def load_and_invert_gene_map(filepath):
    logging.info(f"Loading gene map file from '{filepath}'...");
    try:
        with open(filepath, 'r') as f:
            symbol_to_ensembl = json.load(f)
    except FileNotFoundError:
        logging.error(f"Error: Gene map file not found at '{filepath}'.");
        exit()
    except json.JSONDecodeError:
        logging.error(f"Error: File '{filepath}' is not a valid JSON file.");
        exit()
    logging.info("Creating reverse mapping from Ensembl ID -> Gene Symbol...");
    ensembl_to_symbol_map = {}
    for symbol, ensembl_id in symbol_to_ensembl.items():
        if ensembl_id not in ensembl_to_symbol_map: ensembl_to_symbol_map[ensembl_id] = symbol
    logging.info(f"Successfully created {len(ensembl_to_symbol_map)} Ensembl -> Symbol mappings.");
    return ensembl_to_symbol_map


def get_gene_embeddings_for_cell_type(model, dataset, tokenizer, target_cell_type, cell_type_column, batch_size,
                                      device):
    logging.info(f"Starting to extract gene embeddings for cell type '{target_cell_type}'...")
    cell_type_dataset = dataset.filter(lambda example: example[cell_type_column] == target_cell_type, num_proc=4)
    if len(cell_type_dataset) == 0: raise ValueError(
        f"Cell type '{target_cell_type}' not found in the '{cell_type_column}' column of the dataset.")
    logging.info(f"Found {len(cell_type_dataset)} '{target_cell_type}' cells.")
    model.eval();
    gene_embeddings_sum = defaultdict(lambda: np.zeros(model.config.hidden_size));
    gene_counts = defaultdict(int)
    pad_token_id = tokenizer.gene_token_dict.get("<pad>", 0)
    with torch.no_grad():
        for i in range(0, len(cell_type_dataset), batch_size):
            batch_data = cell_type_dataset[i:i + batch_size];
            input_ids_list = [torch.tensor(ids) for ids in batch_data['input_ids']]
            padded_input_ids = rnn_utils.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id).to(
                device)
            attention_mask = (padded_input_ids != pad_token_id).long().to(device)
            outputs = model(input_ids=padded_input_ids, attention_mask=attention_mask)
            last_hidden_states = outputs.last_hidden_state.cpu().numpy()
            for j in range(len(input_ids_list)):
                actual_tokens = input_ids_list[j]
                for token_idx, token_id_tensor in enumerate(actual_tokens):
                    token_id = token_id_tensor.item()
                    if token_id != pad_token_id: gene_embeddings_sum[token_id] += last_hidden_states[j, token_idx, :];
                    gene_counts[token_id] += 1
    logging.info("All cells processed, calculating average embedding for each gene...");
    avg_gene_embeddings = {}
    for token_id, total_embedding in gene_embeddings_sum.items():
        if gene_counts[token_id] > 0: avg_gene_embeddings[token_id] = total_embedding / gene_counts[token_id]
    id_to_gene = {v: k for k, v in tokenizer.gene_token_dict.items()}
    gene_list_ensembl = [id_to_gene[t] for t in sorted(avg_gene_embeddings.keys()) if
                         id_to_gene.get(t) and not id_to_gene[t].startswith("<")]
    embedding_matrix = np.array([avg_gene_embeddings[tokenizer.gene_token_dict[gene]] for gene in gene_list_ensembl])
    logging.info(f"Extraction complete! Final gene embedding matrix shape: {embedding_matrix.shape}, corresponding to {len(gene_list_ensembl)} genes.");
    return embedding_matrix, gene_list_ensembl


def run_gene_module_task(gene_adata_base, gene_list_ensembl, output_dir, resolution, target_cell_type, dataset_name):
    logging.info(f"--- Starting Gene Module Discovery Task (Resolution: {resolution}) ---")
    task_dir = output_dir / "gene_module_discovery" / f"resolution_{resolution}";
    task_dir.mkdir(exist_ok=True, parents=True)
    gene_adata = gene_adata_base.copy();
    leiden_key = f'leiden_{resolution}'
    sc.tl.leiden(gene_adata, resolution=resolution, key_added=leiden_key, random_state=42);
    num_modules = len(gene_adata.obs[leiden_key].unique())
    logging.info(f"Discovered {num_modules} gene modules.")
    logging.info("Generating and saving UMAP plot of gene modules (PNG and PDF)...")
    fig, ax = plt.subplots(figsize=(15, 15))
    sc.pl.umap(gene_adata, color=leiden_key, legend_loc='on data', legend_fontsize=32, show=False, ax=ax)
    ax.set_title(f"Gene Modules in {target_cell_type} cell ({dataset_name})\n(Leiden, res={resolution})", fontsize=32)
    output_basename = task_dir / f"umap_gene_modules_res_{resolution}"
    plt.savefig(f"{output_basename}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_basename}.pdf", bbox_inches='tight')
    plt.close(fig)
    gene_modules_df = pd.DataFrame({'ensembl_id': gene_list_ensembl, 'gene_symbol': gene_adata.obs_names,
                                    'gene_modules': gene_adata.obs[leiden_key].values})
    gene_modules_df.to_csv(task_dir / f"gene_modules_res_{resolution}.csv", index=False)
    logging.info(f"Gene module discovery task (res={resolution}) complete! Results saved in: {task_dir}")
    return {"num_gene_modules": num_modules, "resolution": resolution}


def run_grn_inference_task(similarity_df, output_dir, threshold, target_cell_type, dataset_name):
    logging.info(f"--- Starting Gene Regulatory Network Inference Task (Threshold: {threshold}) ---")
    task_dir = output_dir / "grn_inference" / f"threshold_{threshold}";
    task_dir.mkdir(exist_ok=True, parents=True)
    adj_matrix = similarity_df.copy();
    adj_matrix[adj_matrix <= threshold] = 0;
    np.fill_diagonal(adj_matrix.values, 0)
    G = nx.from_pandas_adjacency(adj_matrix);
    G.remove_edges_from(list(nx.selfloop_edges(G)));
    G.remove_nodes_from(list(nx.isolates(G)))
    num_nodes = G.number_of_nodes();
    num_edges = G.number_of_edges()
    if num_nodes == 0: logging.warning(f"At threshold {threshold}, there are no nodes in the network. Skipping this task."); return {"grn_nodes": 0,
                                                                                                     "grn_edges": 0,
                                                                                                     "top_hub_gene": None}
    logging.info(f"Network constructed: {num_nodes} nodes, {num_edges} edges.")
    degree_centrality = nx.degree_centrality(G);
    hub_genes_df = pd.DataFrame.from_dict(degree_centrality, orient='index', columns=['degree_centrality'])
    hub_genes_df = hub_genes_df.sort_values('degree_centrality', ascending=False);
    hub_genes_df.to_csv(task_dir / f"hub_genes_thresh_{threshold}.csv")
    logging.info("Top 10 Hub Genes:");
    logging.info(hub_genes_df.head(10))
    top_hub_gene_symbol = hub_genes_df.index[0] if not hub_genes_df.empty and hub_genes_df.iloc[0][
        'degree_centrality'] > 0 else None

    if top_hub_gene_symbol:
        TOP_N_NEIGHBORS = 25
        logging.info(f"Visualizing Top {TOP_N_NEIGHBORS} strongest neighbors of Top Hub Gene '{top_hub_gene_symbol}'...")

        neighbors_with_weights = []
        if top_hub_gene_symbol in G:
            for neighbor in G.neighbors(top_hub_gene_symbol):
                weight = G[top_hub_gene_symbol][neighbor].get('weight', 0)
                neighbors_with_weights.append((neighbor, weight))

        neighbors_with_weights.sort(key=lambda x: x[1], reverse=True)
        top_neighbors = [neighbor for neighbor, weight in neighbors_with_weights[:TOP_N_NEIGHBORS]]
        nodes_for_viz = [top_hub_gene_symbol] + top_neighbors
        ego_graph_viz = G.subgraph(nodes_for_viz)

        plt.figure(figsize=(20, 20))
        pos = nx.spring_layout(ego_graph_viz, seed=42, k=1.2, iterations=100)

        node_colors = ['tomato' if node == top_hub_gene_symbol else 'skyblue' for node in ego_graph_viz.nodes()]
        node_sizes = [9000 if node == top_hub_gene_symbol else 7000 for node in ego_graph_viz.nodes()]
        nx.draw_networkx_nodes(ego_graph_viz, pos, node_color=node_colors, node_size=node_sizes)

        nx.draw_networkx_edges(ego_graph_viz, pos, edge_color='gray', width=0.8, alpha=0.7)

        neighbor_labels = {node: node for node in top_neighbors}
        nx.draw_networkx_labels(ego_graph_viz, pos, labels=neighbor_labels, font_size=18, font_family='sans-serif')

        hub_label = {top_hub_gene_symbol: top_hub_gene_symbol}
        nx.draw_networkx_labels(ego_graph_viz, pos, labels=hub_label, font_size=20, font_weight='bold',
                                font_family='sans-serif')

        plt.title(
            f"Top {len(top_neighbors)} Neighbors of Hub Gene: {top_hub_gene_symbol}\nin {target_cell_type} cell ({dataset_name}, Thresh={threshold})",
            fontsize=28)
        plt.axis('off')

        output_basename = task_dir / f"hub_gene_{top_hub_gene_symbol}_top_{TOP_N_NEIGHBORS}_neighbors"
        plt.savefig(f"{output_basename}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_basename}.pdf", bbox_inches='tight')
        plt.close()

    logging.info(f"GRN inference task (thresh={threshold}) complete! Results saved in: {task_dir}")
    return {"grn_nodes": num_nodes, "grn_edges": num_edges, "top_hub_gene": top_hub_gene_symbol}


def run_grn_topology_scan(similarity_df, output_dir, target_cell_type, dataset_name):
    logging.info("--- Starting GRN Topology Property Scan Task ---")
    task_dir = output_dir / "grn_topology_scan"
    task_dir.mkdir(exist_ok=True, parents=True)

    scan_thresholds = np.arange(0.5, 0.91, 0.025)
    topology_results = []

    for thresh in scan_thresholds:
        adj_matrix = similarity_df.copy()
        adj_matrix[adj_matrix <= thresh] = 0
        np.fill_diagonal(adj_matrix.values, 0)
        G = nx.from_pandas_adjacency(adj_matrix)
        G.remove_edges_from(list(nx.selfloop_edges(G)))

        num_edges = G.number_of_edges()

        if G.number_of_nodes() == 0:
            topology_results.append({'threshold': thresh, 'num_edges': 0, 'avg_clustering': 0, 'lcc_size': 0})
            continue

        avg_clustering = nx.average_clustering(G)

        if num_edges > 0:
            largest_cc = max(nx.connected_components(G), key=len)
            lcc_size = len(largest_cc)
        else:
            lcc_size = 0 if G.number_of_nodes() == 0 else 1

        topology_results.append({
            'threshold': thresh,
            'num_edges': num_edges,
            'avg_clustering': avg_clustering,
            'lcc_size': lcc_size
        })

    results_df = pd.DataFrame(topology_results)
    results_df.to_csv(task_dir / "grn_topology_scan_results.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle(f'GRN Topology vs. Similarity Threshold\n({target_cell_type} cell from {dataset_name})', fontsize=32)

    axes[0].plot(results_df['threshold'], results_df['num_edges'], 'o-')
    axes[0].set_ylabel('Number of Edges')
    axes[0].set_yscale('log')
    axes[0].grid(True, which="both", ls="--", alpha=0.6)
    axes[0].text(-0.1, 1.05, '(a)', transform=axes[0].transAxes, size=24, weight='bold')

    axes[1].plot(results_df['threshold'], results_df['avg_clustering'], 'o-')
    axes[1].set_ylabel('Average Clustering Coefficient')
    axes[1].grid(True, ls="--", alpha=0.6)
    axes[1].text(-0.1, 1.05, '(b)', transform=axes[1].transAxes, size=24, weight='bold')

    axes[2].plot(results_df['threshold'], results_df['lcc_size'], 'o-')
    axes[2].set_xlabel('Similarity Threshold')
    axes[2].set_ylabel('Size of Largest\nConnected Component')
    axes[2].grid(True, ls="--", alpha=0.6)
    axes[2].text(-0.1, 1.05, '(c)', transform=axes[2].transAxes, size=24, weight='bold')

    for ax in axes:
        ax.axvline(x=0.7, color='r', linestyle='--', linewidth=2, label='Selected Threshold (0.7)')
        ax.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_basename = task_dir / "grn_topology_scan_labeled"
    plt.savefig(f"{output_basename}.png", dpi=300)
    plt.savefig(f"{output_basename}.pdf")
    plt.close()
    logging.info(f"Labeled topology scan plot saved to: {task_dir.resolve()}")


def run_module_quantitative_evaluation(gene_embeddings, gene_list_symbols, adata_cell_type, output_dir,
                                       target_cell_type, dataset_name):
    logging.info("--- Starting Gene Module Quantitative Evaluation Task ---")
    task_dir = output_dir / "quantitative_evaluation"
    task_dir.mkdir(exist_ok=True, parents=True)
    logging.info("Calculating Pearson correlation as a baseline...")
    common_genes = list(set(gene_list_symbols) & set(adata_cell_type.var_names))
    adata_common = adata_cell_type[:, common_genes].copy()
    expression_df = pd.DataFrame(adata_common.X.toarray() if "toarray" in dir(adata_common.X) else adata_common.X,
                                 columns=common_genes, index=adata_common.obs_names)
    pearson_corr_matrix = expression_df.corr(method='pearson').fillna(0)
    logging.info("Calculating cosine similarity of scDMC embeddings...")
    gene_to_idx_map = {gene: i for i, gene in enumerate(gene_list_symbols)}
    gene_indices_in_emb = [gene_to_idx_map[g] for g in common_genes]
    embeddings_common = gene_embeddings[gene_indices_in_emb, :]
    cosine_sim_matrix = cosine_similarity(embeddings_common)
    cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=common_genes, columns=common_genes)
    logging.info("Fetching KEGG pathway gene sets from gseapy and calculating internal consistency...")
    kegg_gene_sets = gp.get_library(name='KEGG_2021_Human', organism='Human')
    results = []
    for term, genes in kegg_gene_sets.items():
        pathway_genes_in_common = [g for g in genes if g in common_genes]
        if len(pathway_genes_in_common) < 10: continue
        sub_pearson_corr = pearson_corr_matrix.loc[pathway_genes_in_common, pathway_genes_in_common].to_numpy()
        sub_cosine_sim = cosine_sim_df.loc[pathway_genes_in_common, pathway_genes_in_common].to_numpy()
        iu = np.triu_indices(len(pathway_genes_in_common), k=1)
        avg_pearson_corr = np.mean(np.abs(sub_pearson_corr[iu]));
        avg_cosine_sim = np.mean(sub_cosine_sim[iu])
        results.append({"Term": term, "scDMC_Similarity": avg_cosine_sim, "Pearson_Correlation": avg_pearson_corr})
    results_df = pd.DataFrame(results);
    results_df.to_csv(task_dir / "pathway_consistency_results.csv", index=False)
    scdmc_wins = (results_df['scDMC_Similarity'] > results_df['Pearson_Correlation']).sum()
    total_terms = len(results_df)
    win_percentage = (scdmc_wins / total_terms) * 100 if total_terms > 0 else 0
    avg_scdmc_sim = results_df['scDMC_Similarity'].mean()
    avg_pearson_corr = results_df['Pearson_Correlation'].mean()
    ttest_result = ttest_rel(results_df['scDMC_Similarity'],
                             results_df['Pearson_Correlation']) if total_terms > 1 else None
    logging.info("\n--- Quantitative Evaluation Results ---")
    logging.info(f"Compared a total of {total_terms} KEGG pathways.")
    logging.info(f"Proportion of pathways where scDMC performed better: {win_percentage:.2f}% ({scdmc_wins}/{total_terms})")
    logging.info(f"scDMC average intra-pathway similarity: {avg_scdmc_sim:.4f}")
    logging.info(f"Pearson average intra-pathway correlation: {avg_pearson_corr:.4f}")
    if ttest_result: logging.info(
        f"Paired t-test result: t-statistic={ttest_result.statistic:.2f}, p-value={ttest_result.pvalue:.2e}")
    logging.info("\nPlotting quantitative evaluation comparison graph...")
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(data=results_df, x="Pearson_Correlation", y="scDMC_Similarity", ax=ax, alpha=0.6, edgecolor="k",
                    linewidth=0.5)
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='y=x')
    ax.set_title(f"Intra-Pathway Gene Similarity\n({target_cell_type} cell from {dataset_name})", fontsize=20,
                 weight='bold');
    ax.set_xlabel("Baseline (Avg. Pearson Correlation)", fontsize=20)
    ax.set_ylabel("scDMC (Avg. Cosine Similarity)", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower right', fontsize=20)
    plt.tight_layout()
    output_file = task_dir / "module_quality_comparison.png"
    plt.savefig(output_file, dpi=300)
    plt.savefig(task_dir / "module_quality_comparison.pdf")
    plt.close(fig)
    logging.info(f"Quantitative evaluation plot saved to: {task_dir.resolve()}")
    return {"win_percentage": win_percentage, "avg_scdmc_sim": avg_scdmc_sim, "avg_pearson_corr": avg_pearson_corr,
            "p_value": ttest_result.pvalue if ttest_result else np.nan}


def save_module_results_to_excel(args, module_results_list, base_output_path, model_name, dataset_name):
    if not module_results_list: return
    logging.info("\nAppending [Gene Module Discovery] results to Excel file...")
    excel_file_path = base_output_path / "gene_module_summary.xlsx";
    all_rows = []
    for result_item in module_results_list:
        new_row_data = {'model': model_name, 'dataset': dataset_name, 'target_cell_type': args.target_cell_type,
                        'leiden_resolution': result_item.get('resolution'),
                        'num_gene_modules': result_item.get('num_gene_modules'),
                        'seed': args.seed, 'date': datetime.datetime.now().strftime("%Y-%m-%d")}
        all_rows.append(new_row_data)
    df_new_rows = pd.DataFrame(all_rows)
    try:
        if excel_file_path.exists():
            df_existing = pd.read_excel(excel_file_path)
            df_combined = pd.concat([df_existing, df_new_rows], ignore_index=True).drop_duplicates(
                subset=['model', 'dataset', 'target_cell_type', 'leiden_resolution', 'seed'], keep='last')
        else:
            df_combined = df_new_rows
        df_combined.to_excel(excel_file_path, index=False)
        logging.info(f"Successfully saved/appended to: {excel_file_path}")
    except Exception as e:
        logging.error(f"Error: Failed to save to Excel file '{excel_file_path}': {e}")


def save_grn_results_to_excel(args, grn_results_list, base_output_path, model_name, dataset_name):
    if not grn_results_list: return
    logging.info("\nAppending [GRN Inference] results to Excel file...")
    excel_file_path = base_output_path / "grn_inference_summary.xlsx";
    all_rows = []
    for result_item in grn_results_list:
        new_row_data = {'model': model_name, 'dataset': dataset_name, 'target_cell_type': args.target_cell_type,
                        'grn_threshold': result_item.get('grn_threshold'), 'grn_nodes': result_item.get('grn_nodes'),
                        'grn_edges': result_item.get('grn_edges'), 'top_hub_gene': result_item.get('top_hub_gene'),
                        'seed': args.seed, 'date': datetime.datetime.now().strftime("%Y-%m-%d")}
        all_rows.append(new_row_data)
    df_new_rows = pd.DataFrame(all_rows)
    try:
        if excel_file_path.exists():
            df_existing = pd.read_excel(excel_file_path)
            df_combined = pd.concat([df_existing, df_new_rows], ignore_index=True).drop_duplicates(
                subset=['model', 'dataset', 'target_cell_type', 'grn_threshold', 'seed'], keep='last')
        else:
            df_combined = df_new_rows
        df_combined.to_excel(excel_file_path, index=False)
        logging.info(f"Successfully saved/appended to: {excel_file_path}")
    except Exception as e:
        logging.error(f"Error: Failed to save to Excel file '{excel_file_path}': {e}")


# --- 3. Main Function ---
def main():
    args = get_args()
    set_seed(args.seed)

    path_parts = Path(args.model_path).parts
    base_model_name = path_parts[-2] if len(path_parts) > 1 else "unknown_model"
    task_dataset_name = Path(args.tokenized_data_path).name
    # --- MODIFIED: Create a shorter dataset name for display purposes ---
    display_dataset_name = task_dataset_name.split('_')[0]

    run_output_dir = Path(
        args.output_path) / base_model_name / task_dataset_name / f"analysis_{args.target_cell_type.replace(' ', '_').replace('/', '_')}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Experiment started, all outputs will be saved in: {run_output_dir}")
    logging.info(f"Arguments: {args}")

    gene_map = load_and_invert_gene_map(args.gene_map_path)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model = BertModel.from_pretrained(args.model_path, output_hidden_states=True).to(device)
    tokenizer = TranscriptomeTokenizer()
    dataset = load_from_disk(args.tokenized_data_path)

    try:
        gene_embeddings, gene_list_ensembl = get_gene_embeddings_for_cell_type(model, dataset, tokenizer,
                                                                               args.target_cell_type,
                                                                               args.cell_type_column,
                                                                               args.embedding_batch_size, device)
    except Exception as e:
        logging.error(f"Error during gene embedding extraction: {e}", exc_info=True);
        return

    module_results_list = [];
    grn_results_list = []
    gene_list_symbols = [gene_map.get(eid, eid) for eid in gene_list_ensembl]

    if RUN_GENE_MODULE_DISCOVERY:
        logging.info("Pre-calculating for gene module discovery (neighbors graph and UMAP)...")
        gene_adata_base = sc.AnnData(gene_embeddings);
        gene_adata_base.obs_names = gene_list_symbols
        sc.pp.neighbors(gene_adata_base, n_neighbors=15, use_rep='X', random_state=args.seed)
        sc.tl.umap(gene_adata_base, random_state=args.seed)
        try:
            resolutions = [float(res.strip()) for res in args.leiden_resolutions.split(',')]
            logging.info(f"Will run the following resolutions for gene module discovery: {resolutions}")
        except ValueError:
            logging.error("Error: '--leiden_resolutions' parameter format is incorrect. Please ensure it is a comma-separated list of numbers (e.g., '0.2,0.5,1.0').")
        else:
            for res in resolutions:
                try:
                    # --- MODIFIED CALL: Pass display_dataset_name ---
                    module_results = run_gene_module_task(gene_adata_base, gene_list_ensembl, run_output_dir, res,
                                                          args.target_cell_type, display_dataset_name)
                    module_results_list.append(module_results)
                except Exception as e:
                    logging.error(f"An error occurred while executing the gene module discovery task for resolution {res}: {e}", exc_info=True)

    similarity_df = None
    if RUN_GRN_INFERENCE or RUN_GRN_TOPOLOGY_SCAN or RUN_MODULE_QUANTITATIVE_EVAL:
        logging.info("Pre-calculating for GRN and quantitative evaluation...")
        similarity_df = pd.DataFrame(cosine_similarity(gene_embeddings), index=gene_list_symbols,
                                     columns=gene_list_symbols)
        (run_output_dir / "grn_inference").mkdir(exist_ok=True)
        similarity_df.to_csv(run_output_dir / "grn_inference" / "gene_similarity_matrix_symbols.csv")

    if RUN_GRN_INFERENCE:
        try:
            thresholds = [float(thr.strip()) for thr in args.grn_thresholds.split(',')]
            logging.info(f"Will run the following thresholds for GRN inference: {thresholds}")
            for thr in thresholds:
                # --- MODIFIED CALL: Pass display_dataset_name ---
                grn_results = run_grn_inference_task(similarity_df, run_output_dir, thr, args.target_cell_type,
                                                     display_dataset_name)
                grn_results['grn_threshold'] = thr
                grn_results_list.append(grn_results)
        except ValueError:
            logging.error("Error: '--grn_thresholds' parameter format is incorrect.")

    if RUN_GRN_TOPOLOGY_SCAN:
        # --- MODIFIED CALL: Pass display_dataset_name ---
        run_grn_topology_scan(similarity_df, run_output_dir, args.target_cell_type, display_dataset_name)

    if RUN_MODULE_QUANTITATIVE_EVAL:
        try:
            adata_raw = sc.read_h5ad(args.raw_adata_path)
            adata_cell_type = adata_raw[adata_raw.obs[args.cell_type_column] == args.target_cell_type].copy()
            if adata_cell_type.shape[0] == 0:
                logging.warning(f"Cell type '{args.target_cell_type}' not found in the raw adata file, skipping quantitative evaluation.")
            else:
                sc.pp.normalize_total(adata_cell_type, target_sum=1e4)
                # --- MODIFIED CALL: Pass display_dataset_name ---
                eval_results = run_module_quantitative_evaluation(gene_embeddings, gene_list_symbols, adata_cell_type,
                                                                  run_output_dir, args.target_cell_type,
                                                                  display_dataset_name)
        except FileNotFoundError:
            logging.error(f"Error: Raw adata file not found: {args.raw_adata_path}. Quantitative evaluation task cannot be performed.")

    # Excel saving still uses the full task_dataset_name for clarity
    save_module_results_to_excel(args, module_results_list, Path(args.output_path), base_model_name, task_dataset_name)
    save_grn_results_to_excel(args, grn_results_list, Path(args.output_path), base_model_name, task_dataset_name)

    logging.info("All tasks completed!")


if __name__ == "__main__":
    main()