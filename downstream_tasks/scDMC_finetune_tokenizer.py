from __future__ import annotations
import re
import logging
import os
import warnings
from collections import Counter
from pathlib import Path
from typing import Literal, Union
import pickle
import json
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from datasets import Dataset
from tqdm import tqdm
from datetime import datetime
import shutil
import loompy as lp  # noqa

# 忽略特定警告
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")  # noqa

logger = logging.getLogger(__name__)

# 定义文件路径，增加对 __file__ 是否存在的检查
if "__file__" in globals():
    BASE_DIR = Path(__file__).parent
else:
    BASE_DIR = Path.cwd()

GENE_MEDIAN_FILE = BASE_DIR / "../tokenizer/token_files/gene_median_dictionary_95m.pkl"
TOKEN_DICTIONARY_FILE = BASE_DIR / "../tokenizer/token_files/token_ids_60534.json"
ENSEMBL_MAPPING_FILE = BASE_DIR / "../tokenizer/token_files/gene_to_ensembl_mapping_95m.pkl"

def rank_genes(gene_vector, gene_tokens):
    sorted_indices = np.argsort(-gene_vector)
    return gene_tokens[sorted_indices]

def tokenize_cell(gene_vector, gene_tokens):
    nonzero_mask = np.nonzero(gene_vector)[0]
    return rank_genes(gene_vector[nonzero_mask], gene_tokens[nonzero_mask])

def sum_ensembl_ids(
        data_directory: Union[Path, str],
        collapse_gene_ids: bool,
        gene_mapping_dict: dict,
        gene_token_dict: dict,
        file_format: Literal["loom", "h5ad"] = "loom",
        chunk_size: int = 512,
):
    data_directory = Path(data_directory)
    if file_format == "loom":
        try:
            with lp.connect(str(data_directory)) as data:
                # 检查必需的属性
                if "ensembl_id" not in data.ra.keys():
                    raise KeyError("'ensembl_id' column missing from adata.ra.keys()")

                if "ensembl_id_collapsed" in data.ra.keys():
                    raise KeyError("'ensembl_id_collapsed' column already exists in adata.ra.keys()")

                gene_ids_in_dict = [
                    gene for gene in data.ra["ensembl_id"] if gene in gene_token_dict
                ]
                if not collapse_gene_ids:
                    if len(gene_ids_in_dict) == len(set(gene_ids_in_dict)):
                        return data_directory
                    else:
                        raise ValueError("Error: adata Ensembl IDs non-unique.")

                gene_ids_collapsed = [
                    gene_mapping_dict.get(gene_id.upper(), gene_id)
                    for gene_id in data.ra["ensembl_id"]
                ]
                gene_ids_collapsed_in_dict = [
                    gene for gene in gene_ids_collapsed if gene in gene_token_dict
                ]

                if len(set(gene_ids_in_dict)) == len(set(gene_ids_collapsed_in_dict)):
                    data.ra["ensembl_id_collapsed"] = gene_ids_collapsed
                    return data_directory
                else:
                    dedup_filename = data_directory.with_name(
                        f"{data_directory.stem}__dedup.loom"
                    )
                    data.ra["ensembl_id_collapsed"] = gene_ids_collapsed
                    dup_genes = [
                        idx
                        for idx, count in Counter(data.ra["ensembl_id_collapsed"]).items()
                        if count > 1
                    ]
                    num_chunks = int(np.ceil(data.shape[1] / chunk_size))
                    first_chunk = True
                    for _, _, view in tqdm(
                            data.scan(axis=1, batch_size=chunk_size), total=num_chunks
                    ):

                        def process_chunk(view, duplic_genes):
                            data_count_view = pd.DataFrame(
                                view, index=data.ra["ensembl_id_collapsed"]
                            )
                            unique_data_df = data_count_view.loc[
                                ~data_count_view.index.isin(duplic_genes)
                            ]
                            dup_data_df = data_count_view.loc[
                                data_count_view.index.isin(
                                    [i for i in duplic_genes if "None" not in i]
                                )
                            ]
                            summed_data = dup_data_df.groupby(dup_data_df.index).sum()
                            if not summed_data.index.is_unique:
                                raise ValueError(
                                    "Error: Ensembl IDs in summed adata frame non-unique."
                                )
                            data_count_view = pd.concat(
                                [unique_data_df, summed_data], axis=0
                            )
                            if not data_count_view.index.is_unique:
                                raise ValueError(
                                    "Error: Ensembl IDs in final adata frame non-unique."
                                )
                            return data_count_view

                        processed_chunk = process_chunk(view[:, :], dup_genes)
                        processed_array = processed_chunk.to_numpy()
                        new_row_attrs = {"ensembl_id_collapsed": processed_chunk.index.to_numpy()}

                        if "n_counts" not in view.ca.keys():
                            total_count_view = np.sum(view[:, :], axis=0).astype(int)
                            view.ca["n_counts"] = total_count_view

                        if first_chunk:
                            lp.create(
                                str(dedup_filename),
                                processed_array,
                                row_attrs=new_row_attrs,
                                col_attrs=view.ca,
                            )
                            first_chunk = False
                        else:
                            with lp.connect(str(dedup_filename), mode="r+") as dsout:
                                dsout.add_columns(processed_array, col_attrs=view.ca)
                    return dedup_filename

        except Exception as e:
            logger.error(f"Error processing loom file {data_directory}: {e}")
            raise

    elif file_format == "h5ad":
        try:
            data = sc.read_h5ad(str(data_directory))

            if "ensembl_id" not in data.var.columns:
                raise KeyError("'ensembl_id' column missing from adata.var")

            if "ensembl_id_collapsed" in data.var.columns:
                raise KeyError("'ensembl_id_collapsed' column already exists in adata.var")

            gene_ids_in_dict = [
                gene for gene in data.var["ensembl_id"] if gene in gene_token_dict
            ]
            if not collapse_gene_ids:
                if len(gene_ids_in_dict) == len(set(gene_ids_in_dict)):
                    return data
                else:
                    raise ValueError("Error: adata Ensembl IDs non-unique.")

            gene_ids_collapsed = [
                gene_mapping_dict.get(gene_id.upper(), gene_id)
                for gene_id in data.var["ensembl_id"]
            ]
            gene_ids_collapsed_in_dict = [
                gene for gene in gene_ids_collapsed if gene in gene_token_dict
            ]

            if len(set(gene_ids_in_dict)) == len(set(gene_ids_collapsed_in_dict)):
                data.var["ensembl_id_collapsed"] = data.var["ensembl_id"].map(gene_mapping_dict)
                # 确保 var_names 唯一
                data.var_names = data.var["ensembl_id_collapsed"].astype(str)
                data.var_names_make_unique()
                return data

            else:
                data.var["ensembl_id_collapsed"] = gene_ids_collapsed
                data.var_names = gene_ids_collapsed
                # 确保 var_names 唯一
                data.var_names_make_unique()
                data = data[:, ~data.var.index.isna()]
                dup_genes = [
                    idx for idx, count in Counter(data.var_names).items() if count > 1
                ]
                if dup_genes:
                    num_chunks = int(np.ceil(data.shape[0] / chunk_size))
                    processed_genes = []
                    for i in tqdm(range(num_chunks)):
                        start_idx = i * chunk_size
                        end_idx = min((i + 1) * chunk_size, data.shape[0])
                        data_chunk = data[start_idx:end_idx, :]

                        processed_chunks = []
                        for dup_gene in dup_genes:
                            data_dup_gene = data_chunk[:, data_chunk.var_names == dup_gene]
                            if data_dup_gene.shape[1] == 0:
                                continue
                            df = pd.DataFrame.sparse.from_spmatrix(
                                data_dup_gene.X,
                                index=data_dup_gene.obs_names,
                                columns=data_dup_gene.var_names,
                            )
                            df_sum = pd.DataFrame(df.sum(axis=1))
                            df_sum.columns = [dup_gene]
                            df_sum.index = data_dup_gene.obs.index
                            processed_chunks.append(df_sum)

                        if processed_chunks:
                            processed_chunks = pd.concat(processed_chunks, axis=1)
                            processed_genes.append(processed_chunks)
                    if processed_genes:
                        processed_genes = pd.concat(processed_genes, axis=0)
                        var_df = pd.DataFrame({"ensembl_id_collapsed": processed_genes.columns})
                        var_df.index = processed_genes.columns
                        processed_genes = sc.AnnData(X=processed_genes, obs=data.obs, var=var_df)

                        data_dedup = data[:, ~data.var.index.isin(dup_genes)]
                        data_dedup = sc.concat([data_dedup, processed_genes], axis=1)
                        data_dedup.obs = data.obs
                        return data_dedup
                    else:
                        return data
                else:
                    return data

        except Exception as e:
            logger.error(f"Error processing h5ad file {data_directory}: {e}")
            raise
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


class TranscriptomeTokenizer:
    # --- 添加 model_input_names 属性 ---
    # 假设你的BERT模型需要这三个标准输入
    model_input_names = ["input_ids", "token_type_ids", "attention_mask"]
    # 如果你确定 group_by_length 只需要知道 input_ids 来确定长度，可以简化为:
    # model_input_names = ["input_ids"]
    # ------------------------------------

    def __init__(
            self,
            custom_attr_name_dict: dict = None,
            nproc: int = 1,
            chunk_size: int = 512,
            model_input_size: int = 2048,
            special_token: bool = True,
            collapse_gene_ids: bool = True,
            gene_median_file: Union[Path, str] = GENE_MEDIAN_FILE,
            token_dictionary_file: Union[Path, str] = TOKEN_DICTIONARY_FILE,
            gene_mapping_file: Union[Path, str] = ENSEMBL_MAPPING_FILE,
    ):
        self.custom_attr_name_dict = custom_attr_name_dict
        self.nproc = nproc
        self.chunk_size = chunk_size
        self.model_input_size = model_input_size
        self.special_token = special_token
        self.collapse_gene_ids = collapse_gene_ids
        self.gene_median_file_path = Path(gene_median_file)
        self.token_dictionary_file_path = Path(token_dictionary_file)
        self.gene_mapping_file_path = Path(gene_mapping_file) if gene_mapping_file else None

        # 加载 gene_median_dict
        try:
            with open(gene_median_file, "rb") as f:
                self.gene_median_dict = pickle.load(f)
        except (FileNotFoundError, pickle.UnpicklingError) as e:
            logger.error(f"无法加载基因中位数文件 {gene_median_file}: {e}")
            raise

        # 加载 gene_token_dict
        try:
            with open(token_dictionary_file, "r", encoding="utf-8") as f:
                self.gene_token_dict = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"无法加载 token 字典文件 {token_dictionary_file}: {e}")
            raise

        # 检查特殊标记
        if self.special_token:
            if ("<cls>" not in self.gene_token_dict) or ("<eos>" not in self.gene_token_dict):
                logger.error("<cls> 和 <eos> 是 special_token 为 True 时所需。")
                raise ValueError("<cls> 和 <eos> 标记在 gene_token_dict 中缺失。")
        else:
            if ("<cls>" in self.gene_token_dict) and ("<eos>" in self.gene_token_dict):
                logger.warning(
                    "<cls> 和 <eos> 在 gene_token_dict 中，但 special_token 为 False。"
                )

        # 加载 gene_mapping_dict
        if gene_mapping_file is not None:
            try:
                with open(gene_mapping_file, "rb") as f:
                    self.gene_mapping_dict = pickle.load(f)
            except (FileNotFoundError, pickle.UnpicklingError) as e:
                logger.error(f"无法加载基因映射文件 {gene_mapping_file}: {e}")
                raise
        else:
            self.gene_mapping_dict = {k: k for k in self.gene_token_dict.keys()}

        # 过滤 gene_mapping_dict 中不在 gene_token_dict 的键值
        gene_keys_set = set(self.gene_token_dict.keys())
        self.gene_mapping_dict = {
            k: v for k, v in self.gene_mapping_dict.items() if v in gene_keys_set
        }

        self.gene_keys = list(self.gene_token_dict.keys())
        self.genelist_dict = {k: True for k in self.gene_keys}

    def tokenize_data(
            self,
            data_directory: Union[Path, str],
            output_directory: Union[Path, str],
            output_prefix: str,
            file_format: Literal["loom", "h5ad"] = "loom",
            use_generator: bool = False,
    ):
        def generator_func():
            for example in self.tokenize_files(Path(data_directory), file_format):
                yield example

        tokenized_dataset = self.create_dataset(
            generator_func,
            use_generator=use_generator,
        )

        output_path = (Path(output_directory) / output_prefix).with_suffix(".dataset")
        # output_path = (Path(output_directory) / output_prefix).with_suffix(".parquet")

        tokenized_dataset.save_to_disk(str(output_path))
        # tokenized_dataset.to_parquet(str(output_path), compression='zstd', )

    def tokenize_files(
            self, data_directory: Path, file_format: Literal["loom", "h5ad"] = "loom"
    ):
        tokenize_file_fn = (
            self.tokenize_loom if file_format == "loom" else self.tokenize_anndata
        )
        file_paths = list(data_directory.glob(f"*.{file_format}"))

        def sort_key(f):
            match = re.search(r'partition_(\d+)', f.name)
            return int(match.group(1)) if match else 0

        file_paths = sorted(file_paths, key=sort_key)

        if not file_paths:
            logger.error(
                f"目录 {data_directory} 中未找到 .{file_format} 文件。"
            )
            raise FileNotFoundError(
                f"目录 {data_directory} 中未找到 .{file_format} 文件。"
            )

        for file_path in file_paths:
            logger.info(f"标记化处理 {file_path}")
            for example in tokenize_file_fn(file_path):
                yield example

    def tokenize_anndata(self, adata_file_path: Path, target_sum: int = 10_000):
        adata = sum_ensembl_ids(
            adata_file_path,
            self.collapse_gene_ids,
            self.gene_mapping_dict,
            self.gene_token_dict,
            file_format="h5ad",
            chunk_size=self.chunk_size,
        ).copy()  # 确保操作的是副本而非视图

        coding_miRNA_loc = np.where(
            [self.genelist_dict.get(i, False) for i in adata.var["ensembl_id_collapsed"].iloc[:]]
        )[0]

        norm_factor_vector = np.array([
            self.gene_median_dict.get(i, 1)  # 提供默认值以防止 KeyError
            for i in adata.var["ensembl_id_collapsed"].iloc[coding_miRNA_loc]
        ])

        coding_miRNA_ids = adata.var["ensembl_id_collapsed"].iloc[coding_miRNA_loc]
        try:
            coding_miRNA_tokens = np.array([
                self.gene_token_dict[i] for i in coding_miRNA_ids
            ])
        except KeyError as e:
            logger.error(f"Missing gene token for Ensembl ID: {e}")
            raise

        filter_pass_loc = np.arange(adata.shape[0])
        if "filter_pass" in adata.obs:
            filter_pass_loc = np.where(adata.obs["filter_pass"] == 1)[0]
        else:
            logger.info(
                f"{adata_file_path} 缺少 'filter_pass' 列属性；因此将对所有细胞进行标记化处理。"
            )

        for i in range(0, len(filter_pass_loc), self.chunk_size):
            idx = filter_pass_loc[i:i + self.chunk_size]

            n_counts = adata[idx].obs["n_counts"].values[:, None]
            X_view0 = adata[idx, :].X
            X_view = X_view0[:, coding_miRNA_loc]
            X_norm = X_view / n_counts * target_sum / norm_factor_vector
            X_norm = sp.csr_matrix(X_norm)

            for j in range(X_norm.shape[0]):
                tokens = rank_genes(X_norm[j].data, coding_miRNA_tokens[X_norm[j].indices])

                tokens = tokens[0:self.model_input_size - 2]

                if self.special_token:
                    tokens = [self.gene_token_dict.get("<cls>")] + tokens.tolist() + [self.gene_token_dict.get("<eos>")]

                example = {"input_ids": tokens}

                if self.custom_attr_name_dict is not None:
                    for attr_key, mapped_key in self.custom_attr_name_dict.items():
                        example[mapped_key] = adata[idx].obs[attr_key].iloc[j]

                yield example

    def tokenize_loom(self, loom_file_path: Path, target_sum: int = 10_000):
        try:
            with lp.connect(str(loom_file_path)) as data:
                coding_miRNA_loc = np.where(
                    [self.genelist_dict.get(i, False) for i in data.ra["ensembl_id_collapsed"]]
                )[0]
                norm_factor_vector = np.array(
                    [
                        self.gene_median_dict.get(i, 1)
                        for i in data.ra["ensembl_id_collapsed"][coding_miRNA_loc]
                    ]
                )
                coding_miRNA_ids = data.ra["ensembl_id_collapsed"][coding_miRNA_loc]
                try:
                    coding_miRNA_tokens = np.array([
                        self.gene_token_dict[i] for i in coding_miRNA_ids
                    ])
                except KeyError as e:
                    logger.error(f"Missing gene token for Ensembl ID: {e}")
                    raise

                filter_pass_loc = np.arange(data.shape[1])
                if "filter_pass" in data.ca:
                    filter_pass_loc = np.where(data.ca["filter_pass"] == 1)[0]
                else:
                    logger.info(
                        f"{loom_file_path} 缺少 'filter_pass' 列属性；因此将对所有细胞进行标记化处理。"
                    )

                for _, _, view in data.scan(
                        items=filter_pass_loc, axis=1, batch_size=self.chunk_size
                ):
                    subview = view.view[coding_miRNA_loc, :]
                    subview_norm_array = (
                            subview[:, :]
                            / subview.ca.n_counts
                            * target_sum
                            / norm_factor_vector[:, None]
                    )
                    subview_norm_array = sp.csr_matrix(subview_norm_array)

                    for i in range(subview_norm_array.shape[1]):
                        tokens = tokenize_cell(subview_norm_array[:, i].toarray().flatten(), coding_miRNA_tokens)

                        if self.special_token:
                            tokens = [self.gene_token_dict.get("<cls>")] + tokens.tolist() + [
                                self.gene_token_dict.get("<eos>")]

                        example = {"input_ids": tokens[0:self.model_input_size]}

                        if self.custom_attr_name_dict is not None:
                            for attr_key, mapped_key in self.custom_attr_name_dict.items():
                                example[mapped_key] = subview.ca[attr_key][i]

                        yield example

        except Exception as e:
            logger.error(f"Error processing loom file {loom_file_path}: {e}")
            raise

    def create_dataset(
            self,
            generator_func,
            use_generator: bool = False,
            keep_uncropped_input_ids: bool = False,
    ):
        logger.info("创建数据集。")
        try:
            if use_generator:
                output_dataset = Dataset.from_generator(generator_func, num_proc=self.nproc)
            else:
                # 修改为从生成器直接构建列表，适用于非生成器模式
                dataset_list = list(generator_func())
                if not dataset_list:
                    # 处理空数据集的情况
                    logger.warning("生成器未产生任何数据，返回空数据集。")
                    # 根据需要决定是返回空Dataset还是引发错误
                    # 例如，返回一个空的Dataset
                    return Dataset.from_dict({"input_ids": []}) # 假设至少需要input_ids列
                # 将列表转换为字典格式
                # 首先获取所有可能的键
                all_keys = set().union(*(d.keys() for d in dataset_list))
                dataset_dict = {key: [d.get(key) for d in dataset_list] for key in all_keys}
                output_dataset = Dataset.from_dict(dataset_dict)
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            raise

        def format_cell_features(example):
            if keep_uncropped_input_ids:
                example["input_ids_uncropped"] = example["input_ids"]
                example["length_uncropped"] = len(example["input_ids"])

            # 取消添加special tokens。因为在tokenize_anndata和tokenize_loom中已经添加
            example["length"] = len(example["input_ids"])

            return example

        try:
            output_dataset_truncated = output_dataset.map(
                format_cell_features, num_proc=self.nproc
            )
            return output_dataset_truncated
        except Exception as e:
            logger.error(f"Error formatting cell features: {e}")
            raise

    # --- 添加 save_pretrained 方法 ---
    def save_pretrained(self, save_directory: Union[str, Path]):
        """
        将分词器必要的文件和配置保存到指定目录。
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        saved_files = []

        # 1. 复制必要的文件
        files_to_copy = {
            "vocab_file": self.token_dictionary_file_path,
            # 根据需要取消注释并添加其他文件
            # "gene_median_file": self.gene_median_file_path,
            # "gene_mapping_file": self.gene_mapping_file_path,
        }

        tokenizer_config_files = {} # 用于记录文件名以写入 config

        for key, src_path in files_to_copy.items():
            if src_path and src_path.exists():
                try:
                    dst_path = save_directory / src_path.name
                    shutil.copyfile(src_path, dst_path)
                    logger.info(f"已将 {src_path.name} 复制到 {save_directory}")
                    saved_files.append(str(dst_path))
                    tokenizer_config_files[key] = src_path.name # 记录文件名
                except Exception as e:
                    logger.error(f"复制分词器文件 {src_path.name} 到 {save_directory} 时出错: {e}")
            elif src_path:
                 logger.warning(f"需要复制的分词器文件 {src_path} 不存在。")
            # else: 文件路径本身就是 None 或空字符串，忽略

        # 2. 创建 tokenizer_config.json
        tokenizer_config = {
            "tokenizer_class": self.__class__.__name__,
            "model_input_size": self.model_input_size,
            "special_token": self.special_token,
            "collapse_gene_ids": self.collapse_gene_ids,
            "custom_attr_name_dict": self.custom_attr_name_dict,
            # 添加复制的文件名到配置中
            **tokenizer_config_files
        }

        try:
            config_path = save_directory / "tokenizer_config.json"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(tokenizer_config, f, indent=2)
            logger.info(f"分词器配置已保存到 {config_path}")
            saved_files.append(str(config_path))
        except Exception as e:
            logger.error(f"保存分词器配置到 {config_path} 时出错: {e}")

        return saved_files


class EPETokenizer_norm_log1p(TranscriptomeTokenizer):
    # --- 添加 model_input_names 属性 ---
    # 这个子类可能生成不同的输入，但 Trainer 仍可能依赖父类的属性
    # 如果这个子类生成的输入字段不同，并且直接用于 Trainer，则需要在这里覆盖
    # model_input_names = ["input_ids", "expression_values", "token_type_ids", "attention_mask"] # 示例
    # 保持与父类一致通常是安全的，除非有明确理由覆盖
    # ------------------------------------

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def normalize_expression_values(self, expression_vals):
        expression_vals = np.log1p(expression_vals)
        return expression_vals

    def tokenize_anndata(self, adata_file_path: Path, target_sum: int = 10_000):
        adata = sum_ensembl_ids(
            adata_file_path,
            self.collapse_gene_ids,
            self.gene_mapping_dict,
            self.gene_token_dict,
            file_format="h5ad",
            chunk_size=self.chunk_size,
        )

        coding_miRNA_loc = np.where(
            [self.genelist_dict.get(i, False) for i in adata.var["ensembl_id_collapsed"].iloc[:]]
        )[0]
        norm_factor_vector = np.array([
            self.gene_median_dict.get(i, 1)
            for i in adata.var["ensembl_id_collapsed"].iloc[coding_miRNA_loc]
        ])
        coding_miRNA_ids = adata.var["ensembl_id_collapsed"].iloc[coding_miRNA_loc]
        try:
            coding_miRNA_tokens = np.array([
                self.gene_token_dict[i] for i in coding_miRNA_ids
            ])
        except KeyError as e:
            logger.error(f"Missing gene token for Ensembl ID: {e}")
            raise

        filter_pass_loc = np.arange(adata.shape[0])
        if "filter_pass" in adata.obs:
            filter_pass_loc = np.where(adata.obs["filter_pass"] == 1)[0]
        else:
            logger.info(
                f"{adata_file_path} 缺少 'filter_pass' 列属性；因此将对所有细胞进行标记化处理。"
            )

        for i in range(0, len(filter_pass_loc), self.chunk_size):
            idx = filter_pass_loc[i:i + self.chunk_size]

            n_counts = adata[idx].obs["n_counts"].values[:, None]
            X_view0 = adata[idx, :].X
            X_view = X_view0[:, coding_miRNA_loc]
            X_norm = X_view / n_counts * target_sum / norm_factor_vector
            X_norm = sp.csr_matrix(X_norm)

            for j in range(X_norm.shape[0]):
                row_expr_values = X_norm[j].data
                row_tokens = coding_miRNA_tokens[X_norm[j].indices].tolist()
                normalized_expr_values = self.normalize_expression_values(row_expr_values).tolist()

                row_tokens = row_tokens[0:self.model_input_size - 2]
                normalized_expr_values = normalized_expr_values[0:self.model_input_size - 2]
                if self.special_token:
                    row_tokens = [self.gene_token_dict.get("<cls>")] + row_tokens + [self.gene_token_dict.get("<eos>")]
                    normalized_expr_values = [1.0] + normalized_expr_values + [1.0]

                example = {
                    "input_ids": row_tokens,
                    "expression_values": normalized_expr_values,
                    "length": len(row_tokens[0:self.model_input_size])
                }

                if self.custom_attr_name_dict is not None:
                    for attr_key, mapped_key in self.custom_attr_name_dict.items():
                        example[mapped_key] = adata[idx].obs[attr_key].iloc[j]

                yield example

    def tokenize_loom(self, loom_file_path: Path, target_sum: int = 10_000):
        try:
            with lp.connect(str(loom_file_path)) as data:
                coding_miRNA_loc = np.where(
                    [self.genelist_dict.get(i, False) for i in data.ra["ensembl_id_collapsed"]]
                )[0]
                norm_factor_vector = np.array(
                    [
                        self.gene_median_dict.get(i, 1)
                        for i in data.ra["ensembl_id_collapsed"][coding_miRNA_loc]
                    ]
                )
                coding_miRNA_ids = data.ra["ensembl_id_collapsed"][coding_miRNA_loc]
                try:
                    coding_miRNA_tokens = np.array([
                        self.gene_token_dict[i] for i in coding_miRNA_ids
                    ])
                except KeyError as e:
                    logger.error(f"Missing gene token for Ensembl ID: {e}")
                    raise

                filter_pass_loc = np.arange(data.shape[1])
                if "filter_pass" in data.ca:
                    filter_pass_loc = np.where(data.ca["filter_pass"] == 1)[0]
                else:
                    logger.info(
                        f"{loom_file_path} 缺少 'filter_pass' 列属性；因此将对所有细胞进行标记化处理。"
                    )

                for _, _, view in data.scan(
                        items=filter_pass_loc, axis=1, batch_size=self.chunk_size
                ):
                    subview = view.view[coding_miRNA_loc, :]
                    subview_norm_array = (
                            subview[:, :]
                            / subview.ca.n_counts
                            * target_sum
                            / norm_factor_vector[:, None]
                    )
                    subview_norm_array = sp.csr_matrix(subview_norm_array)

                    for i in range(subview_norm_array.shape[1]):
                        row_expr_values = subview_norm_array[:, i].data
                        # --- 潜在问题：这里的 row_tokens 似乎获取的是单个 token，而不是序列 ---
                        # row_tokens = coding_miRNA_tokens[i] # 这看起来像是索引错误
                        # 应该获取与 row_expr_values 对应的 tokens
                        indices = subview_norm_array[:, i].indices
                        row_tokens = coding_miRNA_tokens[indices].tolist()
                        # --------------------------------------------------------------------
                        normalized_expr_values = self.normalize_expression_values(row_expr_values).tolist()

                        # --- 截断和添加特殊 token 的逻辑需要调整以适应列表 ---
                        row_tokens = row_tokens[0:self.model_input_size - 2]
                        normalized_expr_values = normalized_expr_values[0:self.model_input_size - 2]

                        if self.special_token:
                            # row_tokens = [self.gene_token_dict.get("<cls>")] + [row_tokens] + [ self.gene_token_dict.get("<eos>")] # 原来对单个 token 的操作
                            row_tokens = [self.gene_token_dict.get("<cls>")] + row_tokens + [ self.gene_token_dict.get("<eos>")] # 修改为对列表的操作
                            normalized_expr_values = [1.0] + normalized_expr_values + [1.0]
                        # ------------------------------------------------------

                        example = {
                            "input_ids": row_tokens[0:self.model_input_size], # 再次截断以防万一
                            "expression_values": normalized_expr_values[0:self.model_input_size],
                            # "length": len(row_tokens) # 长度应与截断后的 input_ids 匹配
                            "length": len(row_tokens[0:self.model_input_size])
                        }

                        if self.custom_attr_name_dict is not None:
                            for attr_key, mapped_key in self.custom_attr_name_dict.items():
                                example[mapped_key] = subview.ca[attr_key][i]

                        yield example

        except Exception as e:
            logger.error(f"Error processing loom file {loom_file_path}: {e}")
            raise

    def create_dataset_with_expression(
            self,
            generator_func,
            use_generator: bool = False,
    ):
        logger.info("创建包含表达值的数据集。")
        try:
            if use_generator:
                output_dataset = Dataset.from_generator(generator_func, num_proc=self.nproc)
            else:
                # 修改同上，处理非生成器模式
                dataset_list = list(generator_func())
                if not dataset_list:
                    logger.warning("生成器未产生任何数据，返回空数据集。")
                    # 假设至少需要 input_ids 和 expression_values
                    return Dataset.from_dict({"input_ids": [], "expression_values": []})
                all_keys = set().union(*(d.keys() for d in dataset_list))
                dataset_dict = {key: [d.get(key) for d in dataset_list] for key in all_keys}
                output_dataset = Dataset.from_dict(dataset_dict)
        except Exception as e:
            logger.error(f"Error creating dataset with expression: {e}")
            raise

        def format_cell_features(example):
            example["length"] = len(example["input_ids"])
            return example

        try:
            output_dataset_truncated = output_dataset.map(
                format_cell_features, num_proc=self.nproc
            )
            return output_dataset_truncated
        except Exception as e:
            logger.error(f"Error formatting cell features with expression: {e}")
            raise


class CombinedTokenizer:
    # --- 添加 model_input_names 属性 ---
    # 这个类组合了两个 tokenizer，Trainer 交互的是这个类的实例吗？
    # 如果 Trainer 直接使用 CombinedTokenizer 实例，那么这里也需要 model_input_names
    # 需要确定 Trainer 最终依赖哪个字段来确定长度
    # 假设 Trainer 最终需要 'input_ids_epe' 或 'input_ids_rank' 来确定长度
    model_input_names = ["input_ids_epe"] # 或者 ["input_ids_rank"]，取决于哪个是主要序列
    # ------------------------------------

    def __init__(self, transcriptome_tokenizer: TranscriptomeTokenizer, norm_log1p_tokenizer: EPETokenizer_norm_log1p):
        self.transcriptome_tokenizer = transcriptome_tokenizer
        self.norm_log1p_tokenizer = norm_log1p_tokenizer

    def tokenize_data(
            self,
            data_directory: Union[Path, str],
            output_directory: Union[Path, str],
            output_prefix: str,
            file_format: Literal["loom", "h5ad"] = "h5ad",
            use_generator: bool = False,
            overwrite: bool = False,
    ):
        def combined_generator():
            transcriptome_gen = self.transcriptome_tokenizer.tokenize_files(Path(data_directory), file_format)
            norm_log1p_gen = self.norm_log1p_tokenizer.tokenize_files(Path(data_directory), file_format)

            while True:
                try:
                    example1 = next(transcriptome_gen)
                    example2 = next(norm_log1p_gen)
                    combined_example = {
                        "input_ids_rank": example1["input_ids"],
                        "input_ids_epe": example2["input_ids"],
                        "expression_values": example2["expression_values"],
                        "length": example2["length"] # 确保这个 length 与 group_by_length 依赖的字段匹配
                    }
                    # --- 重要: 确认 group_by_length 应该使用哪个 length ---
                    # 如果 group_by_length 依赖 input_ids_epe，则 length 应该基于它
                    # combined_example["length"] = len(example2["input_ids"])
                    # 如果 group_by_length 依赖 input_ids_rank，则 length 应该基于它
                    # combined_example["length"] = len(example1["input_ids"])
                    # 当前代码依赖 example2["length"]，这可能来自 EPETokenizer_norm_log1p 的计算
                    # 需要确保 EPETokenizer_norm_log1p 中的 length 计算与 Trainer 期望的一致
                    # -----------------------------------------------------
                    yield combined_example
                except StopIteration:
                    break
                except Exception as e:
                    logger.error(f"Error combining examples: {e}")
                    raise

        tokenized_dataset = self.create_combined_dataset(
            combined_generator,
            use_generator=use_generator,
            overwrite=overwrite # 传递 overwrite 参数
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # output_path = (Path(output_directory) / f"{output_prefix}_{timestamp}").with_suffix(".dataset")
        output_path = (Path(output_directory) / f"{output_prefix}_{timestamp}").with_suffix(".parquet")

        self._handle_overwrite(output_path, overwrite)
        try:
            # tokenized_dataset.save_to_disk(str(output_path))
            tokenized_dataset.to_parquet(str(output_path), compression="zstd", )

            logger.info(f"Combined dataset saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving combined dataset to {output_path}: {e}")
            raise

    def _handle_overwrite(self, output_path: Path, overwrite: bool):
        if os.path.exists(str(output_path)):
            if overwrite:
                try:
                    # 检查是文件还是目录
                    if output_path.is_dir():
                        shutil.rmtree(str(output_path))
                    elif output_path.is_file():
                        output_path.unlink()
                    else:
                        # 如果路径存在但既不是文件也不是目录（例如符号链接），也尝试删除
                        output_path.unlink(missing_ok=True) # Python 3.8+
                    logger.info(f"Overwriting existing path at {output_path}")
                except Exception as e:
                    logger.error(f"Error overwriting path at {output_path}: {e}")
                    raise
            else:
                raise FileExistsError(
                    f"The output path {output_path} already exists. Use overwrite=True to replace it or provide a different output path."
                )
        elif not output_path.parent.exists():
             # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)


    def create_combined_dataset(
            self,
            combined_gen,
            use_generator: bool = False,
            overwrite: bool = False # 接收 overwrite 参数，虽然在此方法中未使用
    ):
        logger.info("创建合并的数据集...")
        try:
            if use_generator:
                # 注意：from_generator 可能不支持 num_proc
                output_dataset = Dataset.from_generator(combined_gen)
            else:
                # 修改同上
                dataset_list = list(combined_gen)
                if not dataset_list:
                    logger.warning("生成器未产生任何数据，返回空数据集。")
                    # 假设至少需要 input_ids_rank, input_ids_epe, expression_values, length
                    return Dataset.from_dict({"input_ids_rank": [], "input_ids_epe": [], "expression_values": [], "length": []})
                all_keys = set().union(*(d.keys() for d in dataset_list))
                dataset_dict = {key: [d.get(key) for d in dataset_list] for key in all_keys}
                output_dataset = Dataset.from_dict(dataset_dict)
            return output_dataset
        except Exception as e:
            logger.error(f"Error creating combined dataset: {e}")
            raise