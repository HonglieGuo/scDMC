from epe_utils import EPETokenizer_norm_log1p

# datasets Path
input_directory = f"../datasets/data/tk_data"
output_directory = f"../datasets/finetune_datasets"
output_prefix = "tk60534_epe"

# 保留cell_type和organ_major两个维度，其余的都丢弃掉
tokenizer = EPETokenizer_norm_log1p(
    {"cell_type": "cell_type", "tissue": "organ_major"},
    nproc=8,
    chunk_size=2048,
    model_input_size=2048,
    special_token=True,
    collapse_gene_ids=True
)
# Tokenizer
tokenizer.tokenize_data(
    data_directory=input_directory,
    output_directory=output_directory,
    output_prefix=output_prefix,
    file_format="h5ad",  # "loom" or "h5ad", depending on your input_init file format
    use_generator=True,
)

print(f"Tokenized data Rank-Value-Token saved to {output_directory} with prefix {output_prefix}")
