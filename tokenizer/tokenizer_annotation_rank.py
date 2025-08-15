from ExpressionPerceptionEnhanced_Tokenizer import TranscriptomeTokenizer

# datasets Path
input_directory = f"../datasets/data/tk_data"
output_directory = f"../datasets/downstream"
output_prefix = "zheng68k_tk60534"
# 20275
# 25426


tokenizer = TranscriptomeTokenizer(
    {
    # "str_labels": "str_labels",
    "cell_type": "cell_type",
    # "tissue": "organ_major",
    # "batch": "batch",
    # "n_counts": "n_counts",
    # "labels":"labels",
    # "n_genes": "n_genes"
    },
    nproc=8,
    chunk_size=1024,
    model_input_size=2048,
    collapse_gene_ids=True,
    # special_token=False,
    special_token=True
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