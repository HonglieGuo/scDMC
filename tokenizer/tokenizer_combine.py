
from ExpressionPerceptionEnhanced_Tokenizer import TranscriptomeTokenizer, EPETokenizer_norm_log1p, CombinedTokenizer

# datasets Path
input_directory = f"../datasets/pretrain/pretrain_data"
output_directory = f"../datasets/pretrain/pretrain_datasets"
output_prefix = "part"

transcriptome_tokenizer = TranscriptomeTokenizer(
    nproc=16,
    chunk_size=2048,
    model_input_size=2048,
    special_token=True
)
norm_log1p_tokenizer = EPETokenizer_norm_log1p(
    nproc=16,
    chunk_size=2048,
    model_input_size=2048,
    special_token=True
)

combined_tokenizer = CombinedTokenizer(transcriptome_tokenizer, norm_log1p_tokenizer)

combined_tokenizer.tokenize_data(
    data_directory=input_directory,
    output_directory=output_directory,
    output_prefix=output_prefix,
    file_format="h5ad",
    use_generator=True,
    overwrite=False,
)

print(f"Tokenized data Combine-Rank-Expression-Token saved to {output_directory} with prefix {output_prefix}")
