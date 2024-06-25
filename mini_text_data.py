import numpy as np
import tiktoken

file_path = "arxivcs_researchcorpus.txt"

with open(file_path, "r", encoding="utf-8") as file_write:
    text_data = file_write.read()

data_len = len(text_data)
train_size = 0.9 * data_len

train_data = text_data[:train_size]
val_data = text_data[train_size:]

# tiktoken encoding
encoder = tiktoken.get_encoding("gpt2")
train_token_ids = encoder.encode_ordinary(train_data)
val_token_ids = encoder.encode_ordinary(val_data)

print(
    f"train set => {len(train_token_ids):,} \n val set => {len(val_token_ids):,}")

# export to binary
train_token_ids = np.array(train_token_ids, dtype=np.uint16)
val_token_ids = np.array(val_token_ids, dtype=np.uint16)

train_token_ids.tofile("train_tokens.bin")
val_token_ids.tofile("val_tokens.bin")
