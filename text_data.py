import os, re
from tqdm import tqdm
import tiktoken
import numpy as np
from datasets import load_dataset

num_proc = 8


encoder = tiktoken.get_encoding('gpt2')

open_webtext = load_dataset('openwebtext', num_proc=num_proc)

split_dataset = open_webtext['train'].train_test_split(test_size=0.0005,seed=1007, shuffle=True) # type:ignore
split_dataset['val'] = split_dataset.pop('test')

def token_process(text): # for gpt2 byte_pair encoding
    ids = encoder.encode_ordinary(example['text'])
    ids.append(encoder.eot_token)
    
    out = {'ids': ids, 'len': len(ids)}
    return out

tokenized = split_dataset.map(
    token_process,
    remove_columns=['text'],
    desc='tokenizing the splits',
    num_proc=num_proc
)

for split, dset in tokenized.items():
    array_len = np.sum(dset['len', dtype=np.uint16])
    file_name = f'{split}.bin'
    
    dtype = np.uint16
    
    arr = np.memmap(file_name, dtype=dtype, mode='w+', shape=array_len)
    total_batches =1024
    
    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'writing to {file_name}'):
        # batch fro efficinent write
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])
        
        # write into memmap
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    
    arr.flush()