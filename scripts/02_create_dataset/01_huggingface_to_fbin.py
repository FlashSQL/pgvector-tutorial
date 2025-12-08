import struct

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split

test_size = 1000
ds_name = "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M"
split = "train"
column = "text-embedding-3-large-3072-embedding"
dim = 3072
out_fn_prefix = "/file/to/generated/prefix"


def run():
    create_fbin_from_huggingface(ds_name, split, column, dim, out_fn_prefix)


def create_fbin_from_huggingface(ds_name, split, column, dim, out_fn_prefix, n=None):
    data = load_dataset(ds_name, split=split)

    if n is not None and n >= 100_000:
        data = data.select(range(n))

    all_indices = np.arange(len(data))
    train_idx, test_idx = train_test_split(all_indices, test_size, random_state=1)

    embeddings = data.to_pandas()[column].to_numpy()
    embeddings = np.vstack(embeddings).reshape((-1, dim))

    print(test_idx)

    save_fbin(out_fn_prefix + "-train.fbin", embeddings[train_idx])
    save_fbin(out_fn_prefix + "-query.fbin", embeddings[test_idx])


def save_fbin(file_path, data):
    N, D = data.shape
    with open(file_path, 'wb') as f:
        f.write(struct.pack('I', N))
        f.write(struct.pack('I', D))
        f.write(data.astype(np.float32).tobytes())


if __name__ == "__main__":
    run()
