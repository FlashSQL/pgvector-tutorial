import os
import struct

import numpy as np
import pgvector.psycopg
import psycopg

conn = None
cur = None

port = 5434
file_path = "/path/to/train.fbin/file"
table = "table_name"


def setup():
    global conn, cur
    conn = psycopg.connect(
        host="localhost", user="ann", password="ann",
        dbname="ann", autocommit=True, port=port
    )
    pgvector.psycopg.register_vector(conn)
    cur = conn.cursor()


def teardown():
    cur.close()
    conn.close()


def read_fbin_vectors(path):
    with open(path, 'rb') as f:
        num_vectors = struct.unpack('I', f.read(4))[0]
        vector_dim = struct.unpack('I', f.read(4))[0]
        print(f"Loading {num_vectors} vectors of dimension {vector_dim} from {path}")

        expected_total = num_vectors * vector_dim
        vecs = np.fromfile(f, dtype=np.float32, count=expected_total)
        vecs = vecs.reshape((num_vectors, vector_dim))
        print(f"Loading vectors done for {path}.")
    return vecs


def load_data_to_pg(table, vectors, offset):
    print(f"Inserting {len(vectors)} vectors starting from ID {offset}")

    with cur.copy(f"COPY {table} (id, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
        copy.set_types(["int4", "vector"])
        for i, row in enumerate(vectors):
            copy.write_row((offset + i, row))
    print("Insertion done.")


def main():
    setup()
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")

    else:
        vectors = read_fbin_vectors(file_path)
        load_data_to_pg(table, vectors, 0)

    teardown()
    print("✅ All files loaded successfully.")


if __name__ == "__main__":
    main()
