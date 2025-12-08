import os
import struct
import subprocess
import time

import numpy as np
import pgvector.psycopg
import psycopg

# =========================
# User settings
# =========================

test_fbin = "/path/to/test.fbin"
neighbors_ibin = "/path/to/neighbors.ibin"
table = "table_name"
search_params = [10, 20, 30, 40]

PG_CTL = "/path/to/bin/pg_ctl"
PGDB = "/path/to/pgdb/"
PORT = 5434
RESTART = False


def start():
    try:
        subprocess.run([PG_CTL, "-D", PGDB, "start"], check=True)
    except subprocess.CalledProcessError as e:
        print("Failed to start PostgreSQL:", e)


def stop():
    try:
        subprocess.run([PG_CTL, "-D", PGDB, "stop"], check=True)
    except subprocess.CalledProcessError as e:
        print("Failed to stop PostgreSQL:", e)


conn = None
cur = None


# =========================
# FBIN / IBIN loaders
# =========================
def read_fbin(path):
    """
    Load FBIN as memmap with header auto-detection.
    Supports:
      [int32 N][int32 dim] + float32 data
      [int32 dim] + float32 data   (N inferred)
    Returns: (memmap ndarray float32 of shape (N, dim), N, dim)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    filesize = os.path.getsize(path)

    with open(path, "rb") as f:
        header8 = f.read(8)
        if len(header8) < 4:
            raise ValueError("Invalid FBIN: header too small")

        # Try [N][dim] first
        if len(header8) == 8:
            N, dim = struct.unpack("ii", header8)
            expected = 8 + N * dim * 4
            if expected == filesize and N > 0 and dim > 0:
                data = np.memmap(path, dtype=np.float32, mode="r", offset=8, shape=(N, dim))
                return data, N, dim

        # Fallback: [dim] only
        dim = struct.unpack("i", header8[:4])[0]
        if dim <= 0:
            raise ValueError(f"Invalid dim={dim} in FBIN header")
        # remaining bytes after 4-byte dim
        remain = filesize - 4
        if remain % (dim * 4) != 0:
            raise ValueError(f"FBIN size mismatch for dim={dim}: {filesize} bytes")
        N = remain // (dim * 4)
        data = np.memmap(path, dtype=np.float32, mode="r", offset=4, shape=(N, dim))
        return data, N, dim


def read_ibin(path):
    with open(path, "rb") as f:
        N, K = struct.unpack("ii", f.read(8))
    ids = np.memmap(path, dtype="<u4", mode="r", offset=8, shape=(N, K))
    dists = np.memmap(path, dtype="<f4", mode="r", offset=8 + N * K * 4, shape=(N, K))
    return ids.astype(np.int32), dists, N, K


# =========================
# DB setup/teardown
# =========================
def setup():
    global conn, cur
    conn = psycopg.connect(
        host="localhost", user="ann", password="ann", dbname="ann", autocommit=True, port=PORT
    )
    pgvector.psycopg.register_vector(conn)
    cur = conn.cursor()


def teardown():
    if cur:
        cur.close()
    if conn:
        conn.close()


# =========================
# Metrics
# =========================
def recall_at_k(run_result, ground_truth, k=10):
    # run_result: list[list[int]]  (retrieved ids)
    # ground_truth: np.ndarray (N x Kgt)
    recalls = []
    for i in range(len(run_result)):
        retrieved_top_k = set(run_result[i][:k])
        gt_top_k = set(ground_truth[i][:k])
        recalls.append(len(retrieved_top_k & gt_top_k) / k)
    return float(np.mean(recalls)) if recalls else 0.0


def pad_to_k(lst, k, pad_val=-1):
    return lst[:k] + [pad_val] * (k - len(lst))


def print_hit_ratio():
    cur.execute("""
        SELECT
            relname AS index_name,
            idx_blks_hit,
            idx_blks_read,
            ROUND(100.0 * idx_blks_hit / NULLIF(idx_blks_hit + idx_blks_read, 0), 2) AS index_hit_ratio
        FROM pg_statio_user_indexes
        ORDER BY index_hit_ratio DESC;
    """)
    rows = cur.fetchall()
    print(" Index Hit Ratios:")
    for r in rows:
        if r[1] > 0:
            print(f"    Index: {r[0]}, Hits: {r[1]}, Reads: {r[2]}, Hit Ratio: {r[3]}%, Reads: {r[1] + r[2]}")


# =========================
# Query routine
# =========================
def query_to_pg_recall(test_vectors, neighbors, search_param=-1, limit_k=10):
    Nq, dim = test_vectors.shape
    print(f"Querying data... {Nq} queries, dim={dim}")

    query = "SELECT id FROM " + table + " ORDER BY embedding <=> %s LIMIT %s"

    cur.execute("SET ivfflat.probes = %d" % search_param)
    cur.execute("SET hnsw.ef_search = %d" % search_param)
    cur.execute("SET diskann.query_search_list_size = %d" % search_param)
    cur.execute("SET enable_seqscan = OFF")
    cur.execute("SELECT pg_stat_reset();")
    print(query)

    retrieved_ids = []

    count = 0
    start_time = time.time()

    # Iterate queries
    for i in range(Nq):
        v = np.asarray(test_vectors[i], dtype=np.float32)
        cur.execute(query, (v, limit_k), binary=True, prepare=True)
        retrieved_ids.append([row[0] for row in cur.fetchall()])

        count += 1

        total_time = time.time() - start_time
        qps = count / total_time if total_time > 0 else 0.0
        print(f"Querying done. Total time: {total_time:.6f} seconds, Count: {count}")
        print(f"  QPS: {qps:.2f}")

    if neighbors is not None:
        for k in (10, 1):
            rr = np.array([pad_to_k(r, k, pad_val=-1) for r in retrieved_ids], dtype=np.int32)
            avg_recall = recall_at_k(rr, neighbors, k=k)
            print(f"  Recall@{k}: {avg_recall:.4f}")

    print_hit_ratio()
    print()


# =========================
# Main
# =========================
def run():
    # Load test vectors
    test_mm, N_test, dim = read_fbin(test_fbin)
    print(f"[TEST] N={N_test}, dim={dim}, dtype=float32, path={test_fbin}")

    # Load neighbors (IBIN)
    ids, dists, N_gt, K_gt = read_ibin(neighbors_ibin)
    neighbors_mm = ids

    if N_gt != N_test:
        raise ValueError(f"neighbors N mismatch: {N_gt} (gt) vs {N_test} (test)")
    print(f"[GT] neighbors: N={N_gt}, K={K_gt}, dtype=int32, path={neighbors_ibin}")

    # For print/peek
    print(f"Sample vec[0][:5]: {np.array(test_mm[0][:5])}")

    if RESTART:
        stop()
        start()

    try:
        for sp in search_params:
            # Setup DB
            setup()
            print(f"Running with search_param: {sp}")
            query_to_pg_recall(test_mm, neighbors_mm, search_param=sp, limit_k=10)
    finally:
        teardown()
        if RESTART:
            stop()


if __name__ == "__main__":
    run()
