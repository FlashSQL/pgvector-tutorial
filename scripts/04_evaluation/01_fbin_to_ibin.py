#!/usr/bin/env python
"""
train.fbin, query.fbin 두 파일을 사용해서 DiskANN 스타일의 ground truth .ibin(.bin) 파일을 생성하는 스크립트입니다.

- 입력 포맷 (DiskANN/ANN-Benchmarks .fbin/.bin과 동일 가정):
    int32: num_vectors (N)
    int32: dimension (D)
    float32[N * D]: row-major 벡터

- 출력 포맷 (compute_groundtruth.cpp 의 save_groundtruth_as_one_file 와 동일):
    int32: num_queries (Q)
    int32: k
    uint32[Q * k]: 각 쿼리의 top-k 이웃 인덱스
    float32[Q * k]: 각 쿼리의 top-k 거리/유사도
      * L2 / COSINE: L2 제곱거리
      * MIPS: inner product 값 (값이 클수록 가까움)

사용 예시:
    python compute_groundtruth_py.py \
        --base train.fbin \
        --query query.fbin \
        --out gt.ibin \
        --k 100 \
        --metric l2
"""

import argparse
import struct
from typing import Tuple

import numpy as np


# ---------------------------
# 1. .fbin / .bin 로드
# ---------------------------
def load_fbin(path: str) -> Tuple[np.ndarray, int, int]:
    """
    DiskANN/ANN-Benchmarks 스타일 .fbin/.bin 로드
    return: (vectors [N, D], N, D)
    """
    with open(path, "rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError(f"File {path} is too small to contain header")
        n, d = struct.unpack("ii", header)
        vecs = np.fromfile(f, dtype=np.float32, count=n * d)
        if vecs.size != n * d:
            raise ValueError(
                f"File {path} size mismatch: expected {n*d} floats, got {vecs.size}"
            )
    return vecs.reshape(n, d), n, d


# ---------------------------
# 2. ground truth .ibin 저장
# ---------------------------
def save_groundtruth(path: str,
                     indices: np.ndarray,
                     dists: np.ndarray) -> None:
    """
    DiskANN compute_groundtruth.cpp 의 save_groundtruth_as_one_file 과 동일 포맷으로 저장

    header:
        int32: num_queries (Q)
        int32: k
    body:
        uint32[Q, k]: indices
        float32[Q, k]: distances
    """
    if indices.shape != dists.shape:
        raise ValueError("indices and dists must have the same shape")

    q, k = indices.shape
    idx_u32 = indices.astype(np.uint32, copy=False)
    dist_f32 = dists.astype(np.float32, copy=False)

    with open(path, "wb") as f:
        f.write(struct.pack("ii", q, k))
        idx_u32.tofile(f)
        dist_f32.tofile(f)


# ---------------------------
# 3. 거리/유사도 계산
# ---------------------------
def compute_l2_groundtruth(
    base: np.ndarray,
    queries: np.ndarray,
    k: int,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    L2 제곱거리 기반 exact k-NN (compute_groundtruth.cpp 의 L2/cosine 부분과 동일한 수학)
    """
    n_points, dim = base.shape
    n_queries, dim_q = queries.shape
    assert dim == dim_q

    base_norm = np.sum(base ** 2, axis=1)  # [N]
    query_norm = np.sum(queries ** 2, axis=1)  # [Q]

    all_indices = np.empty((n_queries, k), dtype=np.int64)
    all_dists = np.empty((n_queries, k), dtype=np.float32)

    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        q_batch = queries[start:end]  # [B, D]
        q_norm_batch = query_norm[start:end]  # [B]

        # dist^2(x, q) = ||x||^2 + ||q||^2 - 2 x·q
        # matmul 결과: [B, N]
        dot = q_batch @ base.T
        dists = base_norm[None, :] + q_norm_batch[:, None] - 2.0 * dot

        # top-k (가까운 것부터) 선택
        idx_part = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
        dist_part = np.take_along_axis(dists, idx_part, axis=1)

        # 각 row 내부 정렬
        order = np.argsort(dist_part, axis=1)
        idx_sorted = np.take_along_axis(idx_part, order, axis=1)
        dist_sorted = np.take_along_axis(dist_part, order, axis=1)

        all_indices[start:end] = idx_sorted
        all_dists[start:end] = dist_sorted.astype(np.float32)

    return all_indices, all_dists


def compute_mips_groundtruth(
    base: np.ndarray,
    queries: np.ndarray,
    k: int,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inner Product 기반 exact k-NN.
    DiskANN에서는 inner product를 “최대화”하는 방향으로,
    dist = -inner_product 로 정렬하고, 최종 저장 시 +inner_product 를 저장합니다.
    """
    n_points, dim = base.shape
    n_queries, dim_q = queries.shape
    assert dim == dim_q

    all_indices = np.empty((n_queries, k), dtype=np.int64)
    all_scores = np.empty((n_queries, k), dtype=np.float32)

    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        q_batch = queries[start:end]  # [B, D]

        # inner product matrix [B, N]
        ip = q_batch @ base.T

        # 상위 k개의 inner product (큰 값부터)
        # argpartition 은 오름차순 기준이므로 -ip 사용
        idx_part = np.argpartition(-ip, kth=k - 1, axis=1)[:, :k]
        score_part = np.take_along_axis(ip, idx_part, axis=1)

        order = np.argsort(-score_part, axis=1)
        idx_sorted = np.take_along_axis(idx_part, order, axis=1)
        score_sorted = np.take_along_axis(score_part, order, axis=1)

        all_indices[start:end] = idx_sorted
        all_scores[start:end] = score_sorted.astype(np.float32)

    return all_indices, all_scores


def compute_cosine_groundtruth(
    base: np.ndarray,
    queries: np.ndarray,
    k: int,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    cosine 거리 기반 exact k-NN.
    compute_groundtruth.cpp 처럼,
    - 벡터를 L2 normalized
    - L2 제곱거리로 계산 (cosine distance와 단조동치)
    """
    # 0 division 방지
    def _normalize(x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        norm[norm == 0.0] = np.finfo(np.float32).eps
        return x / norm

    base_n = _normalize(base.astype(np.float32, copy=True))
    queries_n = _normalize(queries.astype(np.float32, copy=True))

    # 이후는 L2와 동일
    return compute_l2_groundtruth(base_n, queries_n, k, batch_size=batch_size)


# ---------------------------
# 4. main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute exact ground truth .ibin from train(.fbin) and query(.fbin), "
                    "following DiskANN compute_groundtruth.cpp 로직 (간략 버전)."
    )
    parser.add_argument("--base", required=True, help="train 벡터 파일 경로 (.fbin / .bin)")
    parser.add_argument("--query", required=True, help="query 벡터 파일 경로 (.fbin / .bin)")
    parser.add_argument("--out", required=True, help="ground truth 출력 파일 경로 (.ibin / .bin)")
    parser.add_argument("--k", type=int, required=True, help="nearest neighbors 개수")
    parser.add_argument(
        "--metric",
        type=str,
        default="l2",
        choices=["l2", "mips", "cosine"],
        help="거리 함수 (l2 / mips / cosine)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="쿼리 배치 크기 (메모리 상황에 맞게 조정)",
    )

    args = parser.parse_args()

    print(f"[INFO] Loading base vectors from {args.base}")
    base, n_base, dim_base = load_fbin(args.base)
    print(f"[INFO] Base: {n_base} vectors, dim={dim_base}")

    print(f"[INFO] Loading query vectors from {args.query}")
    queries, n_query, dim_query = load_fbin(args.query)
    print(f"[INFO] Query: {n_query} vectors, dim={dim_query}")

    if dim_base != dim_query:
        raise ValueError(
            f"Dimension mismatch: base dim={dim_base}, query dim={dim_query}"
        )

    print(f"[INFO] Computing ground truth (metric={args.metric}, k={args.k})")

    if args.metric == "l2":
        indices, dists = compute_l2_groundtruth(
            base, queries, k=args.k, batch_size=args.batch_size
        )
    elif args.metric == "mips":
        indices, dists = compute_mips_groundtruth(
            base, queries, k=args.k, batch_size=args.batch_size
        )
    else:  # cosine
        indices, dists = compute_cosine_groundtruth(
            base, queries, k=args.k, batch_size=args.batch_size
        )

    print(f"[INFO] Saving ground truth to {args.out}")
    save_groundtruth(args.out, indices, dists)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()