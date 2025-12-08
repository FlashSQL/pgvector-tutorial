# 04. 성능 평가

이 문서는 pgvector 인덱스 성능을 평가하는 기본 개념과 워크플로우를 정리합니다.

------------------------------------------------------------

## 1. Evaluation의 기본 개념

벡터 검색 인덱스를 평가할 때 가장 많이 사용하는 지표 중 하나가 **recall@k**입니다.

- true nearest neighbor:
    - 전체 공간에서 정확한 k개의 최근접 이웃
    - brute-force를 이용해 미리 계산
- 인덱스 결과:
    - IVFFlat, HNSW 등으로 실제 서비스에서 사용하게 될 근사 최근접 이웃(ANN) 결과

recall@k는 다음과 같이 정의할 수 있습니다.

- 어떤 쿼리 q에 대해
    - T(q) = true 최근접 이웃 집합 (크기 k)
    - A(q) = 인덱스를 통해 찾은 결과 집합 (크기 k)
- 이때 recall@k(q) = |T(q) ∩ A(q)| / k

여러 쿼리에 대해 평균을 내어 전체 recall@k를 계산합니다.

- 전체 recall@k = 모든 쿼리 q에 대한 recall@k(q)의 평균

실제 벤치마크에서는 보통 **recall@k vs QPS** 형태로 그래프를 그리고,  
튜닝(파라미터 조정)에 따라 곡선이 어떻게 바뀌는지 비교합니다.

------------------------------------------------------------

## 2. fbin 파일로부터 ibin 생성 (true nearest neighbor 사전 계산)

```scripts/04_create_dataset/01_fbin_to_ibin.py``` 참고
```https://github.com/microsoft/DiskANN/blob/main/apps/utils/compute_groundtruth.cpp 코드를 사용하는 것이 훨씬 빠름```

생성한 벡터셋에 대해 정확한 nearest neighbor를 얻기 위해 다음 과정을 수행합니다.

1. 전체 벡터 간 거리 계산
2. 각 쿼리 벡터별 k개의 정확한 이웃 선정
3. 이를 .ibin 포맷으로 저장

------------------------------------------------------------

## 3. query.fbin 기반 쿼리 수행 및 결과 수집

다음 단계는 **실제 pgvector 인덱스를 사용하여 쿼리를 실행하고 결과를 모으는 것**입니다.

기본 흐름:

1) query.fbin을 로드하여 쿼리 벡터 목록을 메모리에 올립니다.
2) 각 쿼리 벡터에 대해 PostgreSQL에 질의합니다.
    - 예: embedding <-> query_vector 형태로 ORDER BY + LIMIT k
3) 응답으로 받은 상위 k개 id(또는 row index)를 배열 형태로 저장합니다.
4) 모든 쿼리에 대해 이 결과를 묶어서 stdout으로 결과를 출력합니다.
5) 동시에 전체 QPS 등의 통계도 함께 출력합니다.

프로젝트에서는 이 과정을 자동화한 **쿼리 실행 스크립트**를 제공합니다.

```scripts/04_create_dataset/02_run_test.py```

------------------------------------------------------------

## 4. Plot 그리기 (recall–QPS 곡선)

마지막으로, 평가 결과를 시각화하여 비교하기 쉽게 만드는 단계입니다.

일반적으로 다음과 같은 그래프를 그립니다.

- x축: recall@k
- y축: QPS
- 각 곡선: 서로 다른 설정/인덱스(예: 다른 lists, probes, ef_search 값, IVFFlat vs HNSW 등)

```scripts/04_create_dataset/03_plot_result.py```

------------------------------------------------------------