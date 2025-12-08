# 03. 데이터 적재 및 인덱스 생성

이 문서는 준비된 벡터 데이터셋을 PostgreSQL(pgvector)로 적재하고,  
IVFFlat / HNSW 인덱스를 생성해 벡터 검색을 수행할 수 있는 상태로 만드는 과정을 정리합니다.

------------------------------------------------------------

## 1. 테이블 스키마 생성

벡터 테이블을 생성합니다.
psql 쉘에서 실행

예시 스키마:

```
CREATE TABLE vectors (
  id        integer,
  embedding VECTOR(768)    -- 임베딩 차원에 맞게 지정
);
```

------------------------------------------------------------

### 2. pgvector에 load 하기

미리 만들어둔 pgvector table에 이를 적재하기위해 script를 사용합니다.
`scripts/03_create_index/01_fbin_to_pgvector.py` 참고

------------------------------------------------------------

### 3. index 생성

https://github.com/pgvector/pgvector?tab=readme-ov-file#indexing 참고하여 인덱스를 생성합니다.

예시

```
CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops) WITH (m = 24, ef_construction = 200);

CREATE INDEX ON items USING hnsw ((binary_quantize(embedding)::bit(1536)) bit_hamming_ops) WITH (m = 24, ef_construction = 200);

CREATE INDEX ON items USING ivfflat (embedding vector_cosine_ops) WITH (lists = 1000);
```