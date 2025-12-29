# 05. PostgreSQL 인덱스 모니터링 가이드

이 문서는 벡터 인덱스(IVFFlat, HNSW 등)의 생성 과정과  
테이블/인덱스의 크기, 캐시 히트율을 점검하기 위해 사용할 수 있는  
PostgreSQL 모니터링 SQL을 정리한 레퍼런스입니다.

------------------------------------------------------------

## 1. 인덱스 생성 진행 상황 확인

인덱스가 생성되는 동안 현재 진행률과 처리 블록 수, 처리된 튜플 수 등을 실시간으로 확인할 수 있습니다.

```sql
SELECT
  now()::TIME(0),
  a.query,
  p.phase,
  round(p.blocks_done / p.blocks_total::numeric * 100, 2) AS "% done",
  p.blocks_total,
  p.blocks_done,
  p.tuples_total,
  p.tuples_done,
  ai.schemaname,
  ai.relname,
  ai.indexrelname
FROM pg_stat_progress_create_index p
JOIN pg_stat_activity a ON p.pid = a.pid
LEFT JOIN pg_stat_all_indexes ai 
  ON ai.relid = p.relid 
 AND ai.indexrelid = p.index_relid;
```

이 쿼리는 인덱스를 만들 때 특히 유용하며,  
대규모 벡터 테이블에서 IVFFlat/HNSW 인덱싱 진행률을 확인하는 데 사용합니다.

------------------------------------------------------------

## 2. 테이블 및 인덱스 크기 점검

테이블과 인덱스의 실제 디스크 사용량을 확인할 수 있습니다.

```sql
SELECT 
    relname AS relation,
    pg_size_pretty(pg_total_relation_size(relid)) AS total_size,
    pg_size_pretty(pg_relation_size(relid)) AS relation_size,
    pg_size_pretty(pg_indexes_size(relid)) AS indexes_size
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC;
```

- total_size: 테이블 + 인덱스 전체 크기  
- relation_size: 테이블 본체 크기  
- indexes_size: 인덱스 크기  

벡터 인덱스(list 수, HNSW 파라미터)에 따른 크기 변화를 모니터링할 때 유용합니다.

------------------------------------------------------------

## 3. 인덱스 캐시 히트율 (Index Cache Hit Ratio)

PostgreSQL이 인덱스를 얼마나 메모리에서 처리하는지 확인하기 위한 히트율 지표입니다.

```sql
SELECT
  indexrelname AS index_name,
  idx_blks_hit,
  idx_blks_read,
  ROUND(100.0 * idx_blks_hit / NULLIF(idx_blks_hit + idx_blks_read, 0), 2) AS index_hit_ratio
FROM pg_statio_user_indexes
WHERE idx_blks_hit > 0
ORDER BY index_hit_ratio DESC;
```

- index_hit_ratio (%)가 높을수록 메모리 히트율이 좋아 조회 성능이 향상됩니다.