# 06. PostgreSQL 튜닝 가이드 

이 문서는 pgvector 벡터 인덱스 생성(IVFFlat, HNSW) 및  
대규모 데이터 적재/평가 환경에서 PostgreSQL 성능을 최적화하기 위한  
핵심 파라미터 설정 예시를 정리합니다.

아래 설정은 postgresql.conf에서 할 수 있습니다.

------------------------------------------------------------

## 1. 메모리 관련 설정

### shared_buffers
PostgreSQL이 버퍼 풀로 사용하는 메모리 영역입니다.
벡터 인덱스 조회 시 디스크 읽기를 줄이는 데 중요합니다.

postgres.conf:
```
shared_buffers = 10GB
```
---

### maintenance_work_mem
인덱스 생성에 사용되는 메모리입니다.
IVFFlat/HNSW 인덱스 생성 속도에 직접적인 영향을 줍니다.

postgres.conf:
```
maintenance_work_mem = 10GB
```
가능한 한 크게 설정하면 인덱스 생성 속도가 향상됩니다.

------------------------------------------------------------

## 2. 병렬 처리 관련 설정

대규모 인덱스 생성과 쿼리 병렬화를 위해 다음 세 파라미터를 함께 조정합니다.

postgres.conf:
```
max_worker_processes = 50
max_parallel_workers = 50
max_parallel_maintenance_workers = 50
```

세 값을 함께 크게 잡으면 인덱스 생성과 병렬 쿼리에 더 많은 CPU 코어를 활용할 수 있지만,  
동시에 여러 작업이 돌아갈 경우 시스템 전체 부하도 크게 증가할 수 있으므로  
실제 서버 사양(코어 수, 메모리 크기)에 맞춰 조정하는 것이 좋습니다.

`iostat` 등의 명령어로 monitoring 하며 조절

------------------------------------------------------------

## 3. 포트 변경 및 서버 실행 방법

PostgreSQL을 기본 포트(5432)와 격리된 환경에서 사용하기 위해 아래처럼 포트를 변경할 수 있습니다.

postgres.conf:
```
port = 5434
```

포트를 변경한 뒤에는 서버를 다음과 같이 실행합니다.

terminal:
```
psql -p 5434 -d ann
```

---

## 4. 설정 적용 가이드

1) postgresql.conf에 위 설정을 추가 또는 수정  
2) PostgreSQL 재시작 

예:

terminal:
```
pg_ctl -D /path/to/pgdb restart
```

3) 설정 확인:

psql:
```
SHOW shared_buffers;
SHOW maintenance_work_mem;
SHOW max_worker_processes;
```
------------------------------------------------------------

## 5. Direct I/O 설정

OS 페이지 캐시 영향을 제거하고 벤치마크를 수행하려면 Direct I/O를 활성화할 수 있습니다.

postgresql.conf:
```
debug_io_direct='data' 
```
설정 변경 후 서버 재시작이 필요합니다.

Direct I/O 활성화 여부 확인:

psql:
```
SHOW debug_io_direct;
```