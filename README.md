# pgvector-tutorial
PostgreSQL + **pgvector** 기반의 벡터 검색 실험을 한 흐름으로 따라갈 수 있는 실용 가이드입니다. 설치부터 데이터 준비, 인덱스 생성·평가, 그리고 PostgreSQL 튜닝까지 필요한 스크립트와 절차를 정리했습니다.

---

## 무엇을 담고 있나
- PostgreSQL과 pgvector를 직접 빌드하고 확장까지 설치하는 방법
- Hugging Face/ANN-Benchmarks 데이터를 활용해 `.fbin`/`.ibin` 포맷을 만드는 데이터셋 준비법
- 테이블 스키마 예시와 대량 적재 스크립트
- IVFFlat/HNSW 인덱스 생성과 recall–QPS 기반 평가 워크플로우
- 대규모 적재·인덱싱 환경을 위한 PostgreSQL 핵심 튜닝 파라미터

---

## 문서 모음 (`docs/`)
- `docs/00_overview.md`: 프로젝트 개요와 목적
- `docs/01_install_pgvector.md`: PostgreSQL·pgvector 소스 빌드 및 초기화
- `docs/02_create_dataset.md`: Hugging Face/ANN-Benchmarks 데이터 준비와 `.fbin` 생성
- `docs/03_create_index.md`: 테이블 생성, 데이터 적재, IVFFlat/HNSW 인덱스 예제
- `docs/04_evaluation.md`: ground truth 생성, 쿼리 실행, recall/QPS 계산과 시각화
- `docs/05_monitoring_sql.md`: 대용량 적재·인덱싱 시 유용한 PostgreSQL 파라미터 튜닝 가이드  
  *(추가 모니터링/튜닝 팁을 여기에 확장 예정)*

---

## 제공 스크립트 (`scripts/`)
- 데이터셋 생성: `scripts/02_create_dataset/01_huggingface_to_fbin.py`  
  Hugging Face 데이터셋을 `.fbin`(train/query)으로 변환.
- 데이터 적재: `scripts/03_create_index/01_fbin_to_pgvector.py`  
  `.fbin`을 읽어 `COPY` 바이너리로 pgvector 테이블에 삽입.
- 평가 준비: `scripts/04_evaluation/01_fbin_to_ibin.py`  
  train/query `.fbin`으로 DiskANN 스타일 `.ibin` ground truth 생성.
- 쿼리 실행: `scripts/04_evaluation/02_run_test.py`  
  PostgreSQL에 recall@k, QPS 측정 쿼리를 반복 수행.
- 결과 시각화: `scripts/04_evaluation/03_plot_result.py`  
  recall–QPS 곡선 예제 플롯.

---

## 빠른 시작
1) **PostgreSQL & pgvector 설치**  
   `docs/01_install_pgvector.md`의 빌드/초기화 절차를 따라 `CREATE EXTENSION vector;`까지 완료합니다.
2) **데이터 준비**  
   Hugging Face 예시: `python scripts/02_create_dataset/01_huggingface_to_fbin.py`에서 경로/차원 등을 수정 후 실행 → `*-train.fbin`, `*-query.fbin` 생성 (`docs/02_create_dataset.md` 참고).
3) **테이블 생성 & 적재**  
   `CREATE TABLE vectors (id integer, embedding VECTOR(<dim>));` 실행 후 `scripts/03_create_index/01_fbin_to_pgvector.py`의 파일 경로와 테이블명을 맞춰 실행 (`docs/03_create_index.md`).
4) **인덱스 생성**  
   예: `CREATE INDEX ON vectors USING hnsw (embedding vector_cosine_ops) WITH (m=24, ef_construction=200);`  
   필요 시 IVFFlat/bit_hamming 예제를 문서에서 선택.
5) **성능 평가**  
   - Ground truth: `python scripts/04_evaluation/01_fbin_to_ibin.py --base train.fbin --query query.fbin --out neighbors.ibin --k 100 --metric l2`  
   - 쿼리/recall 측정: `scripts/04_evaluation/02_run_test.py`에서 파일 경로, 테이블명, 포트 등을 설정 후 실행  
   - 그래프: `scripts/04_evaluation/03_plot_result.py`의 데이터 포인트를 수정해 플롯
6) **튜닝**  
   `docs/05_monitoring_sql.md`의 `shared_buffers`, `maintenance_work_mem`, 병렬 파라미터 등을 환경에 맞게 적용.

---

## 요구 사항
- Python 3.10+ 와 주요 패키지: `numpy`, `datasets`, `scikit-learn`, `psycopg`, `pgvector`, `matplotlib`
- PostgreSQL 소스 빌드 환경, pgvector 소스
- 실험용 DB: 예시에서는 `dbname=ann`, `user=ann`, `port=5434`를 사용

---

## 기여
이슈/제안/PR 환영합니다. 실험 중 얻은 설정 팁이나 추가 스크립트가 있다면 공유해주세요.
