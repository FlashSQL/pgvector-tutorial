# 02. 데이터셋 생성하기

이 문서는 pgvector 벤치마크를 위한 데이터셋을 준비하는 두 가지 방법을 정리합니다.

1) 직접 임베딩을 생성하여 데이터셋을 만드는 방법
2) ann-benchmarks에서 제공하는 데이터셋을 그대로 활용하는 방법

------------------------------------------------------------

## 1. 임베딩 생성하기

huggingface에서 제공하는 데이터를 활용하거나, 원하는 데이터셋을 직접 구성할 수 있습니다.

### HuggingFace에서 데이터 확보

Hugging Face Hub의 공개 데이터셋을 가져와 필요한 텍스트나 이미지 데이터를 준비합니다.
huggingface.co/datasets 에서 원하는 embedding model 등을 검색하여 사용

### 임베딩 생성 및 파일 변환

`scripts/02_create_dataset/01_huggingface_to_fbin.py` 참고

- 벡터 데이터 → train.fbin 파일
- 쿼리 데이터 → query.fbin 파일

------------------------------------------------------------

## 2. ann-benchmark 데이터셋 활용하기

ann-benchmarks(https://github.com/erikbern/ann-benchmarks/)에서는  
다양한 벡터 검색용 데이터셋을 제공하며, 이를 통해 `.hdf5` 형식의 파일을 쉽게 얻을 수 있습니다.

### 데이터셋 다운로드

아래 파일에서 데이터셋별 다운로드 URL과 파일 변환 방식을 확인할 수 있습니다.

ann_benchmarks/datasets.py  
https://github.com/erikbern/ann-benchmarks/blob/main/ann_benchmarks/datasets.py

이 파일에는 다음과 같은 내용이 포함됩니다.

- 데이터셋 다운로드 URL 목록
- 다운로드한 원본 파일을 `.hdf5` 형식으로 변환하는 `write_output()` 함수

이 함수는 그대로 사용하거나 약간 수정해서 원하는 데이터셋을 쉽게 준비하는 데 활용할 수 있습니다.

### dataset 로드 및 recall 계산 코드

ann-benchmarks에서는 실행 스크립트에 dataset 로드, query 수행, recall 계산 등이 포함되어 있습니다.

참고 파일:  
https://github.com/erikbern/ann-benchmarks/blob/main/ann_benchmarks/runner.py

이 로직을 참고하여 다음 기능을 구현할 수 있습니다.

- dataset 불러오기
- query vector 실행
- 검색 결과 비교
- recall@k 계산