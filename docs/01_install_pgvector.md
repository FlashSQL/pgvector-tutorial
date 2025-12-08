# 01. pgvector 설치하기

이 문서는 PostgreSQL을 직접 빌드하고 그 위에 pgvector 확장을 설치하는 과정을 단계별로 정리한 가이드입니다.

------------------------------------------------------------

## 1. PostgreSQL과 pgvector 클론하기

공식 저장소:
- https://github.com/postgres/postgres
- https://github.com/pgvector/pgvector

작업 디렉토리 구조 예시:

```
postgres-pgvector
│─ postgres
└─ pgvector
```
------------------------------------------------------------

## 2. PostgreSQL 빌드 (사용자 지정 경로)

설치 경로 지정:
```
PG_OUT=/path/to/pg_db
```
configure 옵션 예시:
```
# 기본 빌드
./configure --prefix=$PG_OUT --without-icu

# 디버그 빌드
./configure --prefix=$PG_OUT --enable-debug --enable-cassert --without-icu CFLAGS='-ggdb -O0'

# 페이지 크기 변경
./configure --prefix=$PG_OUT --without-icu --with-blocksize=32
```
컴파일 및 설치:
```
make -j
make install
make clean   # build 내용 삭제에 사용
```
------------------------------------------------------------

## 3. pgvector 빌드
```
cd pgvector
make PG_CONFIG=$PG_OUT/bin/pg_config -j
make install PG_CONFIG=$PG_OUT/bin/pg_config
make PG_CONFIG=$PG_OUT/bin/pg_config clean   # 선택
```
------------------------------------------------------------

## 4. PostgreSQL 초기화 및 실행
```
cd $PG_OUT/bin
./initdb -D $PG_OUT/pgdb
./pg_ctl -D $PG_OUT/pgdb start
./pg_ctl -D $PG_OUT/pgdb stop   # 종료
```
------------------------------------------------------------

## 5. 데이터베이스 및 사용자 설정
```
cd $PG_OUT/bin
./createdb ann
./psql ann
```
PostgreSQL 셸에서 실행:
```
CREATE USER ann WITH ENCRYPTED PASSWORD 'ann';
GRANT ALL PRIVILEGES ON DATABASE ann TO ann;
GRANT [USER] TO ann;  -- 실제 사용자명으로 변경
CREATE EXTENSION vector;
```
------------------------------------------------------------

설치 완료.