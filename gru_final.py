#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

모델
  - t시까지의 정보로 대여소마다 t+1·t+2·t+3시 순증감(반납 − 대여)을 예측
  - 과거 입력 창 24시간 × 23개 → GRU (1층, 은닉 64, 시작 상태 0) → 과거 요약 h (64)
  - FiLM (피처 섞기): 대여소 정보 17 + 미래 칸 69 → Linear 86→64, ReLU, Linear 64→128 → γ, β (각 64)
                      → h' = h·(1+γ) + β. 마지막 층을 0으로 시작해 처음엔 h 그대로에서 출발
  - 이어 붙임 [h' 64, 대여소 정보 17, 미래 칸 69] = 150 → 결합층 Linear 150→128, ReLU, Dropout 0.2
  - 출력 3개: ① 순증감 회귀 3 ② 방향 분류 3 × 3 (경계 ±1) ③ 순증감 분포 3 × 17칸 (−8~+8대, 끝 칸은 그 너머 포함)
  - 손실 = 순증감 MSE + 0.5 × 방향 교차 엔트로피 + 0.5 × 분포 교차 엔트로피
  - --mix none 이면 FiLM 없이 이어 붙이기만 하는 구조 (비교 기준 v14_file_ex 와 같음)

입력
  - 과거 창 (시간마다 23): 대여, 반납, 재고, 재고 결측 표시, 경과 시간, 이웃 대여·반납 (1.5km 안 8곳 거리 가중),
    그 시각 평소 대여·반납 (4주 평균), 시간·요일·월 sin·cos, 주말, 공휴일, 기상 6 (기온·습도·풍속·일조·강수·적설)
  - 대여소 정보 (17): 위도, 경도, 거치대수, QR, LCD + 자치구 임베딩 4 + 대여소 ID 임베딩 8 (새 대여소는 '모름' 칸)
  - 미래 칸 (t+1~t+3 시각마다 23): 달력 8 + 기상 6 (관측값) + 평소 패턴 4 + 대여소 속성표 5
  - 이용 통계 4개(총운동량·정기권비율·평균이용시간·평균이동거리)는 쓰지 않음

데이터
  - train_data*.csv (2022~2023), eval_data*.csv (2024), test_data*.csv (2025)
    찾는 순서: 폴더/model_input → 폴더 (--data-dir 로 직접 지정 가능)
  - 처음 한 번 캐시 gru_data_all.npz, gru_stock_file_all.npz 를 결과 폴더에 만듦 (예전 실험 코드와 같은 형식)
  - 전처리 파일에는 이용 0건 시간 행이 없음 → 대여소 × 시간 격자를 새로 만들고 정답은 격자에서 정확히 1·2·3시간 뒤를 읽음
  - 재고 결측: 데이터 담당 보간값(현재재고) + 결측 표시·경과 시간 입력. 이용 0건 시간 재고 = 직전 재고 + 직전 시간 순증감
  - 표준화·평소 패턴·속성표·기준선은 학습 기간(2022~2023) 자료로만 계산. 2023-09-07 7~10시(서비스 장애)는 학습에서 뺌

실행 (최종 설정이 기본값이라 옵션을 따로 적지 않음. run_final_v15_film.bat 이 아래를 차례로 실행)
  python gru_final.py . --tag v15_film_final --stock-eval          학습(2022~2023) + 검증(2024) 평가
  python gru_final.py . --tag v15_film_final --eval-only --stock-eval   저장된 모델로 검증만 다시 평가
  python gru_final.py . --tag v15_film_final --test --stock-eval   테스트(2025) 평가. 맨 마지막에 한 번만
  python gru_final.py . --tag v14_new --mix none                   비교 기준 (FiLM 없이 이어 붙이기만)
  오류 확인용 짧은 실행: --epochs 1 --epoch-samples 20000 --eval-frac 0.05 --mask-eval ""

결과 (결과 폴더, <태그>마다)
  gru_best_<태그>.pt (가중치 + 설정), gru_run_result_<태그>.txt (평가만 하면 _eval, 테스트는 _test),
  gru_<val|test>_summary / by_hour / by_station / alert / stock / confusion / error_dist _<태그>.csv,
  gru_<val|test>_confusion_<태그>.png, gru_mix_<태그>.csv (FiLM 이 한 일), gru_experiments.csv (실행마다 한 줄)

필요한 패키지: numpy, pandas, torch, matplotlib
"""
import argparse
import glob
import os
import re
import sys
import time

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ 설정
CHUNK = 1_000_000          # CSV를 한 번에 읽는 행 수
L = 24                     # 입력 길이 (시간). 저장된 모델을 불러오면 그 설정 값으로 바뀜
HORIZON = 3                # t+1 ~ t+3
WEEK = 168
SPLITS = {'train': (2022, 2023), 'val': (2024, 2024), 'test': (2025, 2025)}
TRAIN_END0 = 2023          # 학습 기간 끝 연도
SYS_RATIO, SYS_MIN_ROWS = 0.5, 10   # 그 시간 이용 행의 절반 넘게가 보간이면 전 대여소 수집 장애 시간
SHORT_GAP = 24             # 직전 실측 후 이 시간까지는 '직전 실측 + 누적 순증감', 넘으면 평균
MEAN_DAYS = 28             # 최근 4주
PROFILE_WEEKS = 4          # 평소 패턴: 최근 4주 같은 요일·시간대
NB_K = 8                   # 이웃 대여소 수
NB_MAX_KM, NB_SIGMA_KM = 1.5, 0.5   # 이웃으로 보는 최대 거리, 거리 가중치 exp(−d²/σ²)의 σ
EMB, FC, DROPOUT = 4, 128, 0.2
SID_DROP = 0.1             # 학습 때 이 비율로 대여소 번호를 '모름'으로 바꿔 새 대여소용 벡터도 학습
FILM_HID = 64              # FiLM 신경망의 가운데 층 크기 (대여소·미래 86 → 64 → γ·β 128)
MIX_NAMES = {'film': 'FiLM(대여소·미래 칸 → 과거 요약 조절)'}
DIST_K = 8                 # 순증감 분포 칸 −8 ~ +8 (끝 칸은 그 밖까지 포함)
NBIN = 2 * DIST_K + 1
ALERT_TAUS = (0.4, 0.5, 0.6, 0.7, 0.8)   # 경보 확률 기준
CONF_BINS = 100            # 확신도(가장 큰 확률) 구간 수
COVERAGES = (1.0, 0.8, 0.6, 0.4, 0.2)    # 확신이 큰 예측 상위 몇 % 의 정확도
STOCK_TAUS = (0.3, 0.5, 0.7)             # --stock-eval 확률 경보 기준
STOCK_SHORT, STOCK_OVER = 0.2, 1.3       # --stock-eval 상태: 부족 = 재고 ≤ 거치대 × 0.2, 과잉 = 재고 ≥ 거치대 × 1.3 (서울시 과다거치 기준)
LR, BATCH = 1e-3, 1024
PATIENCE = 5
EPOCH_SAMPLES = 2_000_000  # 에폭마다 무작위로 뽑는 학습 샘플 수
VAL_FRAC = 0.05            # 조기 종료와 재고 장애 가정 평가에 쓰는 검증 샘플 비율 (고정)
PEAK_HOURS = (7, 8, 17, 18, 19)
EMB_STATS = ['평균순증감', '순증감표준편차', '평균이용량', '이용량표준편차']   # 대여소 속성표에서 쓰는 값
EMB_LOG = {'순증감표준편차', '평균이용량', '이용량표준편차'}                   # 0 이상이고 치우친 값은 log1p 후 표준화
EMB_DAYTYPES = ['평일', '토', '일·공휴일']
CLS_NAMES = ['부족', '균형', '과잉']           # 순증감 < 0, = 0, > 0 (예측은 반올림해서 나눔)
ERR_BINS = (0.5, 1, 2, 3, 5)                  # 오차 분포: |예측 − 실제| 가 이 값 이하인 비율
N_DYN = 5                  # 입력 창의 대여소 자신 값: 대여, 반납, 재고, 결측 표시, 경과 시간
STOCK_COL = 2              # 대여소 자신 값에서 재고 위치 (3 = 결측 표시, 4 = 경과 시간)
FLAG_COL = 3
SEED = 42                  # 평가용 샘플을 고르는 시드 (실험마다 같게 고정). 학습 시드는 --seed
EVAL_SEED = SEED

COLS = {
    'station': ['대여소번호'], 'date': ['날짜'], 'hour': ['시간대'],
    'rent': ['대여건수'], 'ret': ['반납건수'], 'stock': ['현재재고'], 'flag': ['재고_보간여부'],
    'exercise': ['총운동량'], 'pass': ['정기권비율'], 'dur': ['평균이용시간_분'],
    'dist': ['평균이동거리_M', '평균이용거리_M'],
    'lat': ['위도'], 'lon': ['경도'], 'docks': ['총거치대수'], 'gu': ['자치구'],
    'temp': ['기온'], 'humid': ['습도'], 'wind': ['풍속'], 'sun': ['일조시간'], 'rain': ['강수량'], 'snow': ['적설'],
    'holiday': ['공휴일여부'], 'op': ['운영방식'], 'fillm': ['재고_채움방식'],
}
OPTIONAL_COLS = {'op', 'fillm'}                                   # 없어도 되는 열 (재고_채움방식은 입력에 안 넣고 집계만)
# 데이터 파일의 재고_채움방식 번호 (0 = 실측). --stock-source file 결과에 방식별 행 수를 보여 줄 때 씀
FILL_NAMES = {0: '실측', 1: '직전 28일 같은 날종류·시간대 평균', 2: '같은 시간대 평균(평일·주말 합침)',
              3: '앞뒤 1시간 평균', 4: '28일 전체 평균', 5: '공휴일: 다른 해 같은 공휴일 실측',
              6: '공휴일: 28일 주말 평균', 7: '신설 대여소: 대여소 전체 평균', 9: '방식 정보 없음'}
# 미래 값이 섞이는 방식 (stock_fill.py 확인, 2026-10-27). 3번(앞뒤 1시간)은 과거 28일 안의 앞뒤 시간대 평균이라 해당 없음
FILL_FUTURE = {5: '2022년 공휴일 행만 2023년 값 (학습 기간 안)', 7: '그 대여소 2022~2025 전체 평균 (검증·테스트 기간 값 포함)'}
FILL_FUTURE_EVAL = {7}     # 검증·테스트 입력에 미래 값이 들어가는 방식
STATS = ['exercise', 'pass', 'dur', 'dist']    # 이용 통계: 입력에는 안 씀. 캐시 형식을 예전 코드와 맞추려고 읽어 저장만 함
WEATHER = ['temp', 'humid', 'wind', 'sun', 'rain', 'snow']      # rain, snow 는 log1p

# 최종 설정 (v15_film_final). 저장되는 모델 설정(cfg)과 같은 키 — 시나리오·JSON 내보내기 코드가 이 키를 읽음
FINAL = {'window': 24, 'cell': 'gru', 'profile': True, 'neighbors': True, 'nb_k': NB_K, 'hidden': 64, 'layers': 1,
         'net_weight': 1.0, 'lr_decay': True, 'epochs': 30, 'station_emb': True, 'emb_source': 'self', 'emb_no_use': True,
         'station_id_emb': 8, 'cls_head': True, 'cls_weight': 0.5, 'cls_margin': 1, 'dist_head': True, 'dist_weight': 0.5,
         'past_profile': True, 'net_only': True, 'no_stats': True, 'stock_source': 'file', 'file_zero_rule': 'net',
         'exclude_hours': '2023-09-07:7-10', 'train_end': TRAIN_END0, 'mix': 'film'}
# 예전 실험에서 쓰던 기능의 '끔' 값 (이 코드에는 없음. 예전 코드가 새로 학습한 모델을 읽을 수 있게 같이 저장)
OFF = {'od_csv': '', 'attention': False, 'huber': False, 'augment': False, 'drop_stats': [], 'file_fill_skip': [],
       'full_feats': False, 'mark_400': False, 'snow_feats': False, 'rain_aug': 0.0, 'rain_aug_miss': 0.3,
       'rain_aug_shift': 0.2}

LINES = []


def out(s=''):
    print(s, flush=True)
    LINES.append(s)


def set_train_end(year):
    """학습 기간 끝 연도를 정함 (시나리오 코드가 Data 를 만들기 전에 부름). 이 코드의 모델은 2023."""
    year = int(year or TRAIN_END0)
    SPLITS['train'] = (SPLITS['train'][0], year)
    return year


def parse_exclude(spec, t0, H):
    """'2023-09-07:7-10,2023-09-08:18' → (H,) bool. 시각 범위는 양 끝 포함, 시각을 빼면 그날 전체."""
    m = np.zeros(H, bool)
    for part in [p.strip() for p in str(spec or '').split(',') if p.strip()]:
        day, _, hrs = part.partition(':')
        d = int((pd.Timestamp(day.strip()) - t0) / pd.Timedelta('1D'))
        if hrs.strip():
            a, _, b = hrs.partition('-')
            h0, h1 = int(a), int(b or a)
        else:
            h0, h1 = 0, 23
        if not (0 <= h0 <= h1 <= 23):
            raise SystemExit(f'--exclude-hours 시각 범위가 이상함: {part}')
        lo, hi = d * 24 + h0, d * 24 + h1 + 1
        if hi <= 0 or lo >= H:
            raise SystemExit(f'--exclude-hours 날짜가 데이터 기간 밖: {part}')
        m[max(lo, 0):min(hi, H)] = True
    return m


# ------------------------------------------------------------------ CSV 읽기 도우미
def sniff(path):
    with open(path, 'rb') as f:
        head = f.readline()
    for enc in ('utf-8-sig', 'cp949'):
        try:
            line = head.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f'인코딩을 알 수 없음: {path}')
    sep = '\t' if line.count('\t') > line.count(',') else ','
    cols = [c.strip().strip('"') for c in line.rstrip('\r\n').split(sep)]
    return enc, sep, cols


def resolve(cols, keys):
    m = {}
    for k in keys:
        hit = [c for c in COLS[k] if c in cols]
        if not hit and k in OPTIONAL_COLS:
            continue
        if not hit:
            raise SystemExit(f'컬럼을 찾지 못함: {COLS[k]} (파일 컬럼: {cols})')
        m[k] = hit[0]
    return m


def norm_station(u):
    try:
        f = float(u)
        if f == int(f):
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return str(u).strip()


def station_keys(s):
    codes, uniq = pd.factorize(s)
    keys = np.array([norm_station(u) for u in uniq] + [''], dtype=object)
    return keys[codes]                                  # codes == -1 → ''


def day_index(s, t0):
    codes, uniq = pd.factorize(s)
    d = pd.to_datetime(pd.Series(uniq).astype(str).str.strip(), errors='coerce')
    days = np.append(((d - t0).dt.days).fillna(-1).astype(np.int64).to_numpy(), -1)
    return days[codes]


def num(s):
    return pd.to_numeric(s, errors='coerce').to_numpy(dtype=np.float64)


def flag_of(s):
    v = pd.to_numeric(s, errors='coerce').to_numpy(dtype=np.float64)
    txt = s.astype(str).str.strip().str.lower().isin(['true', 't', 'y', 'yes']).to_numpy()
    return (v == 1) | txt


def read_chunks(path, keys):
    enc, sep, cols = sniff(path)
    m = resolve(cols, keys)
    for ch in pd.read_csv(path, sep=sep, encoding=enc, usecols=list(set(m.values())), chunksize=CHUNK,
                          low_memory=False):
        yield m, ch


# ------------------------------------------------------------------ 1. 대여소 고르기 (CSV 1회차)
def scan_stations(files, t0):
    parts = []
    for label, path in files.items():
        ts = time.time()
        n = 0
        for m, ch in read_chunks(path, ['station', 'date', 'rent', 'ret']):
            df = pd.DataFrame({'s': station_keys(ch[m['station']]), 'd': day_index(ch[m['date']], t0),
                               'u': np.nan_to_num(num(ch[m['rent']])) + np.nan_to_num(num(ch[m['ret']]))})
            df = df[(df['d'] >= 0) & (df['s'] != '')]
            parts.append(df.groupby('s').agg(dmin=('d', 'min'), dmax=('d', 'max'), u=('u', 'sum')))
            n += len(ch)
            print(f'    [대여소 확인] {label} {n:,}행 ({time.time() - ts:.0f}초)', end='\r', flush=True)
        print(' ' * 70, end='\r')
    allp = pd.concat(parts)
    return allp.groupby(level=0).agg(dmin=('dmin', 'min'), dmax=('dmax', 'max'), u=('u', 'sum'))


def pick_stations(info, t0, last_year):
    """데이터에 나온 대여소 전체 (번호 길이·번호 순)."""
    jan_end = (pd.Timestamp(f'{t0.year}-01-31') - t0).days
    dec_start = (pd.Timestamp(f'{last_year}-12-01') - t0).days
    cont = (info['dmin'] <= jan_end) & (info['dmax'] >= dec_start)
    out(f'데이터에 나온 대여소 {len(info):,}곳 (그중 4년 내내 있는 곳 {int(cont.sum()):,}곳)')
    return sorted(info.index, key=lambda s: (len(s), s))


# ------------------------------------------------------------------ 2. 격자 배열 만들기 (CSV 2회차)
def build_arrays(files, t0, H, stations):
    S = len(stations)
    sidx = {s: i for i, s in enumerate(stations)}
    A = {
        'rent': np.zeros((S, H), np.int16), 'ret': np.zeros((S, H), np.int16),
        'present': np.zeros((S, H), bool), 'real': np.zeros((S, H), bool),
        'stock': np.full((S, H), np.nan, np.float32), 'stats': np.zeros((S, H, len(STATS)), np.float32),
        'rows_h': np.zeros(H, np.int64), 'interp_h': np.zeros(H, np.int64),
        'lat': np.full(S, np.nan), 'lon': np.full(S, np.nan), 'docks': np.full(S, np.nan),
        'holiday_d': np.zeros(H // 24, np.int8),
    }
    wsum, wcnt = np.zeros((H, len(WEATHER))), np.zeros((H, len(WEATHER)))
    gu = [''] * S
    op = [''] * S
    for label, path in files.items():
        ts = time.time()
        n = 0
        for m, ch in read_chunks(path, list(COLS)):
            d = day_index(ch[m['date']], t0)
            hr = pd.to_numeric(ch[m['hour']], errors='coerce').fillna(-1).to_numpy().astype(np.int64)
            h = d * 24 + hr
            ok = (d >= 0) & (hr >= 0) & (hr <= 23) & (h < H)
            stockv = num(ch[m['stock']])
            miss = np.isnan(stockv) | flag_of(ch[m['flag']])

            # 전 대여소 기준: 수집 장애 판정, 기상, 공휴일
            A['rows_h'] += np.bincount(h[ok], minlength=H)
            A['interp_h'] += np.bincount(h[ok & miss], minlength=H)
            for j, k in enumerate(WEATHER):
                v = num(ch[m[k]])
                g = ok & ~np.isnan(v)
                wsum[:, j] += np.bincount(h[g], weights=v[g], minlength=H)
                wcnt[:, j] += np.bincount(h[g], minlength=H)
            hv = num(ch[m['holiday']])
            A['holiday_d'][d[ok & (hv == 1)]] = 1

            # 고른 대여소
            codes, uniq = pd.factorize(station_keys(ch[m['station']]))
            mp = np.array([sidx.get(u, -1) for u in uniq] + [-1], dtype=np.int64)
            si = mp[codes]
            sel = ok & (si >= 0)
            if sel.any():
                i, hh = si[sel], h[sel]
                A['rent'][i, hh] = np.clip(np.nan_to_num(num(ch[m['rent']])[sel]), 0, 32000)
                A['ret'][i, hh] = np.clip(np.nan_to_num(num(ch[m['ret']])[sel]), 0, 32000)
                A['present'][i, hh] = True
                r = ~miss[sel]
                A['real'][i[r], hh[r]] = True
                A['stock'][i[r], hh[r]] = stockv[sel][r]
                for j, k in enumerate(STATS):
                    A['stats'][i, hh, j] = np.nan_to_num(num(ch[m[k]])[sel])
                for k in ('lat', 'lon', 'docks'):
                    v = num(ch[m[k]])[sel]
                    g = ~np.isnan(v)
                    A[k][i[g]] = v[g]
                gdf = pd.DataFrame({'i': i, 'g': ch[m['gu']].astype(str).to_numpy()[sel]}).drop_duplicates('i', keep='last')
                for ii, gg in zip(gdf['i'], gdf['g']):
                    gu[ii] = gg
                if 'op' in m:
                    odf = pd.DataFrame({'i': i, 'o': ch[m['op']].astype(str).to_numpy()[sel]}).drop_duplicates('i', keep='last')
                    for ii, oo in zip(odf['i'], odf['o']):
                        op[ii] = oo
            n += len(ch)
            print(f'    [배열 만들기] {label} {n:,}행 ({time.time() - ts:.0f}초)', end='\r', flush=True)
        print(' ' * 70, end='\r')
    w = np.divide(wsum, wcnt, out=np.full_like(wsum, np.nan), where=wcnt > 0)
    A['weather'] = pd.DataFrame(w).ffill().bfill().fillna(0).to_numpy(np.float32)
    A['gu'] = np.array(gu, dtype=str)
    A['op'] = np.array(op, dtype=str)                    # 운영방식 (QR / LCD / LCD,QR), 없으면 빈 문자열
    A['stations'] = np.array(stations, dtype=str)
    return A


def load_or_build(folder, files, t0, H, n_stations=0, continuous=False, last_year=None):
    """격자 배열 캐시 gru_data_all.npz 를 읽거나 만듦 (전 대여소). 이용 통계(stats)는 입력에 안 써서 읽지 않음.
    n_stations·continuous 는 예전 코드와 같은 모양으로 부를 수 있게 남긴 인자 (0, False 만 됨)."""
    if n_stations or continuous:
        sys.exit('이 코드는 전 대여소만 다룸 (n_stations=0, continuous=False)')
    tag = 'all'
    sig = '|'.join(f'{os.path.basename(p)}:{os.path.getsize(p)}:{int(os.path.getmtime(p))}' for p in files.values())
    sig += f'|{tag}|seed={SEED}'
    path = os.path.join(folder, f'gru_data_{tag}.npz')
    A = load_cache(path, sig, '저장된 데이터 사용', skip=('stats',))
    if A is not None:
        return A
    need = estimate_bytes(files, H)
    check_space(folder, need, '데이터 캐시')                       # CSV를 다 읽기 전에 먼저 확인
    out('CSV 읽는 중 (처음 한 번만, 두 번 읽음) ...')
    info = scan_stations(files, t0)
    stations = pick_stations(info, t0, last_year or SPLITS['test'][1])
    A = build_arrays(files, t0, H, stations)
    save_cache(path, sig, A, '데이터 저장')
    A.pop('stats', None)
    return A


def estimate_bytes(files, H, n_st=3200):
    """데이터 캐시 크기 어림값 (대여소 수를 모를 때 넉넉히)."""
    per = 2 + 2 + 1 + 1 + 4 + 4 * len(STATS)                     # rent, ret, present, real, stock, stats
    return n_st * H * per


def check_space(folder, need, label):
    import shutil
    free = shutil.disk_usage(os.path.abspath(folder)).free
    if free < need * 1.1 + 300e6:
        sys.exit(f'디스크 공간 부족: {label}에 약 {need / 1e9:.1f}GB 필요, 남은 공간 {free / 1e9:.1f}GB '
                 f'({os.path.abspath(folder)}). 공간을 비운 뒤 다시 실행할 것')


def load_cache(path, sig, msg, skip=()):
    """캐시를 읽음 (skip 에 든 배열은 빼고). 서명이 다르면 None, 깨진 파일(쓰다 멈춘 것 등)이면 지우고 None."""
    if not os.path.exists(path):
        return None
    try:
        z = np.load(path, allow_pickle=False)
        if str(z['sig']) != sig:
            return None
        A = {k: z[k] for k in z.files if k != 'sig' and k not in skip}
        out(f'{msg}: {os.path.basename(path)}')
        return A
    except Exception as e:
        out(f'※ {os.path.basename(path)} 가 깨져 있어 지우고 새로 만듦 ({type(e).__name__}: {e})')
        try:
            os.remove(path)
        except OSError:
            pass
        return None


def save_cache(path, sig, arrays, msg):
    """공간을 확인하고 임시 파일에 다 쓴 뒤 이름을 바꿈 (도중에 멈춰도 깨진 캐시가 남지 않음)."""
    need = sum(int(np.asarray(v).nbytes) for v in arrays.values())
    check_space(os.path.dirname(path) or '.', need, os.path.basename(path))
    tmp = path[:-4] + '_tmp.npz'
    try:
        np.savez(tmp, sig=sig, **arrays)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    out(f'{msg}: {os.path.basename(path)} ({need / 1e9:.1f}GB)')


def load_file_stock(folder, files, t0, H, stations, tag):
    """--stock-source file 용: 데이터 파일의 현재재고(보간값 포함)와 재고_채움방식을 (대여소, 시간) 배열로.
    처음 한 번 CSV를 한 번 더 읽어 gru_stock_file_<tag>.npz 로 저장 (gru_data_<tag>.npz 는 건드리지 않음).
    fill: −1 = 그 시간 행 없음(이용 0건), 0 = 실측, 1~7 = 채움방식, 9 = 보간이지만 방식 열 없음"""
    import zlib
    sig = '|'.join(f'{os.path.basename(p)}:{os.path.getsize(p)}:{int(os.path.getmtime(p))}' for p in files.values())
    sig += f'|{tag}|S={len(stations)}|{zlib.crc32(",".join(stations).encode())}'
    path = os.path.join(folder, f'gru_stock_file_{tag}.npz')
    Z = load_cache(path, sig, '저장된 파일 재고 사용')
    if Z is not None:
        return Z
    S = len(stations)
    check_space(folder, S * H * 5, '파일 재고 캐시')
    out('데이터 파일 현재재고 읽는 중 (처음 한 번만, CSV를 한 번 더 읽음) ...')
    sidx = {s: i for i, s in enumerate(stations)}
    stock = np.full((S, H), np.nan, np.float32)
    fill = np.full((S, H), -1, np.int8)
    for label, p in files.items():
        ts = time.time()
        n = 0
        for m, ch in read_chunks(p, ['station', 'date', 'hour', 'stock', 'flag', 'fillm']):
            d = day_index(ch[m['date']], t0)
            hr = pd.to_numeric(ch[m['hour']], errors='coerce').fillna(-1).to_numpy().astype(np.int64)
            h = d * 24 + hr
            ok = (d >= 0) & (hr >= 0) & (hr <= 23) & (h < H)
            codes, uniq = pd.factorize(station_keys(ch[m['station']]))
            mp = np.array([sidx.get(u, -1) for u in uniq] + [-1], dtype=np.int64)
            si = mp[codes]
            sel = ok & (si >= 0)
            if sel.any():
                i, hh = si[sel], h[sel]
                stock[i, hh] = num(ch[m['stock']])[sel]
                fl = flag_of(ch[m['flag']])[sel]
                if 'fillm' in m:
                    fmv = np.nan_to_num(num(ch[m['fillm']])[sel], nan=9).astype(np.int64)
                    fmv = np.where(fl & (fmv == 0), 9, np.where(~fl, 0, fmv))      # 보간여부와 맞춤
                else:
                    fmv = np.where(fl, 9, 0)
                fill[i, hh] = np.clip(fmv, 0, 9).astype(np.int8)
            n += len(ch)
            print(f'    [파일 재고] {label} {n:,}행 ({time.time() - ts:.0f}초)', end='\r', flush=True)
        print(' ' * 70, end='\r')
    Z = {'stock': stock, 'fill': fill}
    save_cache(path, sig, Z, '파일 재고 저장')
    return Z


# ------------------------------------------------------------------ 3. 재고 채우기
def fill_stock(stock, real, present, net, sys_h, train_mask, fallback):
    """한 대여소의 재고 채우기. 반환: 채운 재고, 결측 표시(0/1), 경과 시간(0~1).

    실측 → 그대로
    실측이 아니고, 직전 실측 이후 수집 장애가 없었음 (0건 시간) → 직전 실측 + 누적 순증감, 결측 표시 0
    수집 장애 (보간 행, 또는 전 대여소 장애 시간의 0건 시간, 그 뒤 실측 전까지)
        직전 실측 후 24시간 이하 → 직전 실측 + 누적 순증감
        24시간 초과 → 직전 실측 날까지 4주간의 같은 시간대 실측 평균
    """
    H = len(stock)
    t = np.arange(H)
    last = np.maximum.accumulate(np.where(real, t, -1))
    has = last >= 0
    p = np.where(has, last, 0)
    cs = np.concatenate(([0.0], np.cumsum(net, dtype=np.float64)))
    anchor = stock[p].astype(np.float64) + (cs[t] - cs[p])

    fail = (present & ~real) | (~present & sys_h)
    cf = np.concatenate(([0], np.cumsum(fail)))
    flag = ~real & (~has | ((cf[t + 1] - cf[p + 1]) > 0))
    delta = np.where(has, t - p, 10 ** 6)

    days = H // 24
    v = np.where(real, stock, np.nan).astype(np.float64)[:days * 24].reshape(days, 24)
    ok = ~np.isnan(v)
    csum = np.vstack([np.zeros(24), np.cumsum(np.where(ok, v, 0), 0)])
    ccnt = np.vstack([np.zeros(24), np.cumsum(ok, 0)])
    dd = np.arange(days)
    lo = np.maximum(dd - MEAN_DAYS + 1, 0)
    rsum, rcnt = csum[dd + 1] - csum[lo], ccnt[dd + 1] - ccnt[lo]
    R = np.divide(rsum, rcnt, out=np.full_like(rsum, np.nan), where=rcnt > 0)

    tv = stock[real & train_mask]
    st_mean = float(np.mean(tv)) if len(tv) else fallback
    m4 = R[np.minimum(p // 24, days - 1), t % 24]
    m4 = np.where(np.isnan(m4) | ~has, st_mean, m4)

    filled = np.where(real, stock, np.where(flag & (delta > SHORT_GAP), m4, anchor))
    filled = np.where(np.isnan(filled), fallback, np.maximum(filled, 0))
    elapsed = np.where(flag, np.minimum(delta, SHORT_GAP) / SHORT_GAP, 0.0)
    return filled, flag, elapsed


# ------------------------------------------------------------------ 4. 피처 만들기
class Data:
    def __init__(self, A, t0, H):
        self.t0, self.H = t0, H
        self.stations = [str(s) for s in A['stations']]
        S = self.S = len(self.stations)
        ts = t0 + pd.to_timedelta(np.arange(H), unit='h')
        self.year = ts.year.to_numpy()
        self.month = ts.month.to_numpy()
        self.hod = ts.hour.to_numpy()
        self.dow = ts.dayofweek.to_numpy()
        train_mask = (self.year >= SPLITS['train'][0]) & (self.year <= SPLITS['train'][1])
        self.train_mask = train_mask
        self.use_profile = True                                        # 미래 칸에 평소 패턴 4개
        self.use_emb = False                                           # 미래 칸에 대여소 속성표 5개
        self.emb_tab = self.emb_has = self.emb_any = None
        self.emb_source = None                                         # 'self' (코드가 계산) / 'team' (팀 표 파일)
        self.report_grp = None                                         # 결과를 나눠 볼 400곳 (팀 속성표에 있는 대여소)
        self.use_past_profile = False                                  # 입력 창 각 시각의 평소 패턴 2개
        self.CS = None
        self.use_neighbors = False
        self.nb_idx = self.nb_w = self.NB = None
        self.excl = None                                               # 학습에서 뺄 시각 (H,) bool
        self.stock_crits = []                                          # --stock-eval 상태 기준 [(이름, 부족 비율, 최소 대수, 과잉 비율)]

        rows_h, interp_h = A['rows_h'], A['interp_h']
        self.sys_h = (rows_h >= SYS_MIN_ROWS) & (interp_h > SYS_RATIO * rows_h)

        present = A['present']
        has = present.any(1)
        self.first = np.where(has, present.argmax(1), H)              # 첫 기록 시각
        self.last = np.where(has, H - 1 - present[:, ::-1].argmax(1), -1)

        rent, ret = A['rent'], A['ret']
        self.rent, self.ret = rent, ret                                # int16, 정답에 씀
        self.real = A['real']                                          # 재고 실측 여부 (--stock-eval 의 '실측만' 점수)
        self.D = np.zeros((S, H, N_DYN), np.float16)                   # 대여소별로 바뀌는 5개
        self.scale = {}

        def put(col, x, mask):
            xm = x[mask]
            mu, sd = float(xm.mean()), float(xm.std()) + 1e-6
            self.scale[col] = (mu, sd)
            return mu, sd

        # 0, 1: 대여·반납 (log1p 후 표준화)
        for c, x in ((0, rent), (1, ret)):
            mu, sd = put(c, np.log1p(x[:, train_mask].astype(np.float32)), slice(None))
            for i0 in range(0, S, 256):
                self.D[i0:i0 + 256, :, c] = (np.log1p(x[i0:i0 + 256].astype(np.float32)) - mu) / sd

        # 2~4: 재고, 결측 표시, 경과 시간
        tr_real = A['stock'][:, train_mask][A['real'][:, train_mask]]
        mu_s, sd_s = float(tr_real.mean()), float(tr_real.std()) + 1e-6
        self.scale[2] = (mu_s, sd_s)
        del tr_real
        self.HM = np.full((S, 24), mu_s, np.float32)                  # 학습 기간 대여소×시간대 재고 평균
        for i in range(S):
            f, g, e = fill_stock(A['stock'][i], A['real'][i], present[i],
                                 (ret[i].astype(np.float64) - rent[i]), self.sys_h, train_mask, mu_s)
            self.D[i, :, 2] = (f - mu_s) / sd_s
            self.D[i, :, 3] = g
            self.D[i, :, 4] = e
            sel = A['real'][i] & train_mask
            if sel.any():
                v, hh = A['stock'][i][sel].astype(np.float64), self.hod[sel]
                c = np.bincount(hh, minlength=24)
                self.HM[i] = np.where(c > 0, np.bincount(hh, weights=v, minlength=24) / np.maximum(c, 1), v.mean())

        # 서울 공통 14개: 시간·날짜 8 + 기상 6
        w = A['weather'].astype(np.float64).copy()
        self.rain_raw = w[:, WEATHER.index('rain')].copy()
        for k in ('rain', 'snow'):
            j = WEATHER.index(k)
            w[:, j] = np.log1p(np.maximum(w[:, j], 0))
        mu, sd = w[train_mask].mean(0), w[train_mask].std(0) + 1e-6
        w = (w - mu) / sd
        month = ts.month.to_numpy()
        hol = A['holiday_d'][np.arange(H) // 24].astype(np.float64)
        self.hol_h = hol > 0.5
        tf = np.stack([np.sin(2 * np.pi * self.hod / 24), np.cos(2 * np.pi * self.hod / 24),
                       np.sin(2 * np.pi * self.dow / 7), np.cos(2 * np.pi * self.dow / 7),
                       np.sin(2 * np.pi * (month - 1) / 12), np.cos(2 * np.pi * (month - 1) / 12),
                       (self.dow >= 5).astype(np.float64), hol], 1)
        self.G = np.concatenate([tf, w], 1).astype(np.float32)        # (H, 14)

        # 대여소 고정 속성
        self.lat_raw = A['lat'].astype(np.float64).copy()
        self.lon_raw = A['lon'].astype(np.float64).copy()
        dk = A['docks'].astype(np.float64)
        self.docks_ok = np.isfinite(dk) & (dk > 0)                     # 거치대수 0은 원본 결측을 0으로 채운 것
        self.docks_safe = np.where(self.docks_ok, dk, 1.0)
        st = np.stack([A['lat'], A['lon'], A['docks']], 1).astype(np.float64)
        st = np.where(np.isnan(st), np.nanmean(st, 0), st)
        self.static = ((st - st.mean(0)) / (st.std(0) + 1e-6)).astype(np.float32)   # (S, 3)
        if 'op' in A and (np.char.str_len(A['op'].astype(str)) > 0).any():      # 운영방식 → QR 여부, LCD 여부
            opv = A['op'].astype(str)
            qr, lcd = np.char.find(opv, 'QR') >= 0, np.char.find(opv, 'LCD') >= 0
            self.static = np.concatenate([self.static, np.stack([qr, lcd], 1).astype(np.float32)], 1)   # (S, 5)
            self.op_counts = (int((qr & ~lcd).sum()), int((lcd & ~qr).sum()), int((qr & lcd).sum()))
        else:
            self.op_counts = None
        self.gu_names = sorted(set(A['gu'].tolist()))
        gi = {g: k for k, g in enumerate(self.gu_names)}
        self.gu = np.array([gi[g] for g in A['gu']], dtype=np.int64)

        # 정답 통계 (학습 기간)
        net_sq = 0.0
        n_tr = 0
        sum_rent = sum_ret = 0.0
        for i0 in range(0, S, 256):
            r = rent[i0:i0 + 256, train_mask].astype(np.float64)
            t_ = ret[i0:i0 + 256, train_mask].astype(np.float64)
            net_sq += ((t_ - r) ** 2).sum()
            sum_rent += r.sum()
            sum_ret += t_.sum()
            n_tr += r.size
        self.mean_rent, self.mean_ret = sum_rent / n_tr, sum_ret / n_tr
        mean_net = self.mean_ret - self.mean_rent
        self.sd_net = float(np.sqrt(max(net_sq / n_tr - mean_net ** 2, 1e-6)))

        # 기준선 2: 학습 기간 대여소 × 요일 × 시간대 평균 (대여소 운영 기간만)
        key_all = self.dow * 24 + self.hod
        self.M = np.zeros((S, WEEK, 2), np.float32)
        self.mean_use_station = np.full(S, np.nan)
        for i in range(S):
            a, b = self.first[i], min(self.last[i], np.flatnonzero(train_mask)[-1])
            if b < a:
                continue
            k = key_all[a:b + 1]
            cnt = np.bincount(k, minlength=WEEK).astype(np.float64)
            self.M[i, :, 0] = np.bincount(k, weights=rent[i, a:b + 1], minlength=WEEK) / np.maximum(cnt, 1)
            self.M[i, :, 1] = np.bincount(k, weights=ret[i, a:b + 1], minlength=WEEK) / np.maximum(cnt, 1)
            self.mean_use_station[i] = (rent[i, a:b + 1].astype(np.float64) + ret[i, a:b + 1]).mean()
        # 학습 기간에 기록이 없는 대여소(새 대여소): 기준선2는 기록 있는 대여소들의 평균으로 대신함
        self.new_station = np.isnan(self.mean_use_station)
        if self.new_station.any() and (~self.new_station).any():
            self.M[self.new_station] = self.M[~self.new_station].mean(0)
        # 대여소 ID 임베딩 번호: 학습 기간 기록이 없는 대여소는 '모름'(= S)
        self.sid = np.where(self.new_station, S, np.arange(S)).astype(np.int64)

    # ---- 학습에서 뺄 시각 (--exclude-hours)
    def set_exclude(self, spec):
        self.excl = parse_exclude(spec, self.t0, self.H) if spec else None
        if self.excl is not None:
            hrs = np.flatnonzero(self.excl)
            out(f'학습에서 뺄 시각: {len(hrs)}시각 ({spec}). 정답 시각이나 입력 창에 이 시각이 든 학습 샘플은 뽑지 않음')

    def _excluded(self, t):
        """기준 시각 t 의 입력 창(t−L+1 ~ t) 또는 정답(t+1 ~ t+3)에 뺄 시각이 있는지."""
        if self.excl is None:
            return np.zeros(len(t), bool)
        c = np.concatenate(([0], np.cumsum(self.excl)))
        lo = np.clip(t - L + 1, 0, self.H)
        hi = np.clip(t + HORIZON + 1, 0, self.H)
        return (c[hi] - c[lo]) > 0

    # ---- 날종류
    def day_type(self):
        """시간마다 날종류: 0 평일, 1 토, 2 일·공휴일 (속성표와 같은 기준)."""
        return np.where(self.hol_h | (self.dow == 6), 2, np.where(self.dow == 5, 1, 0)).astype(np.int64)

    # ---- 대여소 속성표를 코드가 직접 계산 (--emb-source self)
    def self_emb_tables(self, years=None):
        """격자 데이터로 해마다 대여소 × 날종류 × 시간대의 평균순증감·순증감표준편차·평균이용량·이용량표준편차를 계산.
        팀 표(station_embedding_YYYY.csv)와 같은 열로 돌려줌: {연도: DataFrame}.
        대여소 운영 기간(첫 기록 ~ 마지막 기록) 안의 시간만, 이용 0건 시간도 포함. 표준편차는 표본 표준편차.
        학습 기간 해(2022·2023)만 만들고, 해마다 쓸 표는 팀 표와 같은 '다른 해' 규칙으로 고름."""
        ts = time.time()
        years = years or list(range(SPLITS['train'][0], SPLITS['train'][1] + 1))
        key = self.day_type() * 24 + self.hod                         # 0~71
        tabs = {}
        for y in years:
            idx = np.flatnonzero(self.year == y)
            if len(idx) == 0:
                continue
            y0, y1 = int(idx[0]), int(idx[-1])
            rows = []
            for i in range(self.S):
                lo, hi = max(y0, int(self.first[i])), min(y1, int(self.last[i]))
                if hi < lo:
                    continue
                k = key[lo:hi + 1]
                r = self.rent[i, lo:hi + 1].astype(np.float64)
                q = self.ret[i, lo:hi + 1].astype(np.float64)
                c = np.bincount(k, minlength=72).astype(np.float64)
                res = [c]
                for v in (q - r, q + r):
                    s1 = np.bincount(k, weights=v, minlength=72)
                    s2 = np.bincount(k, weights=v * v, minlength=72)
                    m = s1 / np.maximum(c, 1)
                    var = np.where(c > 1, (s2 - c * m * m) / np.maximum(c - 1, 1), 0.0)
                    res += [m, np.sqrt(np.maximum(var, 0))]
                c, mn, sn, mu_, su = res
                g = np.flatnonzero(c > 0)
                rows.append(pd.DataFrame({'대여소번호': self.stations[i], '날종류': [EMB_DAYTYPES[j // 24] for j in g],
                                          '시간대': g % 24, '관측수': c[g].astype(int), '평균순증감': mn[g],
                                          '순증감표준편차': sn[g], '평균이용량': mu_[g], '이용량표준편차': su[g]}))
            tabs[y] = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=['대여소번호', '날종류', '시간대'] + EMB_STATS)
            out(f'대여소 속성표 코드 계산 {y}년: {tabs[y]["대여소번호"].nunique():,}곳 ({time.time() - ts:.0f}초)')
        return tabs

    def set_report_group(self, team_paths):
        """결과를 나눠 볼 400곳: 팀 속성표에 있는 대여소."""
        sid = {s: i for i, s in enumerate(self.stations)}
        g = np.zeros(self.S, bool)
        for p in team_paths.values():
            for v in read_emb_table(p)['대여소번호'].map(norm_station).unique():
                if v in sid:
                    g[sid[v]] = True
        self.report_grp = g if g.any() else None

    def apply_file_stock(self, F, skip=(), zero_rule='keep'):
        """--stock-source file: 재고 입력(동적 입력 2번)의 결측 자리를 데이터 파일 현재재고(전처리 보간값)로 바꿈.
        결측 표시(3번)·경과 시간(4번)은 그대로 둬서 '채운 값'만 다르게 비교함.
          파일에 행이 있는 시간 → 파일 현재재고 (skip 에 든 채움방식이면 모델 규칙 값 유지)
          파일에 행이 없는 시간(이용 0건)
            zero_rule 'net' (최종 모델): 직전 행 재고 + 직전 행 시간의 순증감
              재고(h) = h시 정각 값이라, 직전 행 시간(h−1)의 대여·반납이 끝난 뒤 재고 = 재고(h−1) + 순증감(h−1).
              0건 시간이 이어지면 순증감 0 이라 그 값이 계속 이어짐 (모델 규칙 fill_stock 의 '직전 실측 + 누적 순증감'과 같은 식)
            zero_rule 'keep' (예전 실험 모델): 직전 행 재고를 그대로 (순증감을 더하지 않음)
          첫 행 이전 → 모델 규칙 값"""
        ts = time.time()
        mu, sd = self.scale[2]
        t = np.arange(self.H)
        split_of = np.full(self.H, '', dtype=object)
        for nm, (y0, y1) in SPLITS.items():
            split_of[(self.year >= y0) & (self.year <= y1)] = nm
        codes = sorted(set(FILL_NAMES))
        cnt = {nm: np.zeros(len(codes) + 1, np.int64) for nm in SPLITS}        # 방식별 행 수 (마지막 칸: 그밖)
        dsum = {nm: np.zeros(2) for nm in SPLITS}                              # 보간 행에서 |파일 − 모델| 합, 개수
        ci = {c: k for k, c in enumerate(codes)}
        for i in range(self.S):
            fs = F['stock'][i].astype(np.float64)
            fm = F['fill'][i].astype(np.int64)
            present = fm >= 0
            model_v = self.D[i, :, 2].astype(np.float64) * sd + mu
            use = present & ~np.isnan(fs)
            if skip:
                use &= ~np.isin(fm, list(skip))
            v = np.where(use, fs, np.where(present, model_v, np.nan))
            last = np.maximum.accumulate(np.where(~np.isnan(v), t, -1))
            p = np.maximum(last, 0)
            if zero_rule == 'net':                                     # 직전 행 재고 + 그 뒤 누적 순증감
                net = self.ret[i].astype(np.float64) - self.rent[i].astype(np.float64)
                cs = np.concatenate(([0.0], np.cumsum(net)))
                v = np.where(last >= 0, v[p] + (cs[t] - cs[p]), model_v)
            else:
                v = np.where(last >= 0, v[p], model_v)
            v = np.maximum(np.nan_to_num(v, nan=mu), 0)
            self.D[i, :, 2] = (v - mu) / sd
            interp = present & (fm >= 1)
            for nm in SPLITS:
                m = split_of == nm
                if not m.any():
                    continue
                k = np.array([ci.get(c, len(codes)) for c in fm[m & present]], dtype=np.int64)
                cnt[nm] += np.bincount(k, minlength=len(codes) + 1)
                g = m & interp & use
                dsum[nm] += (np.abs(fs[g] - model_v[g]).sum(), g.sum())
        self.stock_source = 'file'
        out('')
        out(f'[재고 입력: 데이터 파일 보간값] 결측 자리 재고를 파일의 현재재고로 씀 ({time.time() - ts:.0f}초)'
            + (f'  (모델 규칙 유지: 채움방식 {",".join(map(str, skip))})' if skip else ''))
        out('  결측 표시·경과 시간은 모델과 같음. 파일에 행이 없는 시간(이용 0건)은 '
            + ('직전 행 재고 + 직전 행 시간 순증감' if zero_rule == 'net' else '직전 값 유지 (예전 방식)'))
        out(f'  {"채움방식":<34}' + ''.join(f'{nm:>14}' for nm in SPLITS))
        for c in codes:
            if all(cnt[nm][ci[c]] == 0 for nm in SPLITS):
                continue
            note = f'  ※ {FILL_FUTURE[c]}' if c in FILL_FUTURE else ''
            used = '' if c == 0 else ('  (모델 규칙 유지)' if c in skip else '')
            out(f'  {str(c) + " " + FILL_NAMES[c]:<34}' + ''.join(f'{int(cnt[nm][ci[c]]):>14,}' for nm in SPLITS) + used + note)
        for nm in SPLITS:
            s, n = dsum[nm]
            if n:
                out(f'  {nm}: 보간 행에서 파일 값과 모델 규칙 값의 평균 차이 {s / n:.2f}대 ({int(n):,}행)')
        fut = [c for c in sorted(FILL_FUTURE_EVAL) if c not in skip and any(cnt[nm][ci[c]] for nm in ('val', 'test'))]
        if fut:
            n_fut = {nm: int(sum(cnt[nm][ci[c]] for c in fut)) for nm in ('val', 'test')}
            out(f'  ※ 채움방식 {fut}(신설 대여소 전체 평균)은 검증·테스트 입력에 뒤쪽 기간 값이 일부 섞임 '
                f'(검증 {n_fut["val"]:,}행, 테스트 {n_fut["test"]:,}행). 보고서에 한 줄로 적고, 빼려면 --file-fill-skip 7')

    # ---- 대여소 속성표 (station_embedding_YYYY.csv)
    def load_station_emb(self, paths, source='team'):
        """paths: {연도: 파일 경로 또는 DataFrame(코드 계산 표)}. 표를 읽어 (대여소, 표, 날종류, 시간대, 값 4개)로 만들고,
        시간마다 어느 해 표를 쓸지('다른 해' 표)와 날종류를 정해 둠."""
        self.emb_source = source
        years = sorted(paths)
        T = len(years)
        sidx = {s: i for i, s in enumerate(self.stations)}
        tab = np.full((self.S, T, 3, 24, len(EMB_STATS)), np.nan, np.float64)
        has = np.zeros((self.S, T), bool)
        for ti, y in enumerate(years):
            src = paths[y]
            df = src if isinstance(src, pd.DataFrame) else read_emb_table(src)
            name = '코드 계산 표' if isinstance(src, pd.DataFrame) else os.path.basename(src)
            st = df['대여소번호'].map(norm_station)
            i = st.map(sidx)
            dt = df['날종류'].map(day_type_code)
            hh = pd.to_numeric(df['시간대'], errors='coerce')
            if dt.isna().any():
                sys.exit(f'{name}: 날종류 값을 알 수 없음 {sorted(set(df.loc[dt.isna(), "날종류"]))[:5]}'
                         f' (평일 / 토 / 일·공휴일 이어야 함)')
            ok = i.notna() & hh.between(0, 23)
            ii, dd, hv = i[ok].astype(int).to_numpy(), dt[ok].astype(int).to_numpy(), hh[ok].astype(int).to_numpy()
            tab[ii, ti, dd, hv, :] = df.loc[ok, EMB_STATS].to_numpy(np.float64)
            has[ii, ti] = True
            n_tab, n_hit = st.nunique(), int(has[:, ti].sum())
            miss = sorted(set(st[~i.notna()]))[:5]
            out(f'대여소 속성표 {y}년: {name}  표의 대여소 {n_tab:,}곳 중 {n_hit:,}곳이 데이터와 맞음'
                + (f'  (안 맞는 예: {", ".join(miss)})' if miss else ''))
            if n_hit == 0:
                sys.exit('속성표의 대여소번호가 데이터와 하나도 안 맞음. 번호 형식을 확인할 것')
            empty = int(np.isnan(tab[has[:, ti], ti]).any(axis=(1, 2, 3)).sum())
            if empty:
                out(f'  ※ {empty:,}곳은 날종류·시간대 칸 일부가 비어 있어 그 칸을 평균값(0)으로 둠')
        # 변환·표준화: 기준값은 학습 기간 해의 표로만 계산
        tr = [ti for ti, y in enumerate(years) if SPLITS['train'][0] <= y <= SPLITS['train'][1]] or list(range(T))
        for j, name in enumerate(EMB_STATS):
            v = tab[..., j]
            if name in EMB_LOG:
                v = np.log1p(np.maximum(v, 0))
            ref = v[:, tr][has[:, tr]]
            ref = ref[np.isfinite(ref)]
            mu, sd = float(ref.mean()), float(ref.std()) + 1e-6
            tab[..., j] = (v - mu) / sd
        tab = np.nan_to_num(tab, nan=0.0)
        tab[~has] = 0.0
        self.emb_tab, self.emb_has, self.emb_any, self.emb_years = tab.astype(np.float32), has, has.any(1), years

        # 해마다 쓸 표: 직전 해 표, 없으면 가장 가까운 다음 해 표 (같은 해 표는 쓰지 않음)
        pick = {}
        for Y in range(int(self.year.min()), int(self.year.max()) + 1):
            prev, nxt = [y for y in years if y < Y], [y for y in years if y > Y]
            pick[Y] = max(prev) if prev else (min(nxt) if nxt else Y)
            role = ('학습' if SPLITS['train'][0] <= Y <= SPLITS['train'][1] else
                    '검증' if SPLITS['val'][0] <= Y <= SPLITS['val'][1] else
                    '테스트' if SPLITS['test'][0] <= Y <= SPLITS['test'][1] else '기타')
            note = ''
            if pick[Y] == Y:
                note = '  ※ 같은 해 표 (다른 해 표가 없음)'
            elif pick[Y] > Y:
                note = '  (다음 해 표: 학습 기간 안에서만 쓰임)'
            out(f'  {Y}년 {role} 샘플 ← {pick[Y]}년 표{note}')
            if role in ('검증', '테스트') and pick[Y] >= Y:
                sys.exit(f'{Y}년 {role}에 {pick[Y]}년 속성표가 쓰임 (같은 해나 미래 정보). {Y}년보다 앞선 해의 표가 있어야 함')
        yi = {y: k for k, y in enumerate(years)}
        self.emb_t = np.array([yi[pick[int(Y)]] for Y in self.year], dtype=np.int64)
        self.emb_dt = self.day_type()
        out(f'  속성표가 있는 대여소: {int(self.emb_any.sum()):,}곳 / {self.S:,}곳 (나머지는 0 + \'표 없음\')')

    def drop_emb_use(self):
        """--emb-no-use: 속성표 값 중 이용량 두 개(평균이용량·이용량표준편차)를 0(= 학습 표 평균)으로 고정. 입력 크기는 그대로."""
        if getattr(self, 'emb_use_dropped', False):
            return
        idx = [EMB_STATS.index(c) for c in ('평균이용량', '이용량표준편차')]
        self.emb_tab[..., idx] = 0.0
        self.emb_use_dropped = True
        out('속성표에서 뺀 값: 평균이용량, 이용량표준편차 (0으로 고정, 남은 값: 평균순증감, 순증감표준편차, 표 있음)')

    def emb_feats(self, s, fut_t):
        """t+1~t+3 각 시각의 속성표 값 (B, 3, 5): 값 4개 + 표 있음(0/1)."""
        ti, dt, hh = self.emb_t[fut_t], self.emb_dt[fut_t], self.hod[fut_t]
        ss = np.broadcast_to(s[:, None], fut_t.shape)
        v = self.emb_tab[ss, ti, dt, hh]
        h = self.emb_has[ss, ti].astype(np.float32)[..., None]
        return np.concatenate([v, h], -1).astype(np.float32)

    @property
    def n_future(self):
        """미래 칸 한 시각의 입력 수: 달력·기상 14 + 평소 패턴 4 + 속성표 5 = 23."""
        return self.G.shape[1] + (4 if self.use_profile else 0) + (len(EMB_STATS) + 1 if self.use_emb else 0)

    @property
    def n_in(self):
        """입력 창 한 시각의 입력 수: 대여소 자신 5 + 이웃 2 + 평소 패턴 2 + 달력·기상 14 = 23."""
        return N_DYN + (2 if self.use_neighbors else 0) + (2 if self.use_past_profile else 0) + self.G.shape[1]

    @property
    def n_static(self):
        return self.static.shape[1]

    # ---- 이웃 대여소 (거리 1.5km 안 8곳, 거리 가중)
    def set_neighbors(self, k):
        """대여소마다 거리 기준 이웃 k곳과 가중치를 정하고, 시각마다 이웃 대여·반납 가중 평균을 미리 계산."""
        k = max(1, min(k, self.S - 1))
        idx, w = self._dist_neighbors(k)
        self.nb_idx, self.nb_w = idx, w.astype(np.float32)
        self.use_neighbors = True
        ts = time.time()
        self.NB = np.zeros((self.S, self.H, 2), np.float16)          # 미리 계산해 둠 (배치마다 다시 안 셈)
        hours = np.arange(self.H)
        for i0 in range(0, self.S, 32):
            nb, ww = self.nb_idx[i0:i0 + 32], self.nb_w[i0:i0 + 32]              # (c, k)
            act = (hours[None, None, :] >= self.first[nb][..., None]) & (hours[None, None, :] <= self.last[nb][..., None])
            a = ww[..., None] * act                                                # (c, k, H)
            den = a.sum(1)                                                         # (c, H)
            for c in range(2):
                num = (a * self.D[nb, :, c].astype(np.float32)).sum(1)
                self.NB[i0:i0 + 32, :, c] = np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)
        has = (w > 0).sum(1)
        out(f'이웃 대여소: 거리 기준 최대 {k}곳, '
            f'평균 {has.mean():.1f}곳, 이웃이 없는 대여소 {int((has == 0).sum())}곳 ({time.time() - ts:.0f}초)')

    def _dist_neighbors(self, k):
        lat, lon = self.lat_raw, self.lon_raw
        ok = ~(np.isnan(lat) | np.isnan(lon))
        lat0 = np.nanmean(lat)
        x = np.where(ok, lon, 0) * np.cos(np.radians(lat0)) * 111.32         # km
        y = np.where(ok, lat, 0) * 110.57
        idx = np.zeros((self.S, k), np.int64)
        w = np.zeros((self.S, k))
        for i0 in range(0, self.S, 512):
            d2 = (x[i0:i0 + 512, None] - x[None, :]) ** 2 + (y[i0:i0 + 512, None] - y[None, :]) ** 2
            d2[np.arange(len(d2)), np.arange(i0, i0 + len(d2))] = np.inf          # 자기 자신 제외
            d2[:, ~ok] = np.inf
            d2[~ok[i0:i0 + 512]] = np.inf
            nb = np.argpartition(d2, k, axis=1)[:, :k]
            dk = np.take_along_axis(d2, nb, 1)
            idx[i0:i0 + 512] = nb
            w[i0:i0 + 512] = np.where(dk <= NB_MAX_KM ** 2, np.exp(-dk / NB_SIGMA_KM ** 2), 0.0)
        return idx, w

    # ---- 샘플 범위
    def ranges(self, split, min_hist):
        """대여소마다 기준 시각 t의 범위 [a, b]. 입력 창이 운영 기간 안에 들고 정답이 분할을 넘지 않게."""
        y0, y1 = SPLITS[split]
        idx = np.flatnonzero((self.year >= y0) & (self.year <= y1))
        if len(idx) == 0:
            return np.zeros(self.S, np.int64), -np.ones(self.S, np.int64)
        a = np.maximum(int(idx[0]), self.first + min_hist)
        b = np.minimum(int(idx[-1]) - HORIZON, self.last - HORIZON)
        return a, b

    def train_samples(self, n, rng):
        a, b = self.ranges('train', L - 1)
        ln = np.clip(b - a + 1, 0, None).astype(np.float64)
        s = rng.choice(self.S, size=n, p=ln / ln.sum())
        t = a[s] + (rng.random(n) * ln[s]).astype(np.int64)
        if self.excl is not None:                                  # 뺄 시각이 든 샘플은 다시 뽑음
            bad = self._excluded(t)
            for _ in range(100):
                if not bad.any():
                    break
                k = int(bad.sum())
                s2 = rng.choice(self.S, size=k, p=ln / ln.sum())
                s[bad] = s2
                t[bad] = a[s2] + (rng.random(k) * ln[s2]).astype(np.int64)
                bad = self._excluded(t)
        return s, t

    def eval_samples(self, split, frac=1.0, seed=SEED):
        a, b = self.ranges(split, max(L - 1, WEEK - 1))
        ln = np.clip(b - a + 1, 0, None)
        s = np.repeat(np.arange(self.S), ln)
        start = np.repeat(np.cumsum(ln) - ln, ln)
        t = a[s] + (np.arange(len(s)) - start)
        if frac < 1.0:
            keep = np.random.default_rng(seed).random(len(s)) < frac
            s, t = s[keep], t[keep]
        return s, t

    # ---- 입력 만들기
    def profile(self, s, fut_t):
        """t+1~t+3 각 시각의 평소 패턴 (B, 3, 4):
        최근 4주 같은 요일·시간대 평균 대여·반납, 1주 전 같은 시각 대여·반납. 대여·반납과 같은 방식으로 표준화.
        개설 전 주는 빼고 평균함 (모두 개설 전이면 0)."""
        lag = fut_t[:, :, None] - WEEK * np.arange(1, PROFILE_WEEKS + 1)     # (B, 3, 4)
        ok = lag >= self.first[s][:, None, None]
        lagc = np.maximum(lag, 0)
        ss = s[:, None, None]
        r = np.where(ok, self.rent[ss, lagc], 0).astype(np.float32)
        q = np.where(ok, self.ret[ss, lagc], 0).astype(np.float32)
        n = np.maximum(ok.sum(-1), 1)
        (mr, sr), (mq, sq) = self.scale[0], self.scale[1]
        f = np.stack([(np.log1p(r.sum(-1) / n) - mr) / sr, (np.log1p(q.sum(-1) / n) - mq) / sq,
                      (np.log1p(r[..., 0]) - mr) / sr, (np.log1p(q[..., 0]) - mq) / sq], -1)
        return f.astype(np.float32)

    def past_profile(self, s, win):
        """--past-profile: 입력 창 각 시각의 평소 패턴 (B, L, 2) = 최근 4주 같은 요일·시간 평균 대여·반납.
        미래 입력의 평소 패턴과 같은 계산(1~4주 전만 씀, 개설 전 주는 뺌)이라 그 시각 이후 값은 안 들어감.
        모델이 '오늘 지금까지가 평소보다 붐비는지'를 직접 비교할 수 있게 함."""
        lag = win[:, :, None] - WEEK * np.arange(1, PROFILE_WEEKS + 1)        # (B, L, 4)
        ok = lag >= self.first[s][:, None, None]
        lagc = np.maximum(lag, 0)
        ss = s[:, None, None]
        r = np.where(ok, self.rent[ss, lagc], 0).astype(np.float32)
        q = np.where(ok, self.ret[ss, lagc], 0).astype(np.float32)
        n = np.maximum(ok.sum(-1), 1)
        (mr, sr), (mq, sq) = self.scale[0], self.scale[1]
        return np.stack([(np.log1p(r.sum(-1) / n) - mr) / sr, (np.log1p(q.sum(-1) / n) - mq) / sq], -1).astype(np.float32)

    def batch(self, s, t):
        """(대여소 번호 s, 기준 시각 t) 묶음 → 입력·정답 numpy 배열."""
        win = t[:, None] + np.arange(-L + 1, 1)
        parts = [self.D[s[:, None], win].astype(np.float32)]       # (B, L, 5) 대여소 자신
        if self.use_neighbors:
            parts.append(self.NB[s[:, None], win].astype(np.float32))   # 이웃 대여·반납 2
        if self.use_past_profile:
            parts.append(self.past_profile(s, win))                # 평소 패턴 2
        x = np.concatenate(parts + [self.G[win]], -1)              # + 달력·기상 14 → (B, L, 23)
        fut_t = t[:, None] + np.arange(1, HORIZON + 1)            # (B, 3)
        fut = self.G[fut_t]                                        # (B, 3, 14)
        if self.use_profile:
            fut = np.concatenate([fut, self.profile(s, fut_t)], -1)   # (B, 3, 18)
        if self.use_emb:
            fut = np.concatenate([fut, self.emb_feats(s, fut_t)], -1)  # (B, 3, 23)
        fut = np.ascontiguousarray(fut.reshape(len(t), -1))
        y = np.stack([self.rent[s[:, None], fut_t], self.ret[s[:, None], fut_t]], -1).astype(np.float32)
        gu = np.stack([self.gu[s], self.sid[s]], 1)                # (B, 2): 자치구 번호, 대여소 번호(모르면 S)
        return x, self.static[s], gu, fut, y, fut_t

    def ensure_cs(self):
        """대여소별 누적 순증감 CS[s, i] = i시 이전까지 순증감 합 (인위 결측에 씀)."""
        if self.CS is None:
            self.CS = np.zeros((self.S, self.H + 1), np.int32)
            for i0 in range(0, self.S, 256):
                net = self.ret[i0:i0 + 256].astype(np.int32) - self.rent[i0:i0 + 256]
                self.CS[i0:i0 + 256, 1:] = np.cumsum(net, 1, dtype=np.int32)

    def outage(self, s, t, x, dur):
        """입력 창의 재고 칸(2~4)을 '기준 시각 t에 dur시간째 전 대여소 수집 장애가 이어지는 중'으로 바꿈.
        dur = 0 인 샘플은 그대로. 채우는 규칙은 fill_stock 과 같음: 장애 직전 재고 + 누적 순증감 (24시간 이하),
        24시간 초과 칸은 학습 기간 대여소×시간대 평균 (fill_stock 의 최근 4주 평균을 계산 편의상 대신함)."""
        self.ensure_cs()
        mu, sd = self.scale[2]
        pos = np.arange(L)
        start = L - dur                                            # 창 안에서 장애가 시작된 위치 (음수면 창 이전)
        inout = pos[None, :] >= start[:, None]
        if not inout.any():
            return x
        delta = (pos[None, :] - start[:, None] + 1).astype(np.float64)   # 장애 직전 시각 이후 시간
        a0 = np.maximum(t - dur, 0)                                # 장애 직전 시각 (절대 위치)
        stock0 = self.D[s, a0, 2].astype(np.float64) * sd + mu
        absp = t[:, None] - L + 1 + pos[None, :]
        anchor = stock0[:, None] + (self.CS[s[:, None], absp] - self.CS[s, a0][:, None])
        longm = self.HM[s[:, None], self.hod[absp]]
        fill = np.maximum(np.where(delta <= SHORT_GAP, anchor, longm), 0)
        x = x.copy()
        x[..., 2] = np.where(inout, (fill - mu) / sd, x[..., 2])
        x[..., 3] = np.where(inout, 1.0, x[..., 3])
        x[..., 4] = np.where(inout, np.minimum(delta, SHORT_GAP) / SHORT_GAP, x[..., 4])
        return x


# ------------------------------------------------------------------ 5. 오차 집계
class Errors:
    """순증감 오차를 지평, 시간대, 대여소, 비 여부, 재고 결측 여부로 나눠 모음. 오차 칸은 (개수, |e| 합, e² 합)."""

    def __init__(self, S):
        self.h = np.zeros((HORIZON, 3))          # [지평, (n, |e|, e²)]
        self.hour = np.zeros((24, 3))
        self.st = np.zeros((S, 3))
        self.rain = np.zeros((2, 3))
        self.miss = np.zeros((2, 3))
        self.hour_bias = np.zeros(24)            # 시간대별 (예측 − 실제) 합
        self.ys = np.zeros((HORIZON, 2))         # 지평별 정답 합, 제곱합 (R²용)
        self.cm = np.zeros((HORIZON, 3, 3))      # 과잉·부족 분류 [지평, 실제, 예측], 0 부족 / 1 균형 / 2 과잉
        self.ebin = np.zeros((HORIZON, len(ERR_BINS)))   # [지평, |오차| ≤ 구간값 인 개수]
        self.cmm = np.zeros((HORIZON, 3, 3))     # 클래스 경계 ±m 기준, 회귀 출력으로 본 방향 [지평, 실제, 예측]
        self.cmc = np.zeros((HORIZON, 3, 3))     # 클래스 경계 ±m 기준, 분류 출력으로 본 방향
        self.has_cls = False
        self.big = np.zeros((HORIZON, 2, 3))     # 큰 변화 [지평, 빠짐(≤−3)/쌓임(≥+3), (n, 회귀가 맞춤, 분류 출력이 맞춤)]
        self.nact = np.zeros((HORIZON, 3))       # 실제 클래스 수 (경계 ±m)
        self.sk = None                           # --stock-eval 재고 상태 통계 (StockStats)
        self.pstat = {}                          # 확률 출력별 판정 규칙 통계: 'cls' 분류 출력, 'dist' 분포 출력
        self.has_dist = False
        self.dmae = np.zeros((HORIZON, 2))       # 분포 중앙값 [n, |오차| 합]
        self.pi = np.zeros((HORIZON, 3))         # 80% 예측구간 [n, 적중 수, 폭 합]

    def _pstats(self, key, p3, cym):
        """방향 확률 (B, 3, 3) → 가장 큰 확률 / 절반 규칙 오차 매트릭스, 경보 기준별 (경보 수, 맞춘 수),
        확신도(가장 큰 확률) 구간별 (샘플 수, 맞춘 수)."""
        st = self.pstat.get(key)
        if st is None:
            st = self.pstat[key] = {'cm': np.zeros((HORIZON, 3, 3)), 'med': np.zeros((HORIZON, 3, 3)),
                                    'alert': np.zeros((HORIZON, 2, len(ALERT_TAUS), 2)),
                                    'conf': np.zeros((HORIZON, CONF_BINS, 2))}
        am = p3.argmax(-1)
        half = np.where(p3[..., 0] >= 0.5, 0, np.where(p3[..., 2] >= 0.5, 2, 1))
        cb = np.minimum((p3.max(-1) * CONF_BINS).astype(np.int64), CONF_BINS - 1)
        for j in range(HORIZON):
            st['cm'][j] += np.bincount(cym[:, j] * 3 + am[:, j], minlength=9).reshape(3, 3)
            st['med'][j] += np.bincount(cym[:, j] * 3 + half[:, j], minlength=9).reshape(3, 3)
            st['conf'][j, :, 0] += np.bincount(cb[:, j], minlength=CONF_BINS)
            st['conf'][j, :, 1] += np.bincount(cb[:, j], weights=(am[:, j] == cym[:, j]).astype(np.float64),
                                               minlength=CONF_BINS)
            for ci, c in enumerate((0, 2)):
                hit = cym[:, j] == c
                for ti, tau in enumerate(ALERT_TAUS):
                    pr = p3[:, j, c] >= tau
                    st['alert'][j, ci, ti] += (pr.sum(), (pr & hit).sum())

    @staticmethod
    def _acc(arr, key, e, n_key):
        arr[:, 0] += np.bincount(key, minlength=n_key)
        arr[:, 1] += np.bincount(key, weights=np.abs(e), minlength=n_key)
        arr[:, 2] += np.bincount(key, weights=e ** 2, minlength=n_key)

    def add(self, D, s, x, pred, y, fut_t, probs=None, dist=None):
        net_y = y[..., 1] - y[..., 0]
        pn = pred[..., 1] - pred[..., 0]                                        # 예측 순증감
        net_e = pn - net_y                                                      # (B, 3)
        for j in range(HORIZON):
            e, yv = net_e[:, j], net_y[:, j]
            self.h[j] += (len(e), np.abs(e).sum(), (e ** 2).sum())
            ae = np.abs(e)
            self.ebin[j] += [(ae <= b).sum() for b in ERR_BINS]
            self.ys[j] += (yv.sum(), (yv.astype(np.float64) ** 2).sum())
        cy = (np.sign(net_y) + 1).astype(np.int64)                              # 실제: 순증감 부호
        cp = np.where(pn >= 0.5, 2, np.where(pn <= -0.5, 0, 1))                 # 예측: 반올림한 순증감 부호
        for j in range(HORIZON):
            self.cm[j] += np.bincount(cy[:, j] * 3 + cp[:, j], minlength=9).reshape(3, 3)
        mg = getattr(D, 'cls_margin', 1)                                       # 클래스 경계 ±m
        cym = np.where(net_y <= -mg, 0, np.where(net_y >= mg, 2, 1))
        cpm = np.where(pn <= -mg + 0.5, 0, np.where(pn >= mg - 0.5, 2, 1))       # 반올림한 예측 순증감 기준
        cc = probs.argmax(-1) if probs is not None else None                   # 분류 출력 (B, 3)
        if cc is not None:
            self.has_cls = True
        for j in range(HORIZON):
            self.cmm[j] += np.bincount(cym[:, j] * 3 + cpm[:, j], minlength=9).reshape(3, 3)
            if cc is not None:
                self.cmc[j] += np.bincount(cym[:, j] * 3 + cc[:, j], minlength=9).reshape(3, 3)
            for k, (mask, hit_r, cls_id) in enumerate(((net_y[:, j] <= -3, pn[:, j] <= -0.5, 0),
                                                       (net_y[:, j] >= 3, pn[:, j] >= 0.5, 2))):
                hit_c = (cc[:, j] == cls_id) if cc is not None else np.zeros(len(mask), bool)
                self.big[j, k] += (mask.sum(), (mask & hit_r).sum(), (mask & hit_c).sum())
            self.nact[j] += np.bincount(cym[:, j], minlength=3)
        if probs is not None:
            self._pstats('cls', probs, cym)
        if dist is not None:                                                    # --dist-head: 칸 k ↔ 순증감 k − DIST_K
            self.has_dist = True
            lo_b, hi_b = DIST_K - mg, DIST_K + mg                               # 부족: 칸 ≤ lo_b, 과잉: 칸 ≥ hi_b
            p3 = np.stack([dist[..., :lo_b + 1].sum(-1), dist[..., lo_b + 1:hi_b].sum(-1), dist[..., hi_b:].sum(-1)], -1)
            self._pstats('dist', p3, cym)
            cdf = np.cumsum(dist, -1)
            med = (cdf < 0.5).sum(-1) - DIST_K                                  # 누적 확률이 처음 0.5 이상인 칸
            q10, q90 = (cdf < 0.1).sum(-1), np.minimum((cdf < 0.9).sum(-1), NBIN - 1)
            lo = np.where(q10 == 0, -np.inf, q10 - DIST_K)                      # 끝 칸은 그 밖까지 포함
            hi = np.where(q90 == NBIN - 1, np.inf, q90 - DIST_K)
            cover = (net_y >= lo) & (net_y <= hi)
            for j in range(HORIZON):
                self.dmae[j] += (len(net_y), np.abs(med[:, j] - net_y[:, j]).sum())
                self.pi[j] += (len(net_y), cover[:, j].sum(), (q90[:, j] - q10[:, j]).sum())
        e = net_e.reshape(-1)                                                   # (B*3,)
        ft = fut_t.reshape(-1)
        self._acc(self.hour, D.hod[ft], e, 24)
        self.hour_bias += np.bincount(D.hod[ft], weights=e, minlength=24)
        self._acc(self.st, np.repeat(s, HORIZON), e, D.S)
        self._acc(self.rain, (D.rain_raw[ft] > 0).astype(np.int64), e, 2)
        w = x[:, :, FLAG_COL].max(1) > 0.5                                  # 입력 창에 재고 결측이 있었는지
        self._acc(self.miss, np.repeat(w, HORIZON).astype(np.int64), e, 2)

    @staticmethod
    def mae(a):
        return np.divide(a[..., 1], a[..., 0], out=np.full(a.shape[:-1], np.nan), where=a[..., 0] > 0)

    @staticmethod
    def rmse(a):
        return np.sqrt(np.divide(a[..., 2], a[..., 0], out=np.full(a.shape[:-1], np.nan), where=a[..., 0] > 0))

    def r2(self):
        n = self.h[:, 0]
        sst = self.ys[:, 1] - self.ys[:, 0] ** 2 / np.maximum(n, 1)
        return 1 - self.h[:, 2] / np.maximum(sst, 1e-9)

    def bias(self):
        return self.hour_bias / np.maximum(self.hour[:, 0], 1)

    def cls3(self, j=0):
        """지평 j의 클래스별 (재현율, 정밀도, F1) 3×3 배열과 macro F1. 행 순서: 부족, 균형, 과잉."""
        c = self.cm[j]
        rec = np.divide(np.diag(c), c.sum(1), out=np.full(3, np.nan), where=c.sum(1) > 0)
        pre = np.divide(np.diag(c), c.sum(0), out=np.full(3, np.nan), where=c.sum(0) > 0)
        f1 = np.divide(2 * rec * pre, rec + pre, out=np.full(3, np.nan), where=(rec + pre) > 0)
        return np.stack([rec, pre, f1], 1), float(np.nanmean(f1))

    def cls(self, j=0):
        """지평 j의 과잉·부족 분류: (정확도, 부족 재현율, 부족 정밀도, 과잉 재현율, 과잉 정밀도)"""
        c = self.cm[j]
        f = lambda a, b: a / b if b > 0 else np.nan
        return (f(np.trace(c), c.sum()), f(c[0, 0], c[0].sum()), f(c[0, 0], c[:, 0].sum()),
                f(c[2, 2], c[2].sum()), f(c[2, 2], c[:, 2].sum()))


class StockStats:
    """--stock-eval: 대시보드 재고 상태 기준 평가.
    시점: 재고 라벨(h) = h시 정각 재고 (fill_stock 기준). 예측 시점 = t시 이용이 끝난 (t+1)시 정각.
    예측 재고 = 현재 재고 라벨(t+1) + 예측 순증감 합(t+1~t+k), 실제 재고 = 라벨(t+1+k) (t+k시 이용 뒤).
    상태: 부족 = 재고 ≤ max(최소 대수, 거치대 × a), 과잉 = 재고 ≥ 거치대 × b, 그 사이 정상.
    쓰는 샘플: 거치대수를 알고, 현재·미래 재고가 수집 장애가 아닌 시간 (실측, 또는 이용 없는 시간의 직전 재고).
    tcheck: 이어진 두 실측 시간의 재고 차이가 순증감(t) 와 맞는지 (정각 기준이 맞으면 이쪽이 맞음), 순증감(t+1) 과 비교.
    행: 'pers' 지금 상태 유지, 'pred' 이 예측(모델·기준선), 'oracle' 순증감을 완벽히 맞혔을 때(재배치 등 순증감 밖 변화의 몫).
    J[행][기준, 지평] = (지금 상태, 실제 상태, 예측 상태) 3×3×3 개수 → 정확도·새로 부족/과잉 재현율·정밀도를 모두 계산."""
    ROWS = ('pers', 'pred', 'oracle')

    def __init__(self, crits):
        self.crits = crits
        nc = len(crits)
        self.J = {r: np.zeros((nc, HORIZON, 3, 3, 3)) for r in self.ROWS}
        self.Jr = {r: np.zeros((nc, HORIZON, 3, 3, 3)) for r in self.ROWS}   # 미래 재고가 실측인 샘플만
        self.tcheck = np.zeros(5)      # 재고 시점 점검 [쌍 수, 차이=순증감(t) 수, |차이−순증감(t)| 합, 차이=순증감(t+1) 수, |…| 합]
        self.W = {r: np.zeros((nc, 2, 2, 2)) for r in self.ROWS}       # 3시간 창 [기준, 부족/과잉, 실제, 예측] (지금 정상)
        self.P = np.zeros((nc, HORIZON, 2, len(STOCK_TAUS), 2))         # 확률 경보 [기준, 지평, 부족/과잉, τ, (경보 수, 맞춘 수)]
        self.Ppos = np.zeros((nc, HORIZON, 2))                          # 지금 정상 → 실제 부족/과잉 수 (확률 경보 재현율 분모)
        self.Pn = np.zeros((nc, HORIZON))                               # 확률 경보를 센 샘플 수
        self.cov = np.zeros(2)                                          # 3시간 누적 순증감 80% 구간 [n, 적중]
        self.has_dist = False
        self.n_base = 0

    @staticmethod
    def _state(v, ts, to):
        return np.where(v <= ts, 0, 1 + (v >= to)).astype(np.int8)

    @staticmethod
    def _conv(a, b):
        out = np.zeros((a.shape[0], a.shape[1] + b.shape[1] - 1))
        for i in range(b.shape[1]):
            out[:, i:i + a.shape[1]] += a * b[:, i:i + 1]
        return out

    def add(self, D, s, x, pred, y, fut_t, dist):
        # 시점: 재고 라벨(h) = h시 정각 재고 = 라벨(h−1) + 순증감(h−1) (fill_stock 과 같은 기준).
        # 예측은 t시 이용이 끝난 (t+1)시 정각에 하므로 현재 재고 = 라벨(t+1), t+k시 이용이 끝난 뒤 실제 재고 = 라벨(t+1+k)
        mu, sd = D.scale[2]
        H_ = D.D.shape[1]
        tn = fut_t[:, 0]                                                              # (t+1)시 정각 = 예측 시점
        st0 = np.floor(D.D[s, tn, STOCK_COL].astype(np.float64) * sd + mu + 0.5)      # 현재 재고
        base = (D.D[s, tn, FLAG_COL] < 0.5) & D.docks_ok[s]
        fa = fut_t + 1                                                                # t+k시 이용이 끝난 시각
        inr = fa <= np.minimum(D.last[s], H_ - 1)[:, None]
        fa = np.minimum(fa, H_ - 1)
        sf = np.floor(D.D[s[:, None], fa, STOCK_COL].astype(np.float64) * sd + mu + 0.5)   # 실제 재고 (t+k시 이용 뒤)
        okf = (D.D[s[:, None], fa, FLAG_COL] < 0.5) & base[:, None] & inr
        realf = D.real[s[:, None], fa] & okf
        t0 = tn - 1                                                                   # 재고 시점 점검: 이어진 두 실측 시간
        pair = D.real[s, t0] & D.real[s, tn]
        if pair.any():
            st_prev = np.floor(D.D[s[pair], t0[pair], STOCK_COL].astype(np.float64) * sd + mu + 0.5)
            d = st0[pair] - st_prev                                                   # 라벨(t+1) − 라벨(t)
            n_t = (D.ret[s[pair], t0[pair]].astype(np.float64) - D.rent[s[pair], t0[pair]])          # 순증감(t)
            n_t1 = (y[pair, 0, 1] - y[pair, 0, 0]).astype(np.float64)                                 # 순증감(t+1)
            self.tcheck += (pair.sum(), (d == n_t).sum(), np.abs(d - n_t).sum(), (d == n_t1).sum(), np.abs(d - n_t1).sum())
        self.n_base += int(base.sum())
        pn = (pred[..., 1] - pred[..., 0]).astype(np.float64)
        yn = (y[..., 1] - y[..., 0]).astype(np.float64)
        stock_pred = {'pers': np.repeat(st0[:, None], HORIZON, 1),
                      'pred': np.floor(st0[:, None] + np.cumsum(pn, 1) + 0.5),
                      'oracle': st0[:, None] + np.cumsum(yn, 1)}
        dk = D.docks_safe[s]
        cums = None
        if dist is not None:                                               # 누적 순증감 분포 (시간끼리 독립 가정, 근사)
            self.has_dist = True
            c1 = dist[:, 0].astype(np.float64)
            c2 = self._conv(c1, dist[:, 1].astype(np.float64))
            c3 = self._conv(c2, dist[:, 2].astype(np.float64))
            cums = [np.cumsum(c, 1) for c in (c1, c2, c3)]                 # 칸 i ↔ 누적 순증감 i − DIST_K·(k+1)
            ok3 = okf.all(1)
            if ok3.any():
                cdf3, tot = cums[2][ok3], yn[ok3].sum(1)
                q10 = (cdf3 < 0.1).sum(1) - 3 * DIST_K
                q90 = np.minimum((cdf3 < 0.9).sum(1), cdf3.shape[1] - 1) - 3 * DIST_K
                self.cov += (ok3.sum(), ((tot >= q10) & (tot <= q90)).sum())
        nc = len(self.crits)                                               # 기준 여러 개를 한 번에
        A_ = np.array([c[1] for c in self.crits]); MN = np.array([c[2] for c in self.crits]); B_ = np.array([c[3] for c in self.crits])
        ts = np.maximum(MN[None, :], dk[:, None] * A_[None, :])           # (B, nc) 부족 경계
        to = dk[:, None] * B_[None, :]                                     # (B, nc) 과잉 경계
        cur = self._state(st0[:, None], ts, to)                            # (B, nc)
        act = self._state(sf[:, None, :], ts[:, :, None], to[:, :, None])  # (B, nc, 3)
        sts = {r: self._state(v[:, None, :], ts[:, :, None], to[:, :, None]) for r, v in stock_pred.items()}
        cidx = (np.arange(nc)[None, :, None] * HORIZON + np.arange(HORIZON)[None, None, :]) * 27   # (1, nc, 3) 기준·지평 칸
        base_code = cidx + cur[:, :, None].astype(np.int32) * 9 + act.astype(np.int32) * 3
        m3 = np.broadcast_to(okf[:, None, :], act.shape)
        mr3 = np.broadcast_to(realf[:, None, :], act.shape)
        for r in self.ROWS:
            code = base_code + sts[r]
            self.J[r] += np.bincount(code[m3], minlength=nc * HORIZON * 27).reshape(nc, HORIZON, 3, 3, 3)
            self.Jr[r] += np.bincount(code[mr3], minlength=nc * HORIZON * 27).reshape(nc, HORIZON, 3, 3, 3)
        w = okf.all(1)[:, None] & (cur == 1)                               # 3시간 창: 지금 정상인 대여소 (B, nc)
        ci2 = np.broadcast_to(np.arange(nc)[None, :], w.shape)
        for k, c in enumerate((0, 2)):
            a_ = (act == c).any(2).astype(np.int64)
            for r in self.ROWS:
                p_ = (sts[r] == c).any(2).astype(np.int64)
                self.W[r][:, k] += np.bincount((ci2 * 4 + a_ * 2 + p_)[w], minlength=nc * 4).reshape(nc, 2, 2)
        if cums is not None:                                              # 확률 경보: 지금 정상인 대여소
            for j in range(HORIZON):
                m = okf[:, j][:, None] & (cur == 1)                       # (B, nc)
                if not m.any():
                    continue
                cdf = cums[j]
                off = DIST_K * (j + 1)
                nb = cdf.shape[1]
                lo = np.floor(ts - st0[:, None]).astype(np.int64) + off   # 누적 ≤ 이 값이면 부족
                hi = np.ceil(to - st0[:, None]).astype(np.int64) + off - 1   # 누적 ≥ 이 값 + 1 이면 과잉
                g_lo = np.take_along_axis(cdf, np.clip(lo, 0, nb - 1), 1)
                g_hi = np.take_along_axis(cdf, np.clip(hi, 0, nb - 1), 1)
                p_short = np.where(lo < 0, 0.0, np.where(lo >= nb, 1.0, g_lo))
                p_over = np.where(hi < 0, 1.0, np.where(hi >= nb, 0.0, 1.0 - g_hi))
                aj = act[:, :, j]
                self.Pn[:, j] += m.sum(0)
                for k, (c, p) in enumerate(((0, p_short), (2, p_over))):
                    hit = (aj == c) & m
                    self.Ppos[:, j, k] += hit.sum(0)
                    for ti, tau in enumerate(STOCK_TAUS):
                        al = (p >= tau) & m
                        self.P[:, j, k, ti, 0] += al.sum(0)
                        self.P[:, j, k, ti, 1] += (al & hit).sum(0)


def stock_scores(J):
    """J (3, 3, 3) = (지금, 실제, 예측) → 정확도, macro F1, 부족·과잉 정밀도·재현율, 새로 부족·과잉 (지금 정상) 재현율·정밀도·건수,
    상태가 바뀐 샘플만 본 정확도."""
    c = J.sum(0)
    acc, rec, pre, f1, mf1 = cm_scores(c)
    out_ = {'n': c.sum(), 'acc': acc, 'mf1': mf1, 'pre_s': pre[0], 'rec_s': rec[0], 'pre_o': pre[2], 'rec_o': rec[2]}
    for k, cl in (('s', 0), ('o', 2)):
        tp = J[1, cl, cl]
        out_[f'new_n_{k}'] = J[1, cl].sum()
        out_[f'new_rec_{k}'] = tp / J[1, cl].sum() if J[1, cl].sum() else np.nan
        out_[f'new_pre_{k}'] = tp / J[1, :, cl].sum() if J[1, :, cl].sum() else np.nan
    chg = sum(J[a, b].sum() for a in range(3) for b in range(3) if a != b)
    out_['chg_n'] = chg
    out_['chg_acc'] = sum(J[a, b, b] for a in range(3) for b in range(3) if a != b) / chg if chg else np.nan
    return out_


def evaluate(D, samples, predict, label, block=BATCH * 8, stock=False):
    s_all, t_all = samples
    E = Errors(D.S)
    if stock and D.stock_crits:
        E.sk = StockStats(D.stock_crits)
    ts = time.time()
    for b in range(0, len(s_all), block):
        s, t = s_all[b:b + block], t_all[b:b + block]
        x, st, gu, fut, y, fut_t = D.batch(s, t)
        pred = predict(D, s, t, x, st, gu, fut, fut_t)
        E.add(D, s, x, pred, y, fut_t, getattr(predict, 'probs', None), getattr(predict, 'dist', None))
        if E.sk is not None:
            E.sk.add(D, s, x, pred, y, fut_t, getattr(predict, 'dist', None))
        if (b // block) % 50 == 0:
            print(f'    [{label} 평가] {min(b + block, len(s_all)):,}/{len(s_all):,} ({time.time() - ts:.0f}초)',
                  end='\r', flush=True)
    print(' ' * 80, end='\r')
    return E, len(s_all)


def pred_lastweek(D, s, t, x, st, gu, fut, fut_t):
    lw = fut_t - WEEK
    return np.stack([D.rent[s[:, None], lw], D.ret[s[:, None], lw]], -1).astype(np.float32)


def pred_mean(D, s, t, x, st, gu, fut, fut_t):
    key = D.dow[fut_t] * 24 + D.hod[fut_t]
    return D.M[s[:, None], key]


def with_outage(predict, dur):
    """예측 전에 입력 창의 재고를 'dur시간째 수집 장애 중'으로 바꿔 주는 감싸기."""
    def p(D, s, t, x, st, gu, fut, fut_t):
        if dur > 0:
            x = D.outage(s, t, x, np.full(len(s), dur, np.int64))
        return predict(D, s, t, x, st, gu, fut, fut_t)
    return p


# ------------------------------------------------------------------ 6. 모델
def make_model(torch, nn, D, cfg):
    F = torch.nn.functional

    class FilmMixer(nn.Module):
        """피처 섞기 (FiLM): 대여소 블록 + 미래 칸 → (γ, β) → 과거 요약 h' = h·(1+γ) + β.
        마지막 층을 0으로 시작 → 처음엔 h 그대로 (섞기 없는 구조와 같은 출력에서 출발)"""

        def __init__(self, n_past, n_station, n_fh):
            super().__init__()
            self.film = nn.Sequential(nn.Linear(n_station + HORIZON * n_fh, FILM_HID), nn.ReLU(),
                                      nn.Linear(FILM_HID, 2 * n_past))
            with torch.no_grad():
                self.film[-1].weight.zero_()
                self.film[-1].bias.zero_()
            self.last_film = None                                  # 마지막 배치의 |γ|, |β| 평균 (보고용)

        def forward(self, h, sv, fut):
            """h (B, 64) 과거 요약, sv (B, 17) 대여소 블록, fut (B, 3 × 23) 미래 칸 (t+1, t+2, t+3 순서)."""
            g, b = self.film(torch.cat([sv, fut], -1)).chunk(2, -1)
            self.last_film = torch.stack([g.abs().mean(-1), b.abs().mean(-1)], -1).detach()
            return h * (1 + g) + b

    class Forecaster(nn.Module):
        """과거 창 24시간 → GRU → 마지막 은닉 h(t) → [FiLM: 대여소·미래 정보로 h 조절]
        → 이어 붙임 [h, 대여소 17, 미래 69] → 결합층 128 → 출력 3개 (순증감 회귀, 방향 분류, 순증감 분포)."""

        def __init__(self):
            super().__init__()
            hid, layers = cfg['hidden'], cfg['layers']
            self.rnn = nn.GRU(D.n_in, hid, num_layers=layers, batch_first=True, dropout=0.1 if layers > 1 else 0.0)
            self.emb = nn.Embedding(len(D.gu_names), EMB)
            n_sid = int(cfg.get('station_id_emb', 0) or 0)            # 대여소 ID 임베딩 (0이면 없음)
            self.sid = nn.Embedding(int(cfg['sid_n']) + 1, n_sid) if n_sid else None   # 마지막 칸 = 모르는 대여소
            if self.sid is not None:
                with torch.no_grad():
                    self.sid.weight.mul_(0.1)                          # 처음엔 작게 시작
            n_station = D.n_static + EMB + n_sid                       # 대여소 블록 (고정 속성 + 자치구 + 대여소 ID)
            self.mixer = FilmMixer(hid, n_station, D.n_future) if (cfg.get('mix') or 'none') == 'film' else None
            n_z = hid + n_station + HORIZON * D.n_future                # 64 + 17 + 69 = 150
            self.head = nn.Sequential(nn.Linear(n_z, FC), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(FC, HORIZON))
            self.cls_head = nn.Linear(FC, HORIZON * 3) if cfg.get('cls_head') else None   # 방향 분류 출력
            self.last_logits = None
            self.dist_head = nn.Linear(FC, HORIZON * NBIN) if cfg.get('dist_head') else None   # 순증감 분포 출력
            self.last_dist = None
            with torch.no_grad():                                      # 처음 출력 = 학습 기간 평균 순증감
                self.head[-1].bias.fill_(float(D.mean_ret - D.mean_rent))

        def forward(self, x, st, gu, fut):
            sv = torch.cat([st, self.emb(gu[:, 0])] + ([self.sid(gu[:, 1])] if self.sid is not None else []), -1)   # 대여소 블록 (B, 17)
            _, h = self.rnn(x)                                         # 시작 상태 0, h: (층, B, 은닉)
            past = h[-1]                                               # 과거 요약 h(t) (B, 64)
            if self.mixer is not None:                                 # 피처 섞기 (FiLM)
                past = self.mixer(past, sv, fut)
            z = torch.cat([past, sv, fut], -1)                         # 이어 붙임 (B, 150)
            hz = self.head[:-1](z)                                     # 결합층 128 (출력 3개가 같이 씀)
            # 순증감 n 하나만 냄. 평가 코드가 (대여, 반납) 모양을 받으므로 (relu(−n), relu(n))로 담음:
            # 반납 − 대여 = n 이 정확히 성립하고, 이 두 값을 대여·반납 예측으로 쓰지는 않음
            n = self.head[-1](hz).view(-1, HORIZON)
            yhat = torch.stack([F.relu(-n), F.relu(n)], -1)
            self.last_logits = self.cls_head(hz).view(-1, HORIZON, 3) if self.cls_head is not None else None
            self.last_dist = self.dist_head(hz).view(-1, HORIZON, NBIN) if self.dist_head is not None else None
            return yhat

    return Forecaster()


def check_supported(cfg, tag):
    """이 코드가 다루는 모델: GRU + FiLM(또는 섞기 없음), 순증감만, 이용 통계 없음, 학습 2022~2023.
    예전 실험 구조(GRU-D, LSTM, 어텐션, 눈 입력 등)는 실험 코드(gru_v15_mix.py)로 불러올 것."""
    bad = []
    if cfg.get('cell', 'gru') != 'gru':
        bad.append(f'셀 {cfg.get("cell")}')
    if cfg.get('attention'):
        bad.append('시간 어텐션')
    if (cfg.get('mix') or 'none') not in ('none', 'film'):
        bad.append(f'섞기 {cfg.get("mix")}')
    if cfg.get('mix_h0'):
        bad.append('GRU 시작 상태 = 대여소')
    if not cfg.get('net_only'):
        bad.append('대여·반납 따로 예측')
    if not cfg.get('no_stats'):
        bad.append('이용 통계 입력')
    for k, label in (('full_feats', '재고 포화도 입력'), ('mark_400', '400곳 표시'), ('snow_feats', '눈 입력'),
                     ('od_csv', '이동량 이웃')):
        if cfg.get(k):
            bad.append(label)
    if int(cfg.get('train_end', TRAIN_END0)) != TRAIN_END0:
        bad.append(f'학습 기간 ~{cfg.get("train_end")}')
    if bad:
        sys.exit(f'{tag}: 이 코드에 없는 설정({", ".join(bad)}). 예전 실험 코드(gru_v15_mix.py)로 불러올 것')


def pick_device(torch):
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.device('xpu')
    return torch.device('cpu')


def tensors(torch, dev, x, st, gu, fut, y=None):
    f = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(dev)
    return f(x), f(st), torch.from_numpy(np.ascontiguousarray(gu)).to(dev), f(fut), (f(y) if y is not None else None)


def make_predict(torch, model, dev):
    def predict(D, s, t, x, st, gu, fut, fut_t):
        model.eval()
        with torch.no_grad():
            xx, ss, gg, ff, _ = tensors(torch, dev, x, st, gu, fut)
            yhat = model(xx, ss, gg, ff).float().cpu().numpy()
            lg = getattr(model, 'last_logits', None)
            predict.probs = torch.softmax(lg, -1).float().cpu().numpy() if lg is not None else None
            ld = getattr(model, 'last_dist', None)
            predict.dist = torch.softmax(ld, -1).float().cpu().numpy() if ld is not None else None
            return yhat
    predict.probs = None
    predict.dist = None
    return predict


def mix_profile(torch, model, dev, D, samples, n=100_000):
    """검증 일부에서 FiLM 이 과거 요약을 얼마나 바꿨는지 평균. 그룹: 전체 / 출퇴근(t+1 이 7·8·17~19시) / 비(t+1 강수 > 0).
    반환: FiLM 합 {그룹: (|γ|, |β|)}, 샘플 수 {그룹}"""
    s_all, t_all = samples
    if len(s_all) > n:                                             # 샘플이 대여소 순서라 고르게 뽑음
        keep = np.sort(np.random.default_rng(SEED).choice(len(s_all), n, replace=False))
        s_all, t_all = s_all[keep], t_all[keep]
    groups = ('전체', '출퇴근', '비')
    F = {g: np.zeros(2) for g in groups}
    cnt = {g: 0 for g in groups}
    mx = model.mixer
    model.eval()
    with torch.no_grad():
        for b in range(0, len(s_all), BATCH * 4):
            s, t = s_all[b:b + BATCH * 4], t_all[b:b + BATCH * 4]
            x, st, gu, fut, _, fut_t = D.batch(s, t)
            xx, ss, gg, ff, _ = tensors(torch, dev, x, st, gu, fut)
            model(xx, ss, gg, ff)
            fg = mx.last_film.float().cpu().numpy()
            t1 = fut_t[:, 0]
            masks = {'전체': np.ones(len(s), bool), '출퇴근': np.isin(D.hod[t1], PEAK_HOURS), '비': D.rain_raw[t1] > 0}
            for g, m in masks.items():
                if m.any():
                    cnt[g] += int(m.sum())
                    F[g] += fg[m].sum(0)
    return F, cnt


def report_mix(folder, tag, F, cnt):
    """FiLM 이 한 일을 출력하고 gru_mix_<태그>.csv 로 저장."""
    out('')
    out('[피처 섞기] 검증 일부에서 본 FiLM 의 동작 (그룹: 전체 / 출퇴근 = t+1 이 7·8·17~19시 / 비 = t+1 강수 > 0)')
    out('  대여소·미래 칸이 과거 요약 64칸을 얼마나 바꾸는지. |γ| = 곱하는 조절(1+γ)의 크기, |β| = 더하는 이동의 크기 (0이면 안 바꿈)')
    rows = []
    for g in F:
        if cnt[g]:
            gm, bm = F[g] / cnt[g]
            out(f'    {g:<6} 샘플 {cnt[g]:>8,}  |γ| {gm:.3f}  |β| {bm:.3f}')
            rows.append({'그룹': g, '종류': 'FiLM', '받는 칸': '과거흐름', '샘플수': cnt[g], '|γ|': gm, '|β|': bm})
    if rows:
        df = pd.DataFrame(rows)
        num = df.select_dtypes('number').columns
        df[num] = df[num].round(4)
        df.to_csv(os.path.join(folder, f'gru_mix_{tag}.csv'), index=False, encoding='utf-8-sig')
        out(f'  (CSV: gru_mix_{tag}.csv)')


def train_gru(D, folder, cfg, epoch_samples):
    import torch
    import torch.nn as nn
    F = torch.nn.functional
    torch.manual_seed(cfg['seed'])
    dev = pick_device(torch)
    out(f'장치: {dev}')
    model = make_model(torch, nn, D, cfg).to(dev)
    out(f'매개변수 {sum(p.numel() for p in model.parameters()):,}개')
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=1) \
        if cfg['lr_decay'] else None

    def loss_fn(pred, y):                                          # 순증감 MSE (학습 기간 표준편차로 나눔)
        en = ((pred[..., 1] - pred[..., 0]) - (y[..., 1] - y[..., 0])) / D.sd_net
        return cfg['net_weight'] * (en ** 2).mean()

    use_cls = bool(cfg.get('cls_head'))
    use_sid = bool(cfg.get('station_id_emb'))
    mg = cfg.get('cls_margin', 1)

    def cls_loss(y):
        net = y[..., 1] - y[..., 0]
        target = (net >= mg).long() * 2 + ((net > -mg) & (net < mg)).long()   # 0 부족, 1 균형, 2 과잉
        return F.cross_entropy(model.last_logits.reshape(-1, 3), target.reshape(-1))

    use_dist = bool(cfg.get('dist_head'))

    def dist_loss(y):                                              # 순증감 17칸 교차 엔트로피
        net = y[..., 1] - y[..., 0]
        target = (torch.clamp(net, -DIST_K, DIST_K) + DIST_K).long()
        return F.cross_entropy(model.last_dist.reshape(-1, NBIN), target.reshape(-1))

    def total_loss(pred, y):
        reg = loss_fn(pred, y)
        tot, cl, dl = reg, None, None
        if use_cls:
            cl = cls_loss(y)
            tot = tot + cfg['cls_weight'] * cl
        if use_dist:
            dl = dist_loss(y)
            tot = tot + cfg['dist_weight'] * dl
        return tot, reg, cl, dl

    vs, vt = D.eval_samples('val', VAL_FRAC, EVAL_SEED)

    def val_loss():
        model.eval()
        tot, treg, tcl, tdl, n = 0.0, 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for b in range(0, len(vs), BATCH * 4):
                x, st, gu, fut, y, _ = D.batch(vs[b:b + BATCH * 4], vt[b:b + BATCH * 4])
                x, st, gu, fut, y = tensors(torch, dev, x, st, gu, fut, y)
                l, reg, cl, dl = total_loss(model(x, st, gu, fut), y)
                tot += l.item() * len(y)
                treg += reg.item() * len(y)
                tcl += (cl.item() if cl is not None else 0.0) * len(y)
                tdl += (dl.item() if dl is not None else 0.0) * len(y)
                n += len(y)
        n = max(n, 1)
        return tot / n, treg / n, tcl / n, tdl / n

    out(f'학습: 에폭마다 샘플 {epoch_samples:,}개, 최대 {cfg["epochs"]}에폭, 조기 종료용 검증 샘플 {len(vs):,}개')
    rng = np.random.default_rng(cfg['seed'])
    best, best_ep, wait = np.inf, 0, 0
    path = os.path.join(folder, f'gru_best_{cfg["tag"]}.pt')
    nb = (epoch_samples + BATCH - 1) // BATCH
    t_start = time.time()
    ep = 0
    for ep in range(1, cfg['epochs'] + 1):
        model.train()
        t0 = time.time()
        s_all, t_all = D.train_samples(epoch_samples, rng)
        tot = 0.0
        for k, b in enumerate(range(0, epoch_samples, BATCH)):
            sb, tb = s_all[b:b + BATCH], t_all[b:b + BATCH]
            x, st, gu, fut, y, ft = D.batch(sb, tb)
            if use_sid:                                            # 일부를 '모르는 대여소'로: 새 대여소용 벡터도 학습
                gu[rng.random(len(gu)) < SID_DROP, 1] = D.S
            x, st, gu, fut, y = tensors(torch, dev, x, st, gu, fut, y)
            opt.zero_grad()
            loss = total_loss(model(x, st, gu, fut), y)[0]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            if k % 200 == 0:
                print(f'    에폭 {ep} {k}/{nb} 배치, 학습 손실 {tot / (k + 1):.4f} ({time.time() - t0:.0f}초)',
                      end='\r', flush=True)
        v, v_reg, v_cls, v_dist = val_loss()
        if sched is not None:
            sched.step(v)
        mark = ''
        if v < best:
            best, best_ep, wait, mark = v, ep, 0, '  ← 저장'
            torch.save({'state': model.state_dict(), 'cfg': cfg}, path)
        else:
            wait += 1
        lr_now = opt.param_groups[0]['lr']
        parts = (f' (회귀 {v_reg:.4f}' + (f' + {cfg["cls_weight"]:g} × 분류 {v_cls:.4f}' if use_cls else '')
                 + (f' + {cfg["dist_weight"]:g} × 분포 {v_dist:.4f}' if use_dist else '') + ')') if (use_cls or use_dist) else ''
        out(f'  에폭 {ep:>2}  학습 손실 {tot / nb:.4f}  검증 손실 {v:.4f}{parts}  학습률 {lr_now:.0e}  ({time.time() - t0:.0f}초){mark}')
        if wait >= PATIENCE:
            out(f'  검증 손실이 {PATIENCE}에폭 동안 안 줄어 멈춤')
            break
    ckpt = torch.load(path, map_location=dev)
    model.load_state_dict(ckpt['state'])
    info = {'epochs': ep, 'best_epoch': best_ep, 'best_val': best, 'minutes': (time.time() - t_start) / 60}
    return model, dev, info


def load_gru(D, folder, tag):
    """저장된 모델을 불러오고, Data 입력 설정(속성표·평소 패턴·이웃 등)을 그 모델에 맞춤."""
    import torch
    import torch.nn as nn
    global L
    dev = pick_device(torch)
    ckpt = torch.load(os.path.join(folder, f'gru_best_{tag}.pt'), map_location=dev)
    cfg = ckpt['cfg']
    check_supported(cfg, tag)
    L = cfg['window']
    D.use_profile = cfg['profile']
    if cfg.get('station_emb') and D.emb_tab is None:
        sys.exit('속성표로 학습한 모델인데 station_embedding_YYYY.csv 를 찾지 못함')
    if cfg.get('station_emb') and D.emb_source != cfg.get('emb_source', 'team'):
        sys.exit(f'{tag}: 속성표 출처가 다름 (모델 {cfg.get("emb_source", "team")}, 지금 {D.emb_source})')
    D.use_emb = bool(cfg.get('station_emb'))
    if cfg.get('emb_no_use') and D.use_emb:
        D.drop_emb_use()
    D.use_past_profile = bool(cfg.get('past_profile'))
    if cfg.get('station_id_emb') and int(cfg.get('sid_n', -1)) != D.S:
        sys.exit(f'{tag}: 대여소 ID 임베딩을 만들 때와 대여소 수가 다름 ({cfg.get("sid_n")} vs {D.S}). 같은 데이터 캐시로 평가할 것')
    if cfg['neighbors']:
        if not (D.use_neighbors and getattr(D, 'nb_cfg', None) == cfg['nb_k']):
            D.set_neighbors(cfg['nb_k'])
            D.nb_cfg = cfg['nb_k']
    else:
        D.use_neighbors = False
    model = make_model(torch, nn, D, cfg).to(dev)
    model.load_state_dict(ckpt['state'])
    return model, dev, cfg


# ------------------------------------------------------------------ 7. 보고
def describe(cfg):
    """설정 한 줄 요약 (요약 CSV·실험 기록의 '설정' 열)."""
    nb = f'거리 {cfg["nb_k"]}곳' if cfg['neighbors'] else '없음'
    parts = [f'입력 {cfg["window"]}시간', '평소 패턴 ' + ('있음' if cfg['profile'] else '없음'), f'이웃 {nb}',
             f'GRU {cfg["layers"]}층 은닉 {cfg["hidden"]}', f'순증감 가중치 {cfg["net_weight"]:g}',
             'Huber' if cfg.get('huber') else 'MSE', '학습률 줄이기' if cfg['lr_decay'] else '학습률 고정',
             '인위 결측 ' + ('있음' if cfg.get('augment') else '없음'), f'시드 {cfg["seed"]}', '이용 통계 4개 입력에서 뺌']
    parts.append('대여소 속성표 ' + (('있음(코드 계산, 전 대여소)' if cfg.get('emb_source') == 'self' else '있음(팀 표)')
                                  if cfg.get('station_emb') else '없음'))
    if cfg.get('station_id_emb'):
        parts.append(f'대여소 ID 임베딩 {cfg["station_id_emb"]}차원')
    if cfg.get('stock_source', 'model') == 'file':
        sk = cfg.get('file_fill_skip') or []
        parts.append('재고 결측: 데이터 파일 보간값' + (f' (채움방식 {",".join(map(str, sk))}은 모델 규칙)' if sk else ''))
        if cfg.get('file_zero_rule', 'keep') == 'net':
            parts.append('이용 0건 시간 재고 = 직전 재고 + 순증감')
    else:
        parts.append('재고 결측: 모델 규칙')
    if cfg.get('emb_no_use'):
        parts.append('속성표 이용량 2개 뺌')
    if cfg.get('cls_head'):
        parts.append(f'방향 분류 출력 (경계 ±{cfg.get("cls_margin", 1)}, 가중치 {cfg.get("cls_weight", 0.5):g})')
    if cfg.get('past_profile'):
        parts.append('입력 창 평소 패턴')
    if cfg.get('dist_head'):
        parts.append(f'순증감 분포 출력 (−{DIST_K}~+{DIST_K}, 가중치 {cfg.get("dist_weight", 0.5):g})')
    parts.append('순증감만 예측 (이용량 뺌)')
    if cfg.get('exclude_hours'):
        parts.append(f'학습 제외 시각 {cfg["exclude_hours"]}')
    if (cfg.get('mix') or 'none') != 'none':
        parts.append('피처 섞기 ' + MIX_NAMES.get(cfg['mix'], cfg['mix']))
    return ', '.join(parts)


def report(D, split, results, folder, tag):
    title = {'val': '검증 2024', 'test': '테스트 2025'}[split]
    names = list(results)
    n = results[names[0]][1]
    n_st = int((results[names[0]][0].st[:, 0] > 0).sum())
    f3 = lambda a: ' / '.join(f'{v:.3f}' for v in a)
    out('')
    out('=' * 88)
    out(f'[{title}] 대여소 {n_st:,}곳, 샘플 {n:,}개 × 지평 3')
    out('(순증감만 예측한 모델: 이용량은 타깃이 아니라 기준선 포함 모든 표에서 뺌)')
    out(f'{"":14}{"순증감 MAE (t+1/t+2/t+3)":<28}{"순증감 RMSE":<24}')
    for name in names:
        E = results[name][0]
        out(f'{name:<14}{f3(Errors.mae(E.h)):<28}{f3(Errors.rmse(E.h)):<24}')
    out(f'{"R²":<14}{"순증감 (t+1/t+2/t+3)":<28}')
    for name in names:
        out(f'{name:<14}{f3(results[name][0].r2()):<28}')
    dnames = [nm for nm in names if results[nm][0].has_dist]
    if dnames:
        out('')
        out(f'[순증감 분포 출력] −{DIST_K}~+{DIST_K} 칸 확률. 중앙값은 MAE 에 맞는 점예측(회귀 출력은 평균 = RMSE 에 맞는 점예측)')
        out(f'{"":14}{"중앙값 MAE (t+1/t+2/t+3)":<28}{"80% 구간 적중률":<24}{"80% 구간 평균 폭(대)":<24}')
        for nm in dnames:
            E = results[nm][0]
            n = np.maximum(E.dmae[:, 0], 1)
            out(f'{nm:<14}{f3(E.dmae[:, 1] / n):<28}{f3(E.pi[:, 1] / n):<24}{f3(E.pi[:, 2] / n):<24}')
        out('  (80% 구간 = 분포의 10%~90% 칸. 적중률이 0.80 근처면 확률이 믿을 만함)')

    out('')
    out('[과잉·부족 분류] 실제 순증감 > 0 과잉, < 0 부족, 0 균형. 예측 순증감은 반올림해서 나눔 (t+1 / t+3)')
    out(f'{"":14}{"정확도":<16}{"부족 재현율":<16}{"부족 정밀도":<16}{"과잉 재현율":<16}{"과잉 정밀도":<16}')
    for name in names:
        E = results[name][0]
        a, b = E.cls(0), E.cls(HORIZON - 1)
        out(f'{name:<14}' + ''.join(f'{f"{x:.3f} / {y:.3f}":<16}' for x, y in zip(a, b)))

    main = names[-1]                                     # 모델 결과 (없으면 마지막 기준선)
    ref = '기준선2 평균'
    Em, Er = results[main][0], results[ref][0]
    report_confusion(results, names, main, folder, split, tag)
    report_direction(results, names, main, folder, split, tag)
    report_alert(results, main, folder, split, tag)
    report_stock(D, results, names, main, folder, split, tag)
    plot_confusion(results, main, folder, split, tag)

    out('')
    out(f'[피크 시간 평균 오차 = 예측 − 실제] 음수면 낮게 예측함 (①⑤에서 지적된 피크 과소예측 확인)')
    bm, br = Em.bias(), Er.bias()
    for hh in PEAK_HOURS:
        out(f'    {hh:>2}시  순증감 {bm[hh]:+.3f} (기준 {br[hh]:+.3f})')

    out('')
    out(f'[나눠 보기] {main} 과 {ref} 의 MAE (지평 3개 합쳐서)')
    for label, a, b, keys in (('비 여부 (목표 시각)', Em.rain, Er.rain, ['비 안 옴', '비 옴']),
                              ('재고 결측이 입력 창에', Em.miss, Er.miss, ['없음', '있음'])):
        out(f'  {label}')
        for j, k in enumerate(keys):
            out(f'    {k:<8} 샘플 {int(a[j, 0]):>11,}  순증감 {Errors.mae(a)[j]:.3f} (기준 {Errors.mae(b)[j]:.3f})')

    hm, hr = Errors.mae(Em.hour), Errors.mae(Er.hour)
    out('  시간대별 (목표 시각 기준)')
    for hh in range(24):
        out(f'    {hh:>2}시  순증감 {hm[hh]:.3f} (기준 {hr[hh]:.3f})')
    by_hour = {'시간대': np.arange(24),
               **{f'{nm}_순증감_MAE': Errors.mae(results[nm][0].hour) for nm in names},
               **{f'{nm}_순증감_평균오차': results[nm][0].bias() for nm in names}}
    pd.DataFrame(by_hour).to_csv(os.path.join(folder, f'gru_{split}_by_hour_{tag}.csv'), index=False, encoding='utf-8-sig')

    sm, sr = Errors.mae(Em.st), Errors.mae(Er.st)
    cols = {'대여소번호': D.stations, '학습기간_평균이용량': np.round(D.mean_use_station, 3),
            '학습기간_기록': np.where(D.new_station, '없음', '있음'),
            '평가샘플수': (Em.st[:, 0] / HORIZON).astype(int),
            f'{main}_순증감_MAE': sm, f'{ref}_순증감_MAE': sr}
    df = pd.DataFrame(cols)
    if D.report_grp is not None:
        df.insert(3, '속성표', np.where(D.report_grp, '있음', '없음'))      # 팀 속성표(400곳)에 있는 대여소
    df = df[df['평가샘플수'] > 0]
    worse = int((df[f'{main}_순증감_MAE'] > df[f'{ref}_순증감_MAE']).sum())
    df = df.sort_values(f'{main}_순증감_MAE', ascending=False)
    df.to_csv(os.path.join(folder, f'gru_{split}_by_station_{tag}.csv'), index=False, encoding='utf-8-sig')
    out(f'  순증감이 기준선2보다 나쁜 대여소: {worse:,}곳 / {len(df):,}곳')
    groups = []
    if D.new_station.any():
        groups.append(('학습 기간 기록 여부 (새 대여소의 기준선2는 기록 있는 대여소 전체 평균)',
                       (('기록 있음', ~D.new_station), ('새 대여소', D.new_station))))
    if D.report_grp is not None:
        src = ({'team': '팀 표', 'self': '코드 계산 표(전 대여소)'}.get(D.emb_source, '') if D.use_emb else '안 씀')
        groups.append((f'팀 속성표 400곳 / 나머지 (입력 속성표: {src})', (('표 있음', D.report_grp), ('표 없음', ~D.report_grp))))
    for title_, parts in groups:
        out(f'  {title_}')
        for label, g in parts:
            a, b = Em.st[g].sum(0), Er.st[g].sum(0)
            if a[0] == 0:
                continue
            n_g = int(((Em.st[:, 0] > 0) & g).sum())
            out(f'    {label:<8} {n_g:>5,}곳  샘플 {int(a[0] / HORIZON):>11,}  순증감 {a[1] / a[0]:.3f} (기준 {b[1] / b[0]:.3f})')
    out(f'  순증감 오차가 가장 큰 대여소 10곳 (전체는 gru_{split}_by_station_{tag}.csv)')
    for _, r in df.head(10).iterrows():
        out(f'    {r["대여소번호"]:>6}  학습기간 평균 이용량 {r["학습기간_평균이용량"]:6.2f}  순증감 MAE {r[f"{main}_순증감_MAE"]:.3f}'
            f' (기준 {r[f"{ref}_순증감_MAE"]:.3f})')


def report_confusion(results, names, main, folder, split, tag):
    """Confusion Matrix(실제 × 예측, 부족·균형·과잉)와 오차 분포를 출력하고 CSV로 저장."""
    E = results[main][0]
    pct = lambda a, b: f'{a / b * 100:5.1f}%' if b > 0 else '   - '
    for j in range(HORIZON):
        c = E.cm[j]
        n = c.sum()
        out('')
        out(f'[Confusion Matrix] {main}, t+{j + 1}  (행 = 실제, 열 = 예측. 괄호는 그 실제 클래스 안에서의 비율)')
        out(f'  {"":10}' + ''.join(f'{"예측 " + k:>24}' for k in CLS_NAMES) + f'{"합계 (비율)":>24}')
        for i, k in enumerate(CLS_NAMES):
            row = c[i]
            out(f'  {"실제 " + k:<10}' + ''.join(f'{int(v):>15,} ({pct(v, row.sum())})' for v in row)
                + f'{int(row.sum()):>15,} ({pct(row.sum(), n)})')
        col = c.sum(0)
        out(f'  {"예측 합계":<10}' + ''.join(f'{int(v):>15,} ({pct(v, n)})' for v in col) + f'{int(n):>15,}')
        m3, mf1 = E.cls3(j)
        out(f'  {"클래스":<10}{"재현율":>10}{"정밀도":>10}{"F1":>10}     (정확도 {np.trace(c) / max(n, 1):.3f}, macro F1 {mf1:.3f})')
        for i, k in enumerate(CLS_NAMES):
            out(f'  {k:<10}{m3[i, 0]:>10.3f}{m3[i, 1]:>10.3f}{m3[i, 2]:>10.3f}')
    rows = []
    for nm in names:
        Ex = results[nm][0]
        for j in range(HORIZON):
            m3, mf1 = Ex.cls3(j)
            for i, k in enumerate(CLS_NAMES):
                rows.append({'모델': nm, '지평': f't+{j + 1}', '실제': k,
                             **{f'예측_{p}': int(Ex.cm[j, i, q]) for q, p in enumerate(CLS_NAMES)},
                             '합계': int(Ex.cm[j, i].sum()), '재현율': round(m3[i, 0], 4), '정밀도': round(m3[i, 1], 4),
                             'F1': round(m3[i, 2], 4), '정확도': round(np.trace(Ex.cm[j]) / max(Ex.cm[j].sum(), 1), 4),
                             'macroF1': round(mf1, 4)})
    pd.DataFrame(rows).to_csv(os.path.join(folder, f'gru_{split}_confusion_{tag}.csv'), index=False, encoding='utf-8-sig')

    out('')
    out(f'[오차 분포] |예측 − 실제| 가 기준 이하인 샘플 비율 (지평 3개 합쳐서)')
    out(f'  {"":14}{"":8}' + ''.join(f'{"≤ " + format(b, "g"):>9}' for b in ERR_BINS))
    rows = []
    for nm in names:
        Ex = results[nm][0]
        share = Ex.ebin.sum(0) / max(Ex.h[:, 0].sum(), 1)
        out(f'  {nm:<14}{"순증감":<8}' + ''.join(f'{v * 100:8.1f}%' for v in share))
        for j in range(HORIZON):
            sj = Ex.ebin[j] / max(Ex.h[j, 0], 1)
            rows.append({'모델': nm, '타겟': '순증감', '지평': f't+{j + 1}',
                         **{f'오차≤{b:g}_비율': round(v, 4) for b, v in zip(ERR_BINS, sj)}})
    pd.DataFrame(rows).to_csv(os.path.join(folder, f'gru_{split}_error_dist_{tag}.csv'), index=False, encoding='utf-8-sig')
    out(f'  (CSV: gru_{split}_confusion_{tag}.csv, gru_{split}_error_dist_{tag}.csv)')


def _print_cm(title, c):
    """3×3 Confusion Matrix와 클래스별 재현율·정밀도·F1 출력. 반환: (정확도, macro F1, 클래스별 [재현율, 정밀도, F1])."""
    n = c.sum()
    pct = lambda a, b: f'{a / b * 100:5.1f}%' if b > 0 else '   - '
    out('')
    out(title)
    out(f'  {"":10}' + ''.join(f'{"예측 " + k:>24}' for k in CLS_NAMES) + f'{"합계 (비율)":>24}')
    for i, k in enumerate(CLS_NAMES):
        row = c[i]
        out(f'  {"실제 " + k:<10}' + ''.join(f'{int(v):>15,} ({pct(v, row.sum())})' for v in row)
            + f'{int(row.sum()):>15,} ({pct(row.sum(), n)})')
    rec = np.divide(np.diag(c), c.sum(1), out=np.full(3, np.nan), where=c.sum(1) > 0)
    pre = np.divide(np.diag(c), c.sum(0), out=np.full(3, np.nan), where=c.sum(0) > 0)
    f1 = np.divide(2 * rec * pre, rec + pre, out=np.full(3, np.nan), where=(rec + pre) > 0)
    acc, mf1 = np.trace(c) / max(n, 1), float(np.nanmean(f1))
    out(f'  {"클래스":<10}{"재현율":>10}{"정밀도":>10}{"F1":>10}     (정확도 {acc:.3f}, macro F1 {mf1:.3f})')
    for i, k in enumerate(CLS_NAMES):
        out(f'  {k:<10}{rec[i]:>10.3f}{pre[i]:>10.3f}{f1[i]:>10.3f}')
    return acc, mf1, np.stack([rec, pre, f1], 1)


def report_direction(results, names, main, folder, split, tag):
    """분류 출력(--cls-head)으로 본 방향과, 같은 클래스 경계에서 회귀 출력으로 본 방향, 큰 변화 방향."""
    E = results[main][0]
    mg = getattr(report_direction, 'margin', 1)
    rows = []
    if E.has_cls or E.has_dist or mg != 1:
        out('')
        out(f'[방향: 클래스 경계 ±{mg}] 순증감 ≤ −{mg} 부족, ≥ +{mg} 과잉, 그 사이 균형')
        out(f'  {"":24}' + ''.join(f'{"t+" + str(j + 1) + " 정확도":>14}' for j in range(HORIZON))
            + f'{"부족 재현율/정밀도 (t+1)":>28}{"과잉 재현율/정밀도 (t+1)":>28}')
        srcs = ([('회귀 출력 (반올림)', E.cmm)] + ([('분류 출력', E.cmc)] if E.has_cls else [])
                + ([('분포 출력', E.pstat['dist']['cm'])] if E.has_dist else []))
        for label, c in srcs:
            acc = [np.trace(c[j]) / max(c[j].sum(), 1) for j in range(HORIZON)]
            c0 = c[0]
            r = lambda i: c0[i, i] / max(c0[i].sum(), 1)
            p = lambda i: c0[i, i] / max(c0[:, i].sum(), 1)
            out(f'  {main + " " + label:<24}' + ''.join(f'{a:>14.3f}' for a in acc)
                + f'{f"{r(0):.3f} / {p(0):.3f}":>28}{f"{r(2):.3f} / {p(2):.3f}":>28}')
        if E.has_cls:
            for j in range(HORIZON):
                acc, mf1, m3 = _print_cm(f'[Confusion Matrix: 분류 출력] {main}, t+{j + 1}, 경계 ±{mg}  (행 = 실제, 열 = 예측)', E.cmc[j])
                for i, k in enumerate(CLS_NAMES):
                    rows.append({'모델': f'{main} 분류 출력', '지평': f't+{j + 1}', '경계': mg, '실제': k,
                                 **{f'예측_{q}': int(E.cmc[j, i, a]) for a, q in enumerate(CLS_NAMES)},
                                 '재현율': round(m3[i, 0], 4), '정밀도': round(m3[i, 1], 4), 'F1': round(m3[i, 2], 4),
                                 '정확도': round(acc, 4), 'macroF1': round(mf1, 4)})
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(folder, f'gru_{split}_confusion_cls_{tag}.csv'),
                                      index=False, encoding='utf-8-sig')
    out('')
    out('[큰 변화 방향] 실제 순증감이 −3 이하(빠짐) / +3 이상(쌓임)인 시간 중 같은 방향으로 예측한 비율 (t+1 / t+3)')
    out(f'  {"":24}{"빠짐 샘플 (t+1)":>16}{"빠짐 맞춤":>18}{"쌓임 샘플 (t+1)":>16}{"쌓임 맞춤":>18}')
    for nm in names:
        Ex = results[nm][0]
        for label, col in [('', 1)] + ([(' 분류 출력', 2)] if Ex.has_cls else []):
            b = Ex.big
            f = lambda j, k: b[j, k, col] / max(b[j, k, 0], 1)
            out(f'  {nm + label:<24}{int(b[0, 0, 0]):>16,}{f"{f(0, 0):.3f} / {f(2, 0):.3f}":>18}'
                f'{int(b[0, 1, 0]):>16,}{f"{f(0, 1):.3f} / {f(2, 1):.3f}":>18}')


def report_alert(results, main, folder, split, tag):
    """같은 방향 확률을 판정 규칙에 따라 다시 셈 (t+1).
    가장 큰 확률 = 지금 방식(정확도가 가장 높게 나오는 규칙), 절반 규칙 = 부족·과잉 확률이 0.5 이상일 때만
    (순서가 있는 세 클래스에서 '틀린 칸 수'를 가장 작게 하는 규칙이라 반대 방향 오답이 줄어듦),
    경보 기준 = 그 확률 이상일 때만 경보 (정밀도 ↑, 재현율 ↓), 확신이 큰 예측 = 가장 큰 확률 상위 x% 만 볼 때 정확도."""
    E = results[main][0]
    if not E.pstat:
        return
    mg = getattr(report_direction, 'margin', 1)
    j = 0
    N = max(E.nact[j].sum(), 1)
    rows = []
    out('')
    out(f'[경보 기준·규칙별 방향 성능] t+1, 경계 ±{mg}. 같은 확률을 어떤 규칙으로 판정하느냐에 따라 정확도·정밀도·재현율이 바뀜')
    for key, label in (('cls', '분류 출력'), ('dist', '분포 출력')):
        st = E.pstat.get(key)
        if st is None:
            continue
        out(f'  {main} {label}')
        out(f'    {"규칙":<22}{"정확도":>8}{"macro F1":>10}{"부족 정밀도/재현율":>20}{"과잉 정밀도/재현율":>20}{"반대 방향 오답":>14}')
        for rule, c in (('가장 큰 확률 (지금 방식)', st['cm'][j]), ('절반 넘을 때만 (P ≥ 0.5)', st['med'][j])):
            acc, rec, pre, f1, mf1 = cm_scores(c)
            opp = (c[0, 2] + c[2, 0]) / max(c[0].sum() + c[2].sum(), 1)
            out(f'    {rule:<22}{acc:>8.3f}{mf1:>10.3f}{f"{pre[0]:.3f} / {rec[0]:.3f}":>20}'
                f'{f"{pre[2]:.3f} / {rec[2]:.3f}":>20}{opp:>14.1%}')
            rows.append({'출력': label, '구분': '규칙', '규칙·기준': rule, '정확도': acc, 'macroF1': mf1,
                         '부족_정밀도': pre[0], '부족_재현율': rec[0], '과잉_정밀도': pre[2], '과잉_재현율': rec[2],
                         '반대방향오답': opp})
        out(f'    경보 확률 기준 (그 확률 이상일 때만 경보)')
        out(f'    {"기준":>6}{"부족 정밀도":>12}{"부족 재현율":>12}{"부족 경보 비율":>14}'
            f'{"과잉 정밀도":>12}{"과잉 재현율":>12}{"과잉 경보 비율":>14}')
        for ti, tau in enumerate(ALERT_TAUS):
            vals, row = [], {'출력': label, '구분': '경보 기준', '규칙·기준': tau}
            for ci, (c, cname) in enumerate(((0, '부족'), (2, '과잉'))):
                npred, tp = st['alert'][j, ci, ti]
                p = tp / npred if npred else np.nan
                r = tp / max(E.nact[j, c], 1)
                vals += [p, r, npred / N]
                row.update({f'{cname}_정밀도': p, f'{cname}_재현율': r, f'{cname}_경보비율': npred / N})
            out(f'    {tau:>6.1f}' + ''.join(f'{v:>12.3f}{w:>12.3f}{z:>14.1%}' for v, w, z in (vals[:3], vals[3:])))
            rows.append(row)
        cf = st['conf'][j][::-1]                                   # 확신이 큰 구간부터
        n_cum, ok_cum = cf[:, 0].cumsum(), cf[:, 1].cumsum()
        tot = max(n_cum[-1], 1)
        parts = []
        for cov in COVERAGES:
            k = min(int(np.searchsorted(n_cum, cov * tot - 1e-9)), len(n_cum) - 1)
            acc_k = ok_cum[k] / max(n_cum[k], 1)
            parts.append(f'상위 {n_cum[k] / tot:.0%} {acc_k:.3f}')
            rows.append({'출력': label, '구분': '확신 상위', '규칙·기준': round(n_cum[k] / tot, 3), '정확도': acc_k})
        out('    확신이 큰 예측만 볼 때 정확도 (가장 큰 확률 순): ' + ', '.join(parts))
    if rows:
        df = pd.DataFrame(rows)
        num = df.select_dtypes('number').columns
        df[num] = df[num].round(4)
        df.to_csv(os.path.join(folder, f'gru_{split}_alert_{tag}.csv'), index=False, encoding='utf-8-sig')
        out(f'  (CSV: gru_{split}_alert_{tag}.csv)')


def report_stock(D, results, names, main, folder, split, tag):
    """--stock-eval: 대시보드 재고 상태 기준 결과. 첫 기준은 자세히, 나머지 기준은 요약만."""
    E = results[main][0]
    sk = E.sk
    if sk is None:
        return
    f3_ = lambda v: '   -  ' if v is None or not np.isfinite(v) else f'{v:.3f}'
    pc = lambda v: '   -  ' if not np.isfinite(v) else f'{v:.1%}'
    b2 = results.get('기준선2 평균', (None,))[0]
    rows_all = []

    def row_sets(ci, j, real=False):
        src = 'Jr' if real else 'J'
        sets = [('지금 상태 유지', getattr(sk, src)['pers'][ci, j])]
        if b2 is not None and b2.sk is not None:
            sets.append(('기준선2 평균', getattr(b2.sk, src)['pred'][ci, j]))
        sets.append((main, getattr(sk, src)['pred'][ci, j]))
        sets.append(('순증감 완벽 예측 (상한)', getattr(sk, src)['oracle'][ci, j]))
        return sets

    out('')
    name0, a0, mn0, b0 = sk.crits[0]
    out(f'[재고 상태 평가: 대시보드 기준 {name0}] 부족 = 재고 ≤ ' + (f'max({mn0:g}대, 거치대 × {a0:g})' if mn0 > 0 else f'거치대 × {a0:g}')
        + f', 과잉 = 재고 ≥ 거치대 × {b0:g}, 그 사이 정상')
    out(f'  예측 시점 = t시 이용이 끝난 (t+1)시 정각. 예측 재고(t+k시 이용 뒤) = 지금 재고 + 예측 순증감 합(t+1~t+k) (반올림). '
        f'거치대수를 모르는 대여소 {int((~D.docks_ok).sum()):,}곳은 뺌.')
    out(f'  현재·미래 재고가 수집 장애가 아닌 시간만 셈 (실측, 또는 이용 없는 시간의 직전 재고 + 순증감). 거치대수는 최신 연도 값')
    tc = sk.tcheck
    if tc[0] > 0:
        out(f'  재고 시점 점검 (이어진 두 실측 시간 {int(tc[0]):,}쌍, 재고 차이 = 라벨(t+1) − 라벨(t)): '
            f'순증감(t) 와 같음 {tc[1] / tc[0]:.1%} (평균 차 {tc[2] / tc[0]:.2f}대), '
            f'순증감(t+1) 과 같음 {tc[3] / tc[0]:.1%} (평균 차 {tc[4] / tc[0]:.2f}대)')
        out('    → 앞쪽이 높아야 정각 기준이 맞음 (이 평가와 재고 채우기가 쓰는 기준). 뒤쪽이 높으면 재고가 그 시간 끝 값이라 기준을 바꿔야 함')
    c1 = sk.J['pers'][0, 0].sum((0, 2))
    out(f'  실제 상태 비율 (t+1, {int(c1.sum()):,}개): 부족 {c1[0] / max(c1.sum(), 1):.1%}, 정상 {c1[1] / max(c1.sum(), 1):.1%}, '
        f'과잉 {c1[2] / max(c1.sum(), 1):.1%}')
    out('  1) 상태 정확도 (지금 상태 유지는 예측 없이도 맞는 몫, 완벽 예측은 재배치 등 순증감 밖 변화 때문에 남는 한계)')
    out(f'    {"":26}{"정확도 t+1/t+2/t+3":>24}{"macro F1 t+3":>14}{"부족 정밀도/재현율 t+3":>24}{"과잉 정밀도/재현율 t+3":>24}')
    for label, _ in row_sets(0, 0):
        sc = [stock_scores(dict(row_sets(0, j))[label]) for j in range(HORIZON)]
        out(f'    {label:<26}{" / ".join(f3_(s_["acc"]) for s_ in sc):>24}{f3_(sc[2]["mf1"]):>14}'
            f'{f3_(sc[2]["pre_s"]) + " / " + f3_(sc[2]["rec_s"]):>24}{f3_(sc[2]["pre_o"]) + " / " + f3_(sc[2]["rec_o"]):>24}')
        for j in range(HORIZON):
            rows_all.append({'기준': name0, '구분': '상태', '모델': label, '지평': f't+{j + 1}',
                             **{k: v for k, v in sc[j].items()}})
    out('  2) 지금 정상인 대여소가 t+k 에 새로 부족·과잉이 되는 경우 (모델 실력이 드러나는 부분, 지금 상태 유지는 늘 0)')
    out(f'    {"":26}{"새 부족 재현율/정밀도 t+1":>26}{"t+3":>18}{"새 과잉 재현율/정밀도 t+1":>26}{"t+3":>18}')
    for label, _ in row_sets(0, 0):
        sc1, sc3 = stock_scores(dict(row_sets(0, 0))[label]), stock_scores(dict(row_sets(0, 2))[label])
        out(f'    {label:<26}{f3_(sc1["new_rec_s"]) + " / " + f3_(sc1["new_pre_s"]):>26}'
            f'{f3_(sc3["new_rec_s"]) + " / " + f3_(sc3["new_pre_s"]):>18}'
            f'{f3_(sc1["new_rec_o"]) + " / " + f3_(sc1["new_pre_o"]):>26}{f3_(sc3["new_rec_o"]) + " / " + f3_(sc3["new_pre_o"]):>18}')
    s1, s3 = stock_scores(sk.J['pers'][0, 0]), stock_scores(sk.J['pers'][0, 2])
    out(f'    (건수: 새 부족 t+1 {int(s1["new_n_s"]):,} / t+3 {int(s3["new_n_s"]):,}, 새 과잉 t+1 {int(s1["new_n_o"]):,} / t+3 {int(s3["new_n_o"]):,})')
    out('  3) 3시간 안에 한 번이라도 부족·과잉 (지금 정상인 대여소, 대시보드 경보와 같은 판정)')
    out(f'    {"":26}{"부족 정밀도/재현율":>22}{"부족 경보 비율":>14}{"과잉 정밀도/재현율":>22}{"과잉 경보 비율":>14}')
    wsets = [('지금 상태 유지', sk.W['pers'][0])]
    if b2 is not None and b2.sk is not None:
        wsets.append(('기준선2 평균', b2.sk.W['pred'][0]))
    wsets += [(main, sk.W['pred'][0]), ('순증감 완벽 예측 (상한)', sk.W['oracle'][0])]
    for label, Wm in wsets:
        cells, row = [], {'기준': name0, '구분': '3시간 창', '모델': label}
        for k, cname in enumerate(('부족', '과잉')):
            m_ = Wm[k]
            tp, npred, npos, n = m_[1, 1], m_[:, 1].sum(), m_[1].sum(), m_.sum()
            p_, r_ = (tp / npred if npred else np.nan), (tp / npos if npos else np.nan)
            cells.append(f'{f3_(p_) + " / " + f3_(r_):>22}{pc(npred / n if n else np.nan):>14}')
            row.update({f'{cname}_정밀도': p_, f'{cname}_재현율': r_, f'{cname}_경보비율': npred / n if n else np.nan,
                        f'{cname}_실제비율': npos / n if n else np.nan})
        out(f'    {label:<26}' + ''.join(cells))
        rows_all.append(row)
    m0 = sk.W['pers'][0]
    out(f'    (지금 정상 {int(m0[0].sum()):,}개 중 3시간 안에 실제 부족 {m0[0][1].sum() / max(m0[0].sum(), 1):.1%}, '
        f'과잉 {m0[1][1].sum() / max(m0[1].sum(), 1):.1%})')
    if sk.has_dist:
        out('  4) 확률 경보 (분포 출력, 지금 정상인 대여소): t+k 에 기준을 넘을 확률이 τ 이상이면 경보')
        out('     누적 확률은 시간마다의 분포를 서로 독립으로 보고 더한 근사. 3시간 누적 순증감 80% 구간 적중률 '
            f'{sk.cov[1] / max(sk.cov[0], 1):.3f} (0.80 근처면 근사가 맞음)')
        out(f'    {"τ":>6}{"부족 정밀도/재현율 t+1":>26}{"경보 비율":>10}{"t+3":>18}{"경보 비율":>10}'
            f'{"과잉 정밀도/재현율 t+1":>26}{"경보 비율":>10}{"t+3":>18}{"경보 비율":>10}')
        for ti, tau in enumerate(STOCK_TAUS):
            cells, row = [], {'기준': name0, '구분': '확률 경보', '모델': main, '규칙': tau}
            for k, cname in enumerate(('부족', '과잉')):
                for j in (0, 2):
                    na, tp = sk.P[0, j, k, ti]
                    p_ = tp / na if na else np.nan
                    r_ = tp / sk.Ppos[0, j, k] if sk.Ppos[0, j, k] else np.nan
                    ar = na / sk.Pn[0, j] if sk.Pn[0, j] else np.nan
                    cells.append(f'{f3_(p_) + " / " + f3_(r_):>{26 if j == 0 else 18}}{pc(ar):>10}')
                    row.update({f'{cname}_정밀도_t{j + 1}': p_, f'{cname}_재현율_t{j + 1}': r_, f'{cname}_경보비율_t{j + 1}': ar})
            out(f'    {tau:>6.1f}' + ''.join(cells))
            rows_all.append(row)
    out('  5) 미래 재고가 실측인 시간만 (그 시간에 이용 기록이 있음): 상태 정확도 t+1 / t+3 (완벽 예측과의 차 = 재배치 등)')
    for label, _ in row_sets(0, 0, real=True):
        sc1 = stock_scores(dict(row_sets(0, 0, True))[label])
        sc3 = stock_scores(dict(row_sets(0, 2, True))[label])
        out(f'    {label:<26}{f3_(sc1["acc"])} / {f3_(sc3["acc"])}   ({int(sc3["n"]):,}개)')
    if len(sk.crits) > 1:
        out('  6) 다른 기준 요약 (t+3): 상태 정확도 [유지 / 모델], 새 부족 재현율/정밀도, 3시간 부족 정밀도/재현율')
        for ci, (nm, a, mn, b) in enumerate(sk.crits[1:], 1):
            sp, sm = stock_scores(sk.J['pers'][ci, 2]), stock_scores(sk.J['pred'][ci, 2])
            m_ = sk.W['pred'][ci][0]
            tp, npred, npos = m_[1, 1], m_[:, 1].sum(), m_[1].sum()
            out(f'    {nm:<22} {f3_(sp["acc"])} / {f3_(sm["acc"])}   {f3_(sm["new_rec_s"])} / {f3_(sm["new_pre_s"])}   '
                f'{f3_(tp / npred if npred else np.nan)} / {f3_(tp / npos if npos else np.nan)}')
            rows_all.append({'기준': nm, '구분': '요약', '모델': main, '지평': 't+3', **sm})
    df = pd.DataFrame(rows_all)
    num = df.select_dtypes('number').columns
    df[num] = df[num].round(4)
    df.to_csv(os.path.join(folder, f'gru_{split}_stock_{tag}.csv'), index=False, encoding='utf-8-sig')
    out(f'  (CSV: gru_{split}_stock_{tag}.csv)')


def cm_scores(c):
    """3×3 행렬(행 = 실제, 열 = 예측) → 정확도, 재현율(3), 정밀도(3), F1(3), macro F1."""
    c = np.asarray(c, dtype=np.float64)
    rec = np.divide(np.diag(c), c.sum(1), out=np.full(3, np.nan), where=c.sum(1) > 0)
    pre = np.divide(np.diag(c), c.sum(0), out=np.full(3, np.nan), where=c.sum(0) > 0)
    f1 = np.divide(2 * rec * pre, rec + pre, out=np.full(3, np.nan), where=(rec + pre) > 0)
    acc = float(np.trace(c) / c.sum()) if c.sum() > 0 else np.nan
    return acc, rec, pre, f1, float(np.nanmean(f1)) if np.isfinite(f1).any() else np.nan


def setup_korean_font(plt):
    """그림의 한글이 깨지지 않게 설치된 한글 글꼴을 고름 (윈도우는 맑은 고딕)."""
    from matplotlib import font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    for nm in ('Malgun Gothic', 'AppleGothic', 'NanumGothic', 'NanumBarunGothic', 'Noto Sans CJK KR', 'Noto Sans KR'):
        if nm in have:
            plt.rcParams['font.family'] = nm
            break
    else:
        out('  ※ 한글 글꼴을 찾지 못해 그림의 한글이 네모로 보일 수 있음')
    plt.rcParams['axes.unicode_minus'] = False


def draw_cm(ax, c, title, ylabel=None):
    """히트맵 한 칸: 색 = 행(실제 클래스) 안 비율, 글자 = 비율과 샘플 수."""
    c = np.asarray(c, dtype=np.float64)
    rs = c.sum(1, keepdims=True)
    p = np.divide(c, rs, out=np.zeros_like(c), where=rs > 0)
    ax.imshow(p, cmap='Blues', vmin=0, vmax=1)
    for i in range(3):
        for k in range(3):
            ax.text(k, i, f'{p[i, k] * 100:.1f}%\n({int(round(c[i, k])):,})', ha='center', va='center',
                    fontsize=8.5, color='white' if p[i, k] > 0.55 else 'black')
    ax.set_xticks(range(3))
    ax.set_xticklabels(CLS_NAMES)
    ax.set_yticks(range(3))
    ax.set_yticklabels(CLS_NAMES)
    ax.set_xlabel('예측')
    ax.set_ylabel(ylabel or '실제')
    acc, _, _, _, mf1 = cm_scores(c)
    ax.set_title(f'{title}\n정확도 {acc:.3f} · macro F1 {mf1:.3f}', fontsize=9.5)


def plot_confusion(results, main, folder, split, tag):
    """오차 매트릭스 히트맵 PNG. 행 = 기준선2 / 회귀 출력(반올림) / 분류 출력, 열 = t+1~t+3."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        out('  matplotlib 이 없어 오차 매트릭스 그림은 건너뜀 (pip install matplotlib)')
        return
    try:
        setup_korean_font(plt)
        E = results[main][0]
        mg = getattr(report_direction, 'margin', 1)
        ref = '기준선2 평균'
        rows = []
        if ref in results and ref != main:
            Er = results[ref][0]
            rows.append((f'{ref} (반올림)', Er.cm if mg == 1 else Er.cmm))
        rows.append((f'{main} 회귀 출력 (반올림)', E.cm if mg == 1 else E.cmm))
        if E.has_cls:
            rows.append((f'{main} 분류 출력', E.cmc))
        nr = len(rows)
        fig, axes = plt.subplots(nr, HORIZON, figsize=(4.3 * HORIZON, 3.9 * nr), squeeze=False)
        for r, (label, cms) in enumerate(rows):
            for j in range(HORIZON):
                draw_cm(axes[r, j], cms[j], f'{label}, t+{j + 1}')
        title = {'val': '검증 2024', 'test': '테스트 2025'}.get(split, split)
        fig.suptitle(f'오차 매트릭스 [{tag}] {title}  (부족: 순증감 ≤ -{mg}, 과잉: ≥ +{mg}, 행 = 실제 클래스 안 비율)',
                     fontsize=11)
        fig.tight_layout()
        path = os.path.join(folder, f'gru_{split}_confusion_{tag}.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        out(f'  (그림: {os.path.basename(path)}  칸 색·위 숫자 = 그 실제 클래스 안 비율, 괄호 = 샘플 수)')
    except Exception as e:                                   # 그림 때문에 결과 저장이 멈추지 않게
        out(f'  ※ 오차 매트릭스 그림을 만들지 못함: {e}')


def save_summary(D, results, folder, split, tag, cfg, info):
    """모델별 주요 지표 한 줄씩 → gru_<평가>_summary_<태그>.csv (compare_runs.py 가 읽음)."""
    mg = getattr(report_direction, 'margin', 1)
    rows = []
    for nm, (E, n) in results.items():
        mae, rmse, r2 = Errors.mae(E.h), Errors.rmse(E.h), E.r2()
        c = E.cm[0] if mg == 1 else E.cmm[0]
        acc, rec, pre, f1, mf1 = cm_scores(c)
        is_model = not nm.startswith('기준선')
        row = {'태그': tag, '평가': split, '모델': nm, '샘플수': n,
               '순증감_MAE': mae.mean(), '순증감_RMSE': rmse.mean(),
               **{f'순증감_MAE_t{j + 1}': mae[j] for j in range(HORIZON)},
               '순증감_R2_t1': r2[0],
               '방향_경계': mg, '방향_정확도_t1': acc, '방향_macroF1_t1': mf1,
               '부족_재현율_t1': rec[0], '부족_정밀도_t1': pre[0], '과잉_재현율_t1': rec[2], '과잉_정밀도_t1': pre[2],
               '반대오답_부족to과잉_t1': c[0, 2] / max(c[0].sum(), 1), '반대오답_과잉to부족_t1': c[2, 0] / max(c[2].sum(), 1)}
        if E.has_cls:
            cc = E.cmc[0]
            acc2, rec2, pre2, _, mf2 = cm_scores(cc)
            row.update({'분류출력_정확도_t1': acc2, '분류출력_macroF1_t1': mf2,
                        '분류출력_부족_재현율_t1': rec2[0], '분류출력_부족_정밀도_t1': pre2[0],
                        '분류출력_과잉_재현율_t1': rec2[2], '분류출력_과잉_정밀도_t1': pre2[2],
                        '분류출력_반대오답_부족to과잉_t1': cc[0, 2] / max(cc[0].sum(), 1),
                        '분류출력_반대오답_과잉to부족_t1': cc[2, 0] / max(cc[2].sum(), 1)})
        for key, pre_ in (('cls', '분류출력'), ('dist', '분포출력')):     # 판정 규칙·경보 기준별 (t+1)
            stt = E.pstat.get(key)
            if stt is None:
                continue
            if key == 'dist':
                a, r, p, _, m = cm_scores(stt['cm'][0])
                cd = stt['cm'][0]
                row.update({'분포출력_정확도_t1': a, '분포출력_macroF1_t1': m,
                            '분포출력_부족_재현율_t1': r[0], '분포출력_부족_정밀도_t1': p[0],
                            '분포출력_과잉_재현율_t1': r[2], '분포출력_과잉_정밀도_t1': p[2],
                            '분포출력_반대오답_t1': (cd[0, 2] + cd[2, 0]) / max(cd[0].sum() + cd[2].sum(), 1)})
            a, r, p, _, m = cm_scores(stt['med'][0])
            ch = stt['med'][0]
            row.update({f'{pre_}_절반규칙_정확도_t1': a, f'{pre_}_절반규칙_macroF1_t1': m,
                        f'{pre_}_절반규칙_부족_정밀도_t1': p[0], f'{pre_}_절반규칙_부족_재현율_t1': r[0],
                        f'{pre_}_절반규칙_과잉_정밀도_t1': p[2], f'{pre_}_절반규칙_과잉_재현율_t1': r[2],
                        f'{pre_}_절반규칙_반대오답_t1': (ch[0, 2] + ch[2, 0]) / max(ch[0].sum() + ch[2].sum(), 1)})
            ti = ALERT_TAUS.index(0.6)
            for ci, (c, cname) in enumerate(((0, '부족'), (2, '과잉'))):
                npred, tp = stt['alert'][0, ci, ti]
                row[f'{pre_}_경보0.6_{cname}_정밀도_t1'] = tp / npred if npred else np.nan
                row[f'{pre_}_경보0.6_{cname}_재현율_t1'] = tp / max(E.nact[0, c], 1)
        if E.sk is not None and E.sk.n_base:                       # 재고 상태 기준 (첫 기준)
            sm3, sp3 = stock_scores(E.sk.J['pred'][0, 2]), stock_scores(E.sk.J['pers'][0, 2])
            sm1 = stock_scores(E.sk.J['pred'][0, 0])
            row.update({'재고상태_정확도_t1': sm1['acc'], '재고상태_정확도_t3': sm3['acc'],
                        '재고상태_유지정확도_t3': sp3['acc'], '재고상태_macroF1_t3': sm3['mf1'],
                        '재고상태_바뀐시간정확도_t3': sm3['chg_acc'],
                        '재고_새부족_재현율_t3': sm3['new_rec_s'], '재고_새부족_정밀도_t3': sm3['new_pre_s'],
                        '재고_새과잉_재현율_t3': sm3['new_rec_o'], '재고_새과잉_정밀도_t3': sm3['new_pre_o']})
            for k, cname in enumerate(('부족', '과잉')):
                m_ = E.sk.W['pred'][0][k]
                tp, npred, npos = m_[1, 1], m_[:, 1].sum(), m_[1].sum()
                row[f'재고_3시간{cname}_정밀도'] = tp / npred if npred else np.nan
                row[f'재고_3시간{cname}_재현율'] = tp / npos if npos else np.nan
        if E.has_dist:
            n = np.maximum(E.dmae[:, 0], 1)
            row.update({'분포중앙값_MAE': float((E.dmae[:, 1] / n).mean()),
                        '예측구간80_적중률': float((E.pi[:, 1] / n).mean()),
                        '예측구간80_폭': float((E.pi[:, 2] / n).mean())})
        if D.report_grp is not None:                               # 400곳 = 팀 속성표에 있는 대여소
            for label, g in (('속성표대여소', D.report_grp), ('나머지대여소', ~D.report_grp)):
                a = E.st[g].sum(0)
                if a[0] > 0:
                    row[f'{label}_순증감_MAE'] = a[1] / a[0]
        if is_model:
            row['설정'] = describe(cfg)
            row['최고에폭'] = info.get('best_epoch', '')
            row['학습분'] = round(info['minutes'], 1) if 'minutes' in info else ''
        rows.append(row)
    df = pd.DataFrame(rows)
    num_cols = df.select_dtypes('number').columns
    df[num_cols] = df[num_cols].round(4)
    path = os.path.join(folder, f'gru_{split}_summary_{tag}.csv')
    df.to_csv(path, index=False, encoding='utf-8-sig')
    out(f'요약 CSV: {os.path.basename(path)}')


def report_outage(masked):
    out('')
    out('[재고 장애 가정 평가] 검증 샘플 일부의 입력 재고를 \'지금 n시간째 전 대여소 수집 장애 중\'으로 바꿔 예측')
    out(f'  {"장애 가정":<14}{"순증감 MAE (t+1/t+3)":<24}')
    for label, E in masked:
        mae = Errors.mae(E.h)
        out(f'  {label:<14}{mae[0]:.3f} / {mae[2]:.3f}')


def log_experiment(folder, cfg, info, E, split, D=None):
    mae, r2 = Errors.mae(E.h), E.r2()
    row = {'실행시각': time.strftime('%Y-%m-%d %H:%M'), '태그': cfg['tag'], '평가': split, '설정': describe(cfg),
           '에폭': info.get('epochs', ''), '최고에폭': info.get('best_epoch', ''),
           '최고검증손실': round(info['best_val'], 4) if 'best_val' in info else '',
           '학습분': round(info['minutes'], 1) if 'minutes' in info else '',
           '순증감MAE_t1': round(mae[0], 4), '순증감MAE_t2': round(mae[1], 4), '순증감MAE_t3': round(mae[2], 4),
           '순증감R2_t1': round(r2[0], 4),
           '과잉부족정확도_t1': round(E.cls(0)[0], 4), '부족재현율_t1': round(E.cls(0)[1], 4),
           '과잉재현율_t1': round(E.cls(0)[3], 4)}
    if E.has_cls:
        c = E.cmc[0]
        row['분류출력_정확도_t1'] = round(np.trace(c) / max(c.sum(), 1), 4)
        row['분류출력_부족재현율_t1'] = round(c[0, 0] / max(c[0].sum(), 1), 4)
        row['분류출력_부족정밀도_t1'] = round(c[0, 0] / max(c[:, 0].sum(), 1), 4)
        row['분류출력_과잉재현율_t1'] = round(c[2, 2] / max(c[2].sum(), 1), 4)
    if D is not None and D.report_grp is not None:
        a = E.st[D.report_grp].sum(0)
        if a[0] > 0:
            row['속성표대여소_순증감MAE'] = round(a[1] / a[0], 4)
    path = os.path.join(folder, 'gru_experiments.csv')
    df = pd.DataFrame([row])
    if os.path.exists(path):                       # 2차 기록(R² 열 없음)과 합칠 때 열을 맞춤
        try:
            df = pd.concat([pd.read_csv(path, encoding='utf-8-sig'), df], ignore_index=True)
        except Exception as e:
            out(f'기존 gru_experiments.csv 를 읽지 못해 gru_experiments_v3.csv 에 따로 씀 ({e})')
            path = os.path.join(folder, 'gru_experiments_v3.csv')
            if os.path.exists(path):
                df = pd.concat([pd.read_csv(path, encoding='utf-8-sig'), pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False, encoding='utf-8-sig')
    out(f'실험 기록 추가: {os.path.basename(path)}')


# ------------------------------------------------------------------
def peek_cfg(folder, tag):
    """저장된 모델의 설정만 읽음 (파일이 없거나 torch 가 없으면 None)."""
    path = os.path.join(folder, f'gru_best_{tag}.pt')
    if not os.path.exists(path):
        return None
    try:
        import torch
        return torch.load(path, map_location='cpu')['cfg']
    except Exception:
        return None


def day_type_code(v):
    """날종류 → 0 평일, 1 토, 2 일·공휴일 (모르는 값은 None)."""
    s = str(v).replace(' ', '')
    if '평일' in s:
        return 0
    if '공휴' in s or s.startswith('일') or '일요' in s:
        return 2
    if '토' in s:
        return 1
    return None


def read_emb_table(path):
    enc, sep, cols = sniff(path)
    df = pd.read_csv(path, encoding=enc, sep=sep)
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    need = ['대여소번호', '날종류', '시간대'] + EMB_STATS
    lack = [c for c in need if c not in df.columns]
    if lack:
        sys.exit(f'{os.path.basename(path)}: 필요한 열이 없음 {lack} (파일 열: {list(df.columns)})')
    for c in EMB_STATS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def find_emb_files(dirs):
    """station_embedding_YYYY.csv 를 찾아 {연도: 경로}. 처음 찾은 폴더의 파일만 씀."""
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        found = {}
        for p in sorted(glob.glob(os.path.join(d, '*station_embedding*.csv')), key=os.path.getmtime):
            m = re.search(r'(20\d{2})', os.path.basename(p))
            if m:
                found[int(m.group(1))] = p                 # 같은 해가 여러 개면 가장 최근 파일
        if found:
            return found
    return {}


def find_files(folder):
    """분할 파일(train_data / eval_data / test_data)을 찾음.
    찾는 순서: 폴더 자체 → 폴더/model_input → 폴더/processed/model_input (처음 찾은 폴더의 파일만 씀)
    반환: (파일 사전, 시작 시각 t0, 마지막 연도). 못 찾으면 ({}, None, None)"""
    for d in (folder, os.path.join(folder, 'model_input'), os.path.join(folder, 'processed', 'model_input')):
        files = {}
        for key, pat in (('train', '*train_data*.csv'), ('eval', '*eval_data*.csv'), ('test', '*test_data*.csv')):
            hits = sorted(glob.glob(os.path.join(d, pat)), key=os.path.getmtime)
            if len(hits) > 1:
                out(f'{pat} 에 맞는 파일이 {len(hits)}개라 가장 최근 것을 씀: {os.path.basename(hits[-1])}')
            if hits:
                files[key] = hits[-1]
        if files:
            if 'train' not in files or 'eval' not in files:
                sys.exit(f'{d}: train_data*.csv 와 eval_data*.csv 가 모두 있어야 함 (찾은 것: {list(files)})')
            out(f'데이터 폴더: {d}')
            return files, pd.Timestamp(f'{SPLITS["train"][0]}-01-01'), SPLITS['test'][1]
    return {}, None, None


def main():
    global L
    ap = argparse.ArgumentParser(description='따릉이 대여소 순증감 예측 최종 모델 (GRU + 피처 섞기 FiLM, v15_film). '
                                             '최종 설정이 기본값이라 학습은 --tag 만 주면 됨')
    ap.add_argument('folder', nargs='?', default=os.path.dirname(os.path.abspath(__file__)),
                    help='결과와 캐시를 저장할 폴더 (기본: 이 스크립트가 있는 폴더)')
    ap.add_argument('--data-dir', default='', help='train_data*.csv 등이 있는 폴더 (생략하면 폴더/model_input, 폴더 순으로 찾음)')
    ap.add_argument('--emb-dir', default='', help='station_embedding_YYYY.csv 가 있는 폴더 (생략하면 데이터 폴더, 결과 폴더에서 찾음)')
    ap.add_argument('--tag', default='v15_film_final', help='모델 이름. 모델 파일은 gru_best_<태그>.pt')
    ap.add_argument('--mix', choices=['film', 'none'], default='film',
                    help='film = 대여소·미래 칸이 과거 요약을 조절 (최종 모델), none = 이어 붙이기만 (비교 기준 v14 구조)')
    ap.add_argument('--epochs', type=int, default=FINAL['epochs'], help='최대 에폭 (기본 30, 검증 손실이 5에폭 안 줄면 멈춤)')
    ap.add_argument('--seed', type=int, default=SEED, help='학습 시드 (기본 42)')
    ap.add_argument('--epoch-samples', type=int, default=EPOCH_SAMPLES, help='에폭마다 뽑는 학습 샘플 수')
    ap.add_argument('--eval-only', action='store_true', help='학습 없이 저장된 gru_best_<태그>.pt 로 검증(2024)만 다시 평가')
    ap.add_argument('--test', action='store_true', help='저장된 모델로 테스트(2025) 평가 (맨 마지막에 한 번만)')
    ap.add_argument('--stock-eval', action='store_true',
                    help='대시보드 재고 상태 기준 평가 추가 (예측 재고 = 현재 재고 + 예측 순증감 합 → 부족·정상·과잉)')
    ap.add_argument('--eval-frac', type=float, default=1.0, help='평가 샘플 일부만 (오류 확인용. 보고용 숫자는 1.0)')
    ap.add_argument('--mask-eval', default='1,24,168', help='재고 장애 가정 시간들 (빈 문자열이면 생략)')
    args = ap.parse_args()

    cands = [args.data_dir] if args.data_dir else [os.path.join(args.folder, 'model_input'), args.folder]
    files = {}
    data_dir = ''
    for d in cands:
        if d and os.path.isdir(d):
            files, t0, last_year = find_files(d)
            if files:
                data_dir = d
                break
    if not files:
        sys.exit('데이터를 찾지 못함. 찾아본 폴더: ' + ', '.join(c for c in cands if c)
                 + '  (--data-dir 로 train_data*.csv 가 있는 폴더를 지정할 것)')
    out('데이터 파일: ' + ', '.join(f'{k}={os.path.basename(p)}' for k, p in files.items()))
    out(f'결과·캐시 저장 폴더: {os.path.abspath(args.folder)}')
    if args.test and 'test' not in files:
        sys.exit('test_data*.csv 가 없어 테스트 평가를 할 수 없음')
    if not 0 < args.eval_frac <= 1:
        sys.exit('--eval-frac 은 0보다 크고 1 이하')
    H = int((pd.Timestamp(f'{last_year + 1}-01-01') - t0).days) * 24

    reload = args.test or args.eval_only                  # 저장된 모델을 불러 평가만 함
    if reload:
        pth = os.path.join(args.folder, f'gru_best_{args.tag}.pt')
        if not os.path.exists(pth):
            sys.exit(f'{pth} 가 없음. 이 폴더에 모델 파일을 두거나 태그를 확인할 것')
        cfg = peek_cfg(args.folder, args.tag)
        if cfg is None:
            sys.exit(f'{pth} 의 설정을 읽지 못함 (torch 가 설치돼 있는지 확인)')
        check_supported(cfg, args.tag)
    else:
        cfg = {'tag': args.tag, **FINAL, **OFF, 'seed': args.seed, 'mix': args.mix, 'epochs': args.epochs}
    L = cfg['window']

    A = load_or_build(args.folder, files, t0, H, 0, False, last_year)
    ts = time.time()
    D = Data(A, t0, H)
    del A
    if cfg.get('stock_source', 'model') == 'file':
        F = load_file_stock(args.folder, files, t0, H, D.stations, 'all')
        D.apply_file_stock(F, cfg.get('file_fill_skip') or [], cfg.get('file_zero_rule', 'keep'))
        del F
    if D.op_counts:
        out(f'운영방식: QR만 {D.op_counts[0]:,}곳, LCD만 {D.op_counts[1]:,}곳, 둘 다 {D.op_counts[2]:,}곳')
    out(f'학습 기간 기록이 없는 대여소(새 대여소): {int(D.new_station.sum()):,}곳')

    # 대여소 속성표: 팀 표(400곳)는 결과를 나눠 보는 기준으로 씀. 입력 표는 코드 계산 표(최종) 또는 팀 표
    want_emb = bool(cfg.get('station_emb'))
    here = os.path.dirname(os.path.abspath(__file__))
    team_paths = find_emb_files([args.emb_dir, data_dir, os.path.dirname(data_dir) if data_dir else '',
                                 args.folder, os.path.join(args.folder, 'model_input'), here])
    if team_paths:
        D.set_report_group(team_paths)
    if want_emb and cfg.get('emb_source', 'team') == 'self':
        D.load_station_emb(D.self_emb_tables(), source='self')
    elif want_emb and team_paths:
        D.load_station_emb(team_paths, source='team')
    elif want_emb:
        sys.exit('station_embedding_YYYY.csv 를 찾지 못함. 데이터 파일과 같은 폴더에 두거나 --emb-dir 로 지정할 것')
    if D.report_grp is None:
        out('팀 속성표(station_embedding_YYYY.csv) 없음: 400곳 / 나머지 나눠 보기는 건너뜀')
    D.use_emb = want_emb
    D.use_profile = cfg['profile']
    if not reload:                                        # 새로 학습: 입력 설정 (불러올 때는 load_gru 가 맞춤)
        if cfg['emb_no_use']:
            D.drop_emb_use()
        D.use_past_profile = cfg['past_profile']
        out('입력 창 각 시각에 평소 패턴(최근 4주 같은 요일·시간 평균 대여·반납) 2개를 더함')
        out('입력에서 뺀 이용 통계: ' + ', '.join(COLS[k][0] for k in STATS))
        D.set_neighbors(cfg['nb_k'])
        D.nb_cfg = cfg['nb_k']
        D.set_exclude(cfg['exclude_hours'])
        cfg['sid_n'] = D.S
        out(f'대여소 ID 임베딩: {D.S:,}곳 + 모르는 대여소 1칸 × {cfg["station_id_emb"]}차원 '
            f'(학습 때 {SID_DROP:.0%}는 모르는 대여소로 바꿔 새 대여소 {int(D.new_station.sum())}곳에 쓸 벡터도 학습)')
    split = 'test' if args.test else 'val'

    try:
        import torch
    except ImportError:
        sys.exit('torch 가 없음 (pip install torch)')
    info = {}
    if reload:
        if args.test:
            out('※ 테스트셋 평가. 이 결과를 보고 모델을 다시 고치면 테스트 점수가 의미를 잃음')
        else:
            out(f'※ 학습 없이 저장된 모델(gru_best_{args.tag}.pt)로 검증(2024)만 다시 평가')
        model, dev, c = load_gru(D, args.folder, args.tag)
        cfg = dict(c, tag=args.tag)                       # 결과 파일 이름은 불러온 모델 파일 이름 기준
        out(f'설정 [{cfg["tag"]}]: {describe(cfg)}')
    else:
        out(f'피처 준비 완료 ({time.time() - ts:.0f}초): 대여소 {D.S:,}곳, 입력 {D.n_in}개 × {L}시간, '
            f'고정 {D.n_static}+임베딩 {EMB}+대여소 {cfg["station_id_emb"]}, 미래 {HORIZON * D.n_future}개')
        out(f'설정 [{cfg["tag"]}]: {describe(cfg)}')
        model, dev, info = train_gru(D, args.folder, cfg, args.epoch_samples)

    D.cls_margin = cfg.get('cls_margin', 1) if cfg.get('cls_head') else 1
    report_direction.margin = D.cls_margin
    if args.stock_eval:                                   # 대시보드 재고 상태 기준 (주 기준 + 최소 2대 기준 요약)
        D.stock_crits = [(f'{STOCK_SHORT:.0%} / {STOCK_OVER:.0%}', STOCK_SHORT, 0.0, STOCK_OVER),
                         (f'max(2대, {STOCK_SHORT:.0%}) / {STOCK_OVER:.0%}', STOCK_SHORT, 2.0, STOCK_OVER)]
    samples = D.eval_samples(split, args.eval_frac)
    if args.eval_frac < 1:
        out(f'※ 평가 샘플의 {args.eval_frac:.0%}만 평가 (오류 확인용. 보고용 숫자로 쓰지 말 것)')
    results = {}
    results['기준선1 1주전'] = evaluate(D, samples, pred_lastweek, '기준선1')
    results['기준선2 평균'] = evaluate(D, samples, pred_mean, '기준선2', stock=True)
    predict = make_predict(torch, model, dev)
    results['GRU'] = evaluate(D, samples, predict, 'GRU', stock=True)
    report(D, split, results, args.folder, cfg['tag'])
    save_summary(D, results, args.folder, split, cfg['tag'], cfg, info)

    sub = D.eval_samples(split, VAL_FRAC, EVAL_SEED + 1)
    if getattr(model, 'mixer', None) is not None:         # 피처 섞기(FiLM)가 한 일
        report_mix(args.folder, cfg['tag'], *mix_profile(torch, model, dev, D, sub))
    durs = [int(v) for v in args.mask_eval.split(',') if v.strip()]
    if durs:
        masked = [('기준선2 평균', evaluate(D, sub, pred_mean, '기준선2 일부')[0]),
                  ('장애 없음', evaluate(D, sub, predict, '장애 없음')[0])]
        for d in durs:
            masked.append((f'{d}시간째 장애', evaluate(D, sub, with_outage(predict, d), f'{d}시간 장애')[0]))
        report_outage(masked)
    if not args.eval_only:
        log_experiment(args.folder, cfg, info, results['GRU'][0], split, D)

    suffix = '_test' if args.test else ('_eval' if args.eval_only else '')
    res = os.path.join(args.folder, f'gru_run_result_{cfg["tag"]}{suffix}.txt')
    with open(res, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(LINES) + '\n')
    print(f'\n결과 저장: {res}')


if __name__ == '__main__':
    main()

