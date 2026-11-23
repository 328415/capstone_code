@echo off
chcp 65001 > nul
setlocal
rem ======================================================================
rem 최종 모델 v15_film 을 처음부터: 학습(2022~2023) → 검증(2024) → 테스트(2025)
rem  - 설정은 v15_film 과 같음 (v14_file_ex 설정 + 피처 섞기 FiLM). --like 없이 옵션을 모두 직접 적음
rem    → gru_best_v14_file_ex.pt 가 없어도 됨
rem  - 이 폴더에 필요한 것: gru_final.py (최종 모델 코드)
rem    데이터: 이 폴더 또는 이 폴더\model_input 에 train_data_2022_2023.csv, eval_data_2024.csv, test_data_2025.csv
rem            (다른 곳에 있으면 아래 DATA_DIR 에 그 폴더 경로를 적을 것)
rem  - 캐시(gru_data_all.npz, gru_stock_file_all.npz)가 없으면 처음 한 번 CSV 를 읽어 새로 만듦
rem  - 시간: 오류 확인 몇 분 + 학습·검증 약 3시간 + 테스트(학습 없이 평가만)
rem  - 테스트(2025)는 한 번만. 테스트 결과를 본 뒤에는 모델을 고치지 않음
rem    테스트를 빼고 돌리려면 RUN_TEST=0
rem ======================================================================

set TAG=v15_film_final
set RUN_TEST=1
set DATA_DIR=

set DD=
if not "%DATA_DIR%"=="" set DD=--data-dir "%DATA_DIR%"

set FLAGS=--mix film --epochs 30 --lr-decay --neighbors --station-emb --emb-source self --emb-no-use --station-id-emb 8 --cls-head --dist-head --past-profile --net-only --no-stats --stock-source file --exclude-hours 2023-09-07:7-10 --seed 42

if not exist gru_final.py (
  echo gru_final.py 가 이 폴더에 없음
  goto err
)
if exist gru_best_%TAG%.pt (
  echo gru_best_%TAG%.pt 가 이미 있음. 덮어쓰지 않으려면 TAG 를 바꾸거나 파일을 옮긴 뒤 다시 실행할 것
  goto err
)

echo [0/2] 오류 확인: 1에폭, 표본 2만 개, 검증 5%%만 (숫자는 보지 말 것)
python gru_final.py . --tag %TAG%_smoke %FLAGS% %DD% --epochs 1 --epoch-samples 20000 --eval-frac 0.05 --mask-eval ""
if errorlevel 1 goto err

echo [1/2] 학습 2022~2023 + 검증 2024 평가 (재고 상태 평가 포함)
python gru_final.py . --tag %TAG% %FLAGS% %DD% --stock-eval
if errorlevel 1 goto err

if "%RUN_TEST%"=="1" (
  echo [2/2] 테스트 2025 평가 - 한 번만
  python gru_final.py . --tag %TAG% --test --stock-eval %DD%
  if errorlevel 1 goto err
) else (
  echo [2/2] 테스트는 건너뜀 ^(RUN_TEST=0^)
)

echo.
echo 끝.
echo  학습·검증: gru_run_result_%TAG%.txt, gru_val_summary_%TAG%.csv, gru_val_stock_%TAG%.csv 등, 모델 gru_best_%TAG%.pt
if "%RUN_TEST%"=="1" echo  테스트: gru_run_result_%TAG%_test.txt, gru_test_summary_%TAG%.csv, gru_test_stock_%TAG%.csv 등
echo  비교 기준 v15_film (검증 2024): 순증감 MAE 1.1320, RMSE 1.8890, 방향 정확도 0.6235, 재고 상태 t+3 0.836
echo  오류 확인용 %TAG%_smoke 결과 파일은 지워도 됨
goto :eof

:err
echo.
echo 오류로 멈춤. 마지막 화면의 오류 메시지를 보낼 것
exit /b 1
