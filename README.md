# jarvis_cook_project
ai programming project

## 환경 설정 (맥/윈도우 공통, 파이썬 3.11)
```
conda create -n jarvis python=3.11 -y
conda activate jarvis
pip install -r requirements.txt
```

## 손 인식 테스트
```
python hand_test.py
```

## 데이터 수집 방법 (팀원용)

### 처음 한 번만
1. 아나콘다(Anaconda)를 설치합니다. 이미 있으면 건너뜁니다.
2. 터미널(윈도우는 Anaconda Prompt)에서 한 줄씩 실행합니다.
```
git clone https://github.com/jaehk28-hash/jarvis_cook_project.git
cd jarvis_cook_project
conda create -n jarvis python=3.11 -y
conda activate jarvis
pip install -r requirements.txt
```

### 수집할 때마다
```
cd jarvis_cook_project
git pull
conda activate jarvis
python collect.py 내이름 1
```
- `내이름`은 영어로 씁니다. 예: `gaon`, `yunho`
- 뒤의 `1`은 세션 번호입니다. 장소나 조명이 바뀌면 `2`, `3`으로 올립니다.

### 창이 뜨면 (창을 한 번 클릭한 뒤 키를 누릅니다)
| 키 | 동작 | 모양 |
|---|---|---|
| 1 | thumb | 따봉 (엄지를 위로 확실히) |
| 2 | v | 브이 |
| 3 | cook | 요리하는 동작 (썰기, 젓기, 칼/국자 쥐기, 재료 집기) |
| 4 | idle | 손을 내리거나 가만히 두기, 아무 모양 |
| 5 | palm | 손바닥 펴기 |
| 6 | fist | 주먹 (엄지는 손가락 위로 접기) |
| 0 | | 기록 멈춤 |
| q | | 종료 |

1. 키를 누르면 화면 왼쪽 위에 `REC: 동작이름`이 뜨고 숫자가 올라갑니다.
2. 그 모양을 유지하면서 손을 천천히 움직입니다. 거리(가깝게/멀게), 각도(앞/옆/기울임), 화면 위치(가운데/구석)를 바꿔 가며 찍습니다.
3. 숫자가 목표만큼 되면 `0`을 누르고 다음 동작으로 넘어갑니다.
4. 다 끝나면 `q`를 누릅니다. `data/내이름_1.csv` 파일이 생깁니다.

### 목표 (1인당, 세션 2개 이상)
- thumb, v, palm, fist, idle: 각 350개 이상
- cook: 1000개 이상
- 한 자리에서만 많이 찍는 것보다 장소와 각도를 바꿔 찍는 게 더 좋습니다.

### 올리기
```
git add data
git commit -m "내이름 데이터 추가"
git pull
git push
```
- 영상은 저장되지 않고 손 좌표 숫자만 저장됩니다.
- 안 될 때는 에러 메시지를 복사해서 팀 채팅방에 올립니다.

## 학습, 비교, 실시간 측정

모든 명령은 프로젝트 폴더에서 `conda activate jarvis`를 한 뒤 실행합니다.
세션 이름은 `data/`의 파일 이름입니다(`data/test_1.csv` → `test_1`).
명령 동작은 `thumb`, `v`, `palm`, `fist`이고, `cook`과 `idle`은 비명령(none)입니다.

### 신경망 학습: train.py
```
python train.py --test test_1 test_2 test_3
python train.py --test test_1 test_2 test_3 --only-commands
```
| 옵션 | 설명 |
|---|---|
| `--test 세션...` | 시험용 세션. 나머지는 학습에 씁니다(각 세션·동작의 마지막 15%는 검증용). |
| `--val 세션...` | 검증용 세션을 따로 지정합니다. `--test`와 같이 써야 합니다. |
| `--only-commands` | cook, idle을 빼고 명령 4개만 학습합니다. 시험은 6개 동작 전체로 하며 cook, idle의 정답은 비명령입니다. `model_cmd_only.pt`로 저장합니다. |
| `--epochs` | 최대 반복 횟수 (기본 300) |
| `--dropout`, `--lr`, `--wd`, `--smooth` | 드롭아웃(0.2), 학습률(1e-3), 가중치 감쇠(0), 라벨 스무딩(0) |
| `--aug` | 학습 데이터 증강 |
| `--cook-weight` | cook 가중치 배수 (`--only-commands`에서는 효과 없음) |

`--test`를 생략하면 세션 안에서 시간 순서대로 70/15/15로 나눕니다.
일반 학습은 `model.pt`에 저장합니다.

### 기준 모델: baseline.py
```
python baseline.py --test test_1 test_2 test_3
```
로지스틱 회귀(`max_iter=3000`)와 랜덤 포레스트(`n_estimators=200`, `random_state=0`)를 학습하고 평가합니다.
`--test`, `--val`의 뜻과 데이터 분할은 `train.py`와 같으며, 신경망이 학습에 쓰는 부분만 학습합니다.

### 결과 모음: results.csv
`train.py`와 `baseline.py`를 실행할 때마다 한 줄씩 추가됩니다.
- 열: 모델이름, test세션, 6클래스정확도, 명령기준정확도, macro F1, 게이트0.0_cook오작동, 게이트0.8_cook오작동, 게이트0.8_명령인식률
- 모델이름: `mlp`, `mlp_cmd_only`, `logreg`, `random_forest`
- 값은 0~1 사이 비율입니다(0.05 = 5%).
- `mlp_cmd_only`는 4개만 출력하므로 6클래스정확도와 macro F1 칸이 비어 있습니다.
- `--aug` 같은 학습 설정은 기록되지 않습니다. 처음부터 다시 모으려면 파일을 지우세요.

### 실시간 실행: realtime.py
```
python realtime.py --log                              # 게이트 켬 (조작 없이 화면 표시만)
python realtime.py --log --gate-off                   # 게이트 끔
python realtime.py --log --model model_cmd_only.pt    # 명령 4개 모델
python realtime.py --live                             # 실제 키보드/마우스 조작
```
| 옵션 | 설명 |
|---|---|
| `--live` | 실제로 키보드/마우스를 조작합니다. 없으면 화면에 글자만 표시합니다. |
| `--model 파일` | 모델 파일 (기본 `model.pt`). 출력 클래스 수는 파일에서 읽습니다. |
| `--thresh` | 신뢰도 기준 (기본 0.8) |
| `--window` | 유지시간(초, 기본 0.5) |
| `--log` | 실행 기록과 측정 기능을 켭니다. |
| `--gate-off` | 신뢰도·유지시간 게이트를 끄고 매 프레임 예측대로 실행합니다. 재실행 대기와 커서 모드 조건도 없습니다. `--thresh`, `--window`는 무시됩니다. |

> 주의: `--gate-off`와 `--live`를 같이 쓰면 매 프레임 키 입력과 클릭이 발생합니다. 비교 측정은 `--live` 없이 하세요.

카메라 창을 클릭한 뒤 키를 누릅니다.

| 키 | 동작 |
|---|---|
| `r` | 측정 시작/정지. 정지하면 터미널에 분당 실행 횟수를 출력합니다. (`--log`) |
| `1` `2` `3` `4` | thumb, v, palm, fist의 반응 시간 측정. 키를 누른 뒤 해당 손 모양을 만듭니다. (`--log`) |
| `q`, `Esc` | 종료 (한글 입력 상태에서는 `q`, `r`이 안 되므로 영문으로 바꾸세요) |

- 기록은 `logs/realtime_날짜시각.csv`에 저장됩니다. 실행된 명령(KEY, CLICK)과 시각이 남습니다.
- 실행 횟수에는 KEY와 CLICK만 들어갑니다(커서 이동은 제외).
- palm은 KEY/CLICK이 없으므로 커서 모드가 켜지는 순간까지를 잽니다(`--gate-off`에서는 첫 커서 이동). 이때 기록에는 `CURSOR`로 남습니다.
- 반응 시간을 잴 때는 손을 내린 상태에서 숫자 키를 누르세요. 10초 안에 실행되지 않으면 그 측정은 버립니다.
- fist(`4`)는 게이트를 켠 상태에서는 커서 모드(손바닥 유지)에서만 클릭됩니다.
- 반응 시간에는 키를 누른 뒤 사람이 손 모양을 만드는 시간도 포함됩니다.

## 더블클릭으로 실행하기
- 맥: `start_mac.command` 더블클릭 (처음 한 번은 우클릭 → 열기). 터미널에 카메라·손쉬운 사용 권한 필요
- 윈도우: `start_windows.bat` 더블클릭
- 둘 다 conda `jarvis` 환경을 자동으로 켜고 `python realtime.py --live`를 실행합니다. 끌 때는 카메라 창에서 Esc
- 감도 조절: 파일 안의 `python realtime.py --live` 뒤에 `--gain 1.5` 처럼 추가
