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
