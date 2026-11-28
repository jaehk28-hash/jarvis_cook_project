#!/bin/bash
# [맥] 더블클릭하면 손동작 제어를 시작합니다. 끌 때는 카메라 창을 클릭하고 Esc.
cd "$(dirname "$0")"
for base in "$HOME/anaconda3" "$HOME/miniconda3" "/opt/anaconda3" "/opt/miniconda3" "/opt/homebrew/anaconda3" "/opt/homebrew/Caskroom/miniconda/base"; do
  if [ -f "$base/etc/profile.d/conda.sh" ]; then source "$base/etc/profile.d/conda.sh"; break; fi
done
if ! command -v conda >/dev/null 2>&1; then
  echo "conda를 찾지 못했습니다. 아나콘다 설치 위치를 확인하세요."; read -p "Enter를 누르면 닫힙니다"; exit 1
fi
conda activate jarvis || { echo "jarvis 환경이 없습니다. README의 환경 설정을 먼저 하세요."; read -p "Enter"; exit 1; }
echo "손동작 제어를 시작합니다. 유튜브 창을 클릭해서 앞에 두세요. 끄려면 카메라 창에서 Esc."
python realtime.py --live
read -p "종료되었습니다. Enter를 누르면 닫힙니다"
