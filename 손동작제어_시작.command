#!/bin/bash
# 더블클릭하면 손동작 제어를 시작합니다. 끌 때는 카메라 창을 클릭하고 Esc.
cd "$(dirname "$0")"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate jarvis
echo "손동작 제어를 시작합니다. 유튜브 창을 클릭해서 앞에 두세요. 끄려면 카메라 창에서 Esc."
python realtime.py --live
