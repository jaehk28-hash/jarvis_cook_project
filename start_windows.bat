@echo off
chcp 65001 >nul
rem [윈도우] 더블클릭하면 손동작 제어를 시작합니다. 끌 때는 카메라 창을 클릭하고 Esc.
cd /d "%~dp0"
set "ACT="
for %%B in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3" "C:\anaconda3" "C:\miniconda3") do if not defined ACT if exist "%%~B\Scripts\activate.bat" set "ACT=%%~B\Scripts\activate.bat"
if not defined ACT (
  echo conda를 찾지 못했습니다. 아나콘다 설치 위치를 확인하세요.
  pause
  exit /b 1
)
call "%ACT%" jarvis
if errorlevel 1 (
  echo jarvis 환경이 없습니다. README의 환경 설정을 먼저 하세요.
  pause
  exit /b 1
)
echo 손동작 제어를 시작합니다. 유튜브 창을 클릭해서 앞에 두세요. 끄려면 카메라 창에서 Esc.
python realtime.py --live
pause
