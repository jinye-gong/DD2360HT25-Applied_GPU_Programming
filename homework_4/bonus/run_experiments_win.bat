@echo off
setlocal enabledelayedexpansion

REM Minimal Windows experiment runner (CMD).
REM It saves console outputs into bonus\results\*.txt.

set EXE=.\vecMult.exe
set OUTDIR=results

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

if not exist "%EXE%" (
  echo Error: %EXE% not found. Build first:
  echo   build.bat
  exit /b 1
)

echo ==========================================
echo Experiment 1: A(1024x2048) x B(2048x1024)
echo ==========================================
%EXE% 1024 2048 2048 1024 > "%OUTDIR%\exp1_1024x2048.txt" 2>&1
type "%OUTDIR%\exp1_1024x2048.txt"

echo.
echo ==========================================
echo Experiment 2: Square sizes (saving per-size logs)
echo ==========================================
for %%S in (512 1024 2048 4096 8192) do (
  echo Running size=%%S...
  %EXE% %%S %%S %%S %%S > "%OUTDIR%\exp2_square_%%S.txt" 2>&1
)

echo.
echo ==========================================
echo Experiment 3: WMMA different warps-per-block
echo ==========================================
REM threadsPerBlock = 32 * warpsPerBlock
for %%W in (1 2 4 8) do (
  echo Running warpsPerBlock=%%W...
  %EXE% 8192 8192 8192 8192 --wmma-warps %%W > "%OUTDIR%\exp3_wmma_warps_%%W.txt" 2>&1
)

echo.
echo Done. Logs are in %OUTDIR%\.
endlocal


