@echo off
setlocal enabledelayedexpansion

REM Optional helper script for Windows builds.
REM If you prefer pure command-line, you can ignore this file.

REM Override GPU arch if needed (examples: 75/80/86/89/90)
if "%SM%"=="" set SM=75

REM Workaround for occasional nvcc/cudafe++ instability on Windows
if "%NVCC_THREADS%"=="" set NVCC_THREADS=1

set TARGET=vecMult.exe
set SRC=vecMult.cu

REM Initialize MSVC environment (VS 2022 BuildTools)
set VSDEV="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
if exist %VSDEV% (
  call %VSDEV% -no_logo -arch=amd64 -host_arch=amd64 >nul
)

set CCBIN="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
if not exist %CCBIN%\cl.exe (
  echo [build] ERROR: cl.exe not found at %CCBIN%
  echo [build] Fix by updating the MSVC version folder inside %CCBIN%
  exit /b 1
)

echo [build] nvcc --threads %NVCC_THREADS% -ccbin %CCBIN% -m64 -O3 -std=c++14 -arch=sm_%SM% -lineinfo -Xcompiler "/utf-8 /EHsc" -o %TARGET% %SRC%
nvcc --threads %NVCC_THREADS% -ccbin %CCBIN% -m64 -O3 -std=c++14 -arch=sm_%SM% -lineinfo -Xcompiler "/utf-8 /EHsc" -o %TARGET% %SRC%
if errorlevel 1 (
  echo [build] ERROR: nvcc failed.
  echo [build] Try:
  echo [build]   set NVCC_THREADS=1
  echo [build]   nvcc --threads 1 -ccbin %CCBIN% -m64 -O0 -std=c++14 -arch=sm_%SM% -o %TARGET% %SRC%
  exit /b 1
)

echo [build] OK: %TARGET%
endlocal

@echo off
setlocal enabledelayedexpansion

REM Build script for Windows (CMD)
REM Requirements:
REM - CUDA toolkit installed (nvcc in PATH)
REM - Visual Studio Build Tools (MSVC) installed (this script will try to auto-locate it)
REM - GPU with Tensor Cores (sm_70+)

REM Optional:
REM - set SM=86                   (GPU arch, e.g. 75/80/86/89/90)
REM - set NVCC_THREADS=1          (workaround for some nvcc/cudafe++ crashes on Windows)
REM - set OPTFLAGS=-O3            (override optimization level)

REM You can override SM from environment, e.g.:
REM   set SM=86
REM   build.bat
if "%SM%"=="" set SM=75
if "%NVCC_THREADS%"=="" set NVCC_THREADS=1

set TARGET=vecMult.exe
set SRC=vecMult.cu

REM ---- Locate and initialize MSVC environment (important for nvcc on Windows) ----
set VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe
set VSINSTALL=
if exist "%VSWHERE%" (
  for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    set VSINSTALL=%%i
  )
)

if not defined VSINSTALL (
  REM Fallback common path (BuildTools)
  if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools" (
    set VSINSTALL=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools
  )
)

if defined VSINSTALL (
  if exist "%VSINSTALL%\Common7\Tools\VsDevCmd.bat" (
    call "%VSINSTALL%\Common7\Tools\VsDevCmd.bat" -no_logo -arch=amd64 -host_arch=amd64 >nul
  )
)

REM If VsDevCmd ran, VCToolsInstallDir is set and we can point nvcc at cl.exe explicitly.
set CCBIN=
if defined VCToolsInstallDir (
  if exist "%VCToolsInstallDir%bin\Hostx64\x64\cl.exe" (
    set CCBIN=%VCToolsInstallDir%bin\Hostx64\x64
  )
)

REM CUDA 13.x Windows note:
REM If nvcc does not use MSVC, some headers may pick the wrong inline-PTX pointer constraint.
REM Passing -ccbin (MSVC) usually fixes it; keeping __LP64__ as an extra guard doesn't hurt.

set OPT=-O3
if not "%OPTFLAGS%"=="" set OPT=%OPTFLAGS%

if defined CCBIN (
  echo [build] Using MSVC from: %CCBIN%
  where cl >nul 2>nul
  if errorlevel 1 (
    echo [build] WARNING: cl.exe not on PATH even after VsDevCmd. Continuing with -ccbin anyway.
  )
  echo [build] nvcc --threads %NVCC_THREADS% -ccbin "%CCBIN%" -m64 %OPT% -std=c++14 -arch=sm_%SM% -lineinfo -D__LP64__ -Xcompiler "/utf-8 /EHsc" -o %TARGET% %SRC%
  nvcc --threads %NVCC_THREADS% -ccbin "%CCBIN%" -m64 %OPT% -std=c++14 -arch=sm_%SM% -lineinfo -D__LP64__ -Xcompiler "/utf-8 /EHsc" -o %TARGET% %SRC%
) else (
  echo [build] ERROR: MSVC x64 toolchain not found.
  echo [build] Please install "Visual Studio 2022 Build Tools" with "Desktop development with C++"
  echo [build] and the component "MSVC v143 - VS 2022 C++ x64/x86 build tools".
  echo [build] Then rerun this script.
  exit /b 1
)
if errorlevel 1 (
  echo [build] ERROR: nvcc failed.
  echo [build] If you see 'cudafe++ died with 0xC0000005', try:
  echo [build]   set NVCC_THREADS=1
  echo [build]   set OPTFLAGS=-O0
  echo [build]   build.bat
  exit /b 1
)

echo [build] OK: %TARGET%
endlocal


