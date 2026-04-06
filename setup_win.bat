@echo off
setlocal EnableExtensions

REM ============================================================
REM PowerShell-oriented local setup for gsplat + dust3r on Windows
REM Expected layout:
REM   .\setup_win.bat
REM   .\requirements.txt
REM   .\datasets\
REM   .\simple_trainer_prune_v2.py
REM   .\dust3r_to_3dgs_verified.py
REM   .\gsplat\
REM   .\gsplat\fused-bilagrid\
REM   .\dust3r\
REM ============================================================

set "ENV_NAME=dust3r-gsplat"
set "ROOT_DIR=%~dp0"
set "ROOT_DIR=%ROOT_DIR:~0,-1%"
set "REQ_FILE=%ROOT_DIR%\requirements.txt"
set "GSPLAT_DIR=%ROOT_DIR%\gsplat"
set "FUSED_BILAGRID_DIR=%GSPLAT_DIR%\fused-bilagrid"
set "DUST3R_DIR=%ROOT_DIR%\dust3r"

echo ============================================================
echo PowerShell local setup for gsplat + dust3r
echo Environment: %ENV_NAME%
echo Root: %ROOT_DIR%
echo ============================================================
echo.

echo [1/11] Checking local source directories...
if not exist "%GSPLAT_DIR%\setup.py" (
    echo [ERROR] Local gsplat source not found: %GSPLAT_DIR%
    goto :fail
)
if not exist "%FUSED_BILAGRID_DIR%\setup.py" (
    echo [ERROR] Local fused-bilagrid source not found: %FUSED_BILAGRID_DIR%
    goto :fail
)
if not exist "%DUST3R_DIR%\croco\models\curope\setup.py" (
    echo [ERROR] Local dust3r curope setup.py not found: %DUST3R_DIR%\croco\models\curope\setup.py
    goto :fail
)
if not exist "%REQ_FILE%" (
    echo [ERROR] requirements.txt not found: %REQ_FILE%
    goto :fail
)
echo [OK] Local sources detected.

echo [2/11] Checking Conda / CUDA / MSVC tools...
where conda >nul 2>nul || (echo [ERROR] conda not found in PATH. & goto :fail)
where nvcc >nul 2>nul || (echo [ERROR] nvcc not found in PATH. Install CUDA Toolkit first. & goto :fail)
where cl >nul 2>nul || (echo [ERROR] cl.exe not found in PATH. Open VS developer shell or install VS C++ Build Tools. & goto :fail)
for /f "delims=" %%i in ('nvcc --version ^| findstr /C:"release"') do echo [INFO] %%i

echo [3/11] Creating Conda environment if needed...
call conda env list | findstr /R /C:"^%ENV_NAME% " >nul
if errorlevel 1 (
    echo [INFO] Creating environment %ENV_NAME% with Python 3.10 ...
    call conda create -n %ENV_NAME% python=3.10 -y
    if errorlevel 1 goto :fail
) else (
    echo [INFO] Environment %ENV_NAME% already exists. Reusing it.
)

echo [4/11] Installing PyTorch CUDA 12.9 wheels...
call conda run -n %ENV_NAME% python -m pip install --upgrade pip
if errorlevel 1 goto :fail
call conda run -n %ENV_NAME% python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
if errorlevel 1 goto :fail

echo [5/11] Installing base build dependencies...
call conda run -n %ENV_NAME% python -m pip install --upgrade setuptools wheel packaging ninja cmake
if errorlevel 1 goto :fail

echo [6/11] Installing project requirements...
call conda run -n %ENV_NAME% python -m pip install --no-cache-dir -r "%REQ_FILE%"
if errorlevel 1 goto :fail

echo [7/11] Installing local gsplat package...
if exist "%GSPLAT_DIR%\build" rmdir /s /q "%GSPLAT_DIR%\build"
for %%d in ("%GSPLAT_DIR%\*.egg-info") do if exist "%%~fd" rmdir /s /q "%%~fd"
call conda run -n %ENV_NAME% python -m pip uninstall -y gsplat >nul 2>nul
call conda run -n %ENV_NAME% python -m pip install --no-cache-dir --no-build-isolation -e "%GSPLAT_DIR%"
if errorlevel 1 (
    echo [ERROR] Failed to install local gsplat package.
    goto :fail
)

echo [8/11] Installing local fused-bilagrid package...
if exist "%FUSED_BILAGRID_DIR%\build" rmdir /s /q "%FUSED_BILAGRID_DIR%\build"
for %%d in ("%FUSED_BILAGRID_DIR%\*.egg-info") do if exist "%%~fd" rmdir /s /q "%%~fd"
call conda run -n %ENV_NAME% python -m pip uninstall -y fused_bilagrid >nul 2>nul
call conda run -n %ENV_NAME% cmd /c "set TORCH_CUDA_ARCH_LIST=12.0&& python -m pip install --no-cache-dir --no-build-isolation -e \"%FUSED_BILAGRID_DIR%\""
if errorlevel 1 (
    echo [ERROR] Failed to install local fused-bilagrid package.
    goto :fail
)

echo [9/11] Building local dust3r curope extension...
call conda run -n %ENV_NAME% cmd /c "set TORCH_CUDA_ARCH_LIST=12.0&& cd /d \"%DUST3R_DIR%\croco\models\curope\" && python setup.py build_ext --inplace"
if errorlevel 1 (
    echo [ERROR] Failed to build dust3r curope extension.
    goto :fail
)

echo [10/11] Writing PowerShell conda hooks...
call conda run -n %ENV_NAME% python -c "import os; p=os.environ['CONDA_PREFIX']; os.makedirs(os.path.join(p,'etc','conda','activate.d'), exist_ok=True); os.makedirs(os.path.join(p,'etc','conda','deactivate.d'), exist_ok=True); act=os.path.join(p,'etc','conda','activate.d','dust3r_gsplat_vars.ps1'); deact=os.path.join(p,'etc','conda','deactivate.d','dust3r_gsplat_vars.ps1'); open(act,'w',encoding='utf-8').write('$env:PYTHONPATH = \"%DUST3R_DIR%;%ROOT_DIR%;\" + $env:PYTHONPATH\n$env:TORCH_CUDA_ARCH_LIST = \"12.0\"\n'); open(deact,'w',encoding='utf-8').write('$env:TORCH_CUDA_ARCH_LIST = $null\n')"
if errorlevel 1 (
    echo [ERROR] Failed to write PowerShell hooks.
    goto :fail
)

echo [11/11] Installing helper packages and verifying...
call conda run -n %ENV_NAME% python -m pip install --no-cache-dir pycolmap "fused-ssim @ git+https://github.com/rahul-goel/fused-ssim/@88169c51c22973ad8fd2429c3298d8356fdd5dc8" "numpy<2.0"
if errorlevel 1 (
    echo [ERROR] Failed to install pycolmap / fused-ssim / numpy constraint.
    goto :fail
)

call conda run -n %ENV_NAME% python -c "import torch; print('torch=', torch.__version__)"
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo [SUCCESS] Setup completed.
echo ============================================================
echo IMPORTANT:
echo   1. Close this PowerShell window.
echo   2. Open a NEW PowerShell window.
echo   3. Run:
echo        conda activate %ENV_NAME%
echo        python -c "from gsplat.exporter import export_splats; from gsplat.rendering import rasterization; print('gsplat ok')"
echo        python -c "from dust3r.model import AsymmetricCroCo3DStereo; print('dust3r ok')"
echo.
pause
goto :eof

:fail
echo.
echo ============================================================
echo [FAILED] Environment setup stopped.
echo ============================================================
pause
exit /b 1
