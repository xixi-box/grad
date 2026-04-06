@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

echo ============================================================
echo Environment Diagnostics for gsplat Compilation
echo ============================================================
echo.

set ENV1=gsplat_env
set ENV2=dust3r-gsplat

if "%1" neq "" set ENV1=%1
if "%2" neq "" set ENV2=%2

echo Comparing: %ENV1% vs %ENV2%
echo.

:: Create output file
set OUTFILE=env_comparison_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%.txt
set OUTFILE=%OUTFILE: =0%
echo Diagnostics Report > %OUTFILE%
echo Generated: %date% %time% >> %OUTFILE%
echo. >> %OUTFILE%

echo ============================================================ >> %OUTFILE%
echo 1. Python and Compiler Versions >> %OUTFILE%
echo ============================================================ >> %OUTFILE%

echo.
echo [1/8] Checking Python versions...
echo. >> %OUTFILE%
echo --- %ENV1% --- >> %OUTFILE%
call conda run -n %ENV1% python --version >> %OUTFILE% 2>&1
echo. >> %OUTFILE%
echo --- %ENV2% --- >> %OUTFILE%
call conda run -n %ENV2% python --version >> %OUTFILE% 2>&1
echo. >> %OUTFILE%

echo [2/8] Checking MSVC compiler versions...
echo ============================================================ >> %OUTFILE%
echo 2. MSVC Compiler Versions >> %OUTFILE%
echo ============================================================ >> %OUTFILE%
echo. >> %OUTFILE%

:: Check cl.exe version
echo --- Checking cl.exe from PATH --- >> %OUTFILE%
where cl >> %OUTFILE% 2>&1
cl 2>&1 | findstr /C:"Compiler Version" >> %OUTFILE%
echo. >> %OUTFILE%

echo [3/8] Checking PyTorch and CUDA versions...
echo ============================================================ >> %OUTFILE%
echo 3. PyTorch and CUDA Configuration >> %OUTFILE%
echo ============================================================ >> %OUTFILE%
echo. >> %OUTFILE%

echo --- %ENV1% --- >> %OUTFILE%
call conda run -n %ENV1% python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'cuDNN version: {torch.backends.cudnn.version()}'); print(f'Build settings: {torch.__config__.show()}')" >> %OUTFILE% 2>&1
echo. >> %OUTFILE%

echo --- %ENV2% --- >> %OUTFILE%
call conda run -n %ENV2% python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'cuDNN version: {torch.backends.cudnn.version()}'); print(f'Build settings: {torch.__config__.show()}')" >> %OUTFILE% 2>&1
echo. >> %OUTFILE%

echo [4/8] Checking installed packages...
echo ============================================================ >> %OUTFILE%
echo 4. Installed Packages >> %OUTFILE%
echo ============================================================ >> %OUTFILE%
echo. >> %OUTFILE%

echo --- %ENV1% (key packages) --- >> %OUTFILE%
call conda run -n %ENV1% pip list | findstr /I "torch gsplat ninja setuptools wheel" >> %OUTFILE% 2>&1
echo. >> %OUTFILE%

echo --- %ENV2% (key packages) --- >> %OUTFILE%
call conda run -n %ENV2% pip list | findstr /I "torch gsplat ninja setuptools wheel" >> %OUTFILE% 2>&1
echo. >> %OUTFILE%

echo [5/8] Checking environment variables...
echo ============================================================ >> %OUTFILE%
echo 5. Environment Variables >> %OUTFILE%
echo ============================================================ >> %OUTFILE%
echo. >> %OUTFILE%

echo CUDA_PATH: %CUDA_PATH% >> %OUTFILE%
echo CUDA_HOME: %CUDA_HOME% >> %OUTFILE%
echo TORCH_CUDA_ARCH_LIST: %TORCH_CUDA_ARCH_LIST% >> %OUTFILE%
echo MAX_JOBS: %MAX_JOBS% >> %OUTFILE%
echo VS version from PATH: >> %OUTFILE%
echo %PATH% | findstr /I "Visual Studio" >> %OUTFILE% 2>&1
echo. >> %OUTFILE%

echo [6/8] Checking CUDA Toolkit...
echo ============================================================ >> %OUTFILE%
echo 6. CUDA Toolkit Version >> %OUTFILE%
echo ============================================================ >> %OUTFILE%
echo. >> %OUTFILE%

nvcc --version >> %OUTFILE% 2>&1
echo. >> %OUTFILE%

nvidia-smi >> %OUTFILE% 2>&1
echo. >> %OUTFILE%

echo [7/8] Testing compilation flags...
echo ============================================================ >> %OUTFILE%
echo 7. Test Compilation (to check actual flags used) >> %OUTFILE%
echo ============================================================ >> %OUTFILE%
echo. >> %OUTFILE%

echo --- Creating test CUDA file --- >> %OUTFILE%
echo #include ^<cuda_runtime.h^> > test_cuda.cu
echo __global__ void test() {} >> test_cuda.cu
echo int main() { return 0; } >> test_cuda.cu

echo --- Compiling with arch 12.0 --- >> %OUTFILE%
nvcc -arch=sm_120 test_cuda.cu -o test_cuda 2>&1 | findstr /I "error warning" >> %OUTFILE%
if exist test_cuda.exe (
    echo Compilation successful >> %OUTFILE%
    del test_cuda.exe
) else (
    echo Compilation failed >> %OUTFILE%
)
del test_cuda.cu 2>nul
echo. >> %OUTFILE%

echo [8/8] Checking gsplat installation status...
echo ============================================================ >> %OUTFILE%
echo 8. gsplat Installation Status >> %OUTFILE%
echo ============================================================ >> %OUTFILE%
echo. >> %OUTFILE%

echo --- %ENV1% --- >> %OUTFILE%
call conda run -n %ENV1% python -c "import gsplat; print(f'gsplat version: {gsplat.__version__}'); print(f'gsplat file: {gsplat.__file__}')" >> %OUTFILE% 2>&1
echo. >> %OUTFILE%

echo --- %ENV2% --- >> %OUTFILE%
call conda run -n %ENV2% python -c "import gsplat; print(f'gsplat version: {gsplat.__version__}'); print(f'gsplat file: {gsplat.__file__}')" >> %OUTFILE% 2>&1 || echo gsplat not installed >> %OUTFILE%
echo. >> %OUTFILE%

echo ============================================================
echo Report saved to: %OUTFILE%
echo ============================================================
type %OUTFILE%
echo.
pause
