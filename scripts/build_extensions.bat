@echo off
REM Usage: scripts\build_extensions.bat <path-to-python.exe>
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set DISTUTILS_USE_SDK=1
set NVCC_APPEND_FLAGS=-DWIN32_LEAN_AND_MEAN -DNOMINMAX -std=c++20
set _CL_=/std:c++20 /DWIN32_LEAN_AND_MEAN /DNOMINMAX
set PY=%~1
if "%PY%"=="" set PY=python
set HERE=%~dp0..
%PY% -m pip install "%HERE%\submodules\i3dgs\submodules\simple-knn" --no-build-isolation --no-deps
%PY% -m pip install "%HERE%\submodules\i3dgs\submodules\diff-gaussian-rasterization" --no-build-isolation --no-deps
%PY% -m pip install "%HERE%\submodules\faster-beta-splatting" --no-build-isolation --no-deps
%PY% -m pip install "git+https://github.com/rahul-goel/fused-ssim" --no-build-isolation --no-deps
%PY% -m pip install "git+https://github.com/nv-tlabs/ppisp.git" --no-build-isolation --no-deps
