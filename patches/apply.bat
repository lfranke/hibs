@echo off
REM Usage: patches\apply.bat <patch-name> [-R]
setlocal
set HERE=%~dp0
if "%~1"=="" (echo Usage: patches\apply.bat ^<patch-name^> [-R] & exit /b 1)
git -C "%HERE%..\submodules\i3dgs" apply %2 "%HERE%%~1.diff"
if errorlevel 1 (echo FAILED & exit /b 1)
echo applied %~1 %2
