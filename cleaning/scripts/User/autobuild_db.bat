@echo off
chcp 65001 >nul
setlocal

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo ============================================================
echo  Yelp User DB Auto-build
echo ============================================================
echo.

echo [1/3] user table...
python user_extractor.py
if %errorlevel% neq 0 (
    echo [FAILED] user_extractor.py
    goto :error
)
echo.

echo [2/3] user_friends table...
python user_friends_extractor.py
if %errorlevel% neq 0 (
    echo [FAILED] user_friends_extractor.py
    goto :error
)
echo.

echo [3/3] user_elite table...
python user_elite_extractor.py
if %errorlevel% neq 0 (
    echo [FAILED] user_elite_extractor.py
    goto :error
)
echo.

echo ============================================================
echo  All tables rebuilt successfully.
echo ============================================================
goto :end

:error
echo.
echo ============================================================
echo  Rebuild stopped due to error. Check logs above.
echo ============================================================

:end
pause
