@echo off
chcp 65001 >nul
setlocal

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo ============================================================
echo  Yelp Checkin DB Auto-build
echo ============================================================
echo.

echo [1/1] checkin table...
python checkin_extractor.py
if %errorlevel% neq 0 (
    echo [FAILED] checkin_extractor.py
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
