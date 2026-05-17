@echo off
chcp 65001 >nul
setlocal

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo ============================================================
echo  Yelp Business DB Auto-build
echo ============================================================
echo.

echo [1/4] business table...
python business_extractor.py
if %errorlevel% neq 0 (
    echo [FAILED] business_extractor.py
    goto :error
)
echo.

echo [2/4] category + business_category table...
python business_category_extractor.py
if %errorlevel% neq 0 (
    echo [FAILED] business_category_extractor.py
    goto :error
)
echo.

echo [3/4] business_hours table...
python business_hours_extractor.py
if %errorlevel% neq 0 (
    echo [FAILED] business_hours_extractor.py
    goto :error
)
echo.

echo [4/4] attribute_definition + business_attribute table...
python business_attributes_extractor.py
if %errorlevel% neq 0 (
    echo [FAILED] business_attributes_extractor.py
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
