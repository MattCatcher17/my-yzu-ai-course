@echo off

echo ========================================
echo Installing Python modules...
echo ========================================

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo ========================================
echo Installation completed!
echo ========================================

pause