@echo off
setlocal
title OR-SterileAssist v2 - Medical Workstation (PySide6)
color 0f
cd /d "%~dp0"

if not exist "%~dp0.venv\Scripts\python.exe" (
    color 0c
    echo ===============================================================================
    echo [PERINGATAN] Virtual Environment (.venv) belum ditemukan!
    echo ===============================================================================
    echo.
    echo Aplikasi belum siap karena dependensi belum diinstall di laptop ini.
    echo.
    echo Silakan ikuti langkah berikut:
    echo 1. Tutup jendela ini.
    echo 2. Dobel klik file "1_SETUP_DEPENDENCIES.bat" di folder ini.
    echo 3. Tunggu hingga proses setup dependensi selesai (ada pesan sukses).
    echo 4. Jalankan kembali file "2_JALANKAN_APLIKASI_V2.bat" ini.
    echo.
    echo ===============================================================================
    pause
    exit /b 1
)

echo ===============================================================================
echo      Memulai OR-SterileAssist v2 (PySide6 Medical Workstation - ETS HMI)
echo ===============================================================================
echo [INFO] Menjalankan aplikasi dengan Python Virtual Environment (.venv)...
echo [INFO] Jangan menutup jendela konsol ini jika Anda ingin melihat log sistem.
echo.

"%~dp0.venv\Scripts\python.exe" "%~dp0src\or_sterile_assist_v2.py"

if %errorlevel% neq 0 (
    echo.
    echo ===============================================================================
    echo [INFO] Aplikasi telah ditutup (kode keluar: %errorlevel%).
    echo ===============================================================================
    pause
)
exit /b 0
