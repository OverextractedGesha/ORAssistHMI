@echo off
setlocal EnableDelayedExpansion
title OR-SterileAssist - Setup Dependensi Otomatis
color 0b

echo ===============================================================================
echo          OR-SterileAssist: Setup Virtual Environment ^& Dependensi
echo ===============================================================================
echo.
echo Sedang memeriksa instalasi Python di sistem Windows Anda...
echo.

set "PY_CMD="

:: 1. Periksa python
python --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PY_CMD=python"
    goto :PYTHON_FOUND
)

:: 2. Periksa py launcher
py --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PY_CMD=py"
    goto :PYTHON_FOUND
)

:PYTHON_NOT_FOUND
color 0c
echo -------------------------------------------------------------------------------
echo [ERROR] Python tidak terdeteksi di komputer ini!
echo -------------------------------------------------------------------------------
echo.
echo Aplikasi ini membutuhkan Python 3.10 atau versi yang lebih baru.
echo.
echo Langkah mudah memperbaikinya:
echo 1. Kunjungi situs resmi: https://www.python.org/downloads/
echo 2. Download dan buka installer Python (misal Python 3.11 atau 3.12).
echo 3. SANGAT PENTING: Pada layar pertama installer, centang kotak:
echo    [X] "Add python.exe to PATH" (atau "Add Python to environment variables")
echo 4. Klik "Install Now" hingga selesai.
echo 5. Buka kembali file "1_SETUP_DEPENDENCIES.bat" ini.
echo.
echo -------------------------------------------------------------------------------
pause
exit /b 1

:PYTHON_FOUND
echo [OK] Python terdeteksi:
%PY_CMD% --version
echo.

:: 3. Pembuatan virtual environment .venv
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo [Langkah 1/3] Membuat Virtual Environment lokal (.venv)...
    echo Mohon tunggu sebentar...
    "%PY_CMD%" -m venv "%~dp0.venv"
    if %errorlevel% neq 0 (
        color 0c
        echo.
        echo [ERROR] Gagal membuat virtual environment di "%~dp0.venv".
        echo Pastikan folder ini memiliki izin tulis (bukan di dalam file zip atau Program Files).
        pause
        exit /b 1
    )
    echo [OK] Virtual Environment berhasil dibuat di folder .venv.
) else (
    echo [Langkah 1/3] Virtual Environment (.venv) sudah ada, melanjutkan...
)
echo.

:: 4. Upgrade pip
echo [Langkah 2/3] Memperbarui pip ke versi terbaru...
"%~dp0.venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
if %errorlevel% neq 0 (
    echo [INFO] Melewati upgrade pip, langsung lanjut install requirements...
) else (
    echo [OK] pip siap digunakan.
)
echo.

:: 5. Install requirements
echo [Langkah 3/3] Mengunduh dan menginstall dependensi (requirements.txt)...
echo (Proses ini memerlukan koneksi internet aktif. Estimasi: 1 - 3 menit)
echo.
"%~dp0.venv\Scripts\pip.exe" install -r "%~dp0requirements.txt"
if %errorlevel% neq 0 (
    color 0c
    echo.
    echo -------------------------------------------------------------------------------
    echo [PERINGATAN] Terjadi kendala saat mengunduh beberapa dependensi.
    echo Periksa koneksi internet Anda, lalu jalankan ulang file batch ini.
    echo -------------------------------------------------------------------------------
    pause
    exit /b 1
)

echo.
color 0a
echo ===============================================================================
echo                         SETUP BERHASIL SELESAI!
echo ===============================================================================
echo.
echo Semua library dan model offline sudah siap digunakan.
echo.
echo Sekarang Anda bisa langsung menjalankan aplikasi melalui salah satu file berikut:
echo.
echo 1. "2_JALANKAN_APLIKASI.bat"
echo    -^> Menjalankan aplikasi dengan jendela terminal (bagus untuk melihat log).
echo.
echo 2. "3_JALANKAN_TANPA_CMD.vbs"
echo    -^> Menjalankan aplikasi secara langsung tanpa jendela hitam terminal (bersih).
echo.
echo Petunjuk gestur ^& perintah suara lengkap ada di:
echo    "PETUNJUK_PENGGUNAAN_TEMAN.txt"
echo ===============================================================================
echo.
pause
exit /b 0
