@echo off
title Machine Assistant - Bühler MEAG
cd /d "%~dp0"

echo ===================================================
echo [1/2] Menyalakan Server AI (llama-server)...
echo ===================================================
start "Llama Server Background" /min cmd /c "cd llama-b11944 && llama-server.exe -m qwen2.5-1.5b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 8080 -c 4096"

echo Menunggu server AI siap (5 detik)...
timeout /t 5 /nobreak >nul

echo.
echo ===================================================
echo [2/2] Membuka Sistem Asisten Teknisi...
echo ===================================================
cd code
call .venv\Scripts\activate

cls
echo ======================================================================
echo    ASISTEN TROUBLESHOOTING MESIN BÜHLER MEAG (SIAP DIGUNAKAN)
echo ======================================================================
echo  Ketik pertanyaan Anda langsung di bawah ini.
echo  Ketik 'keluar' atau 'exit' untuk menutup program.
echo ======================================================================
echo.

:tanya_lagi
set "pertanyaan="
set /p "pertanyaan=>> Masukkan Pertanyaan: "

if "%pertanyaan%"=="" goto tanya_lagi
if /i "%pertanyaan%"=="exit" goto selesai
if /i "%pertanyaan%"=="keluar" goto selesai

echo.
ma ask "%pertanyaan%"
echo.
echo ----------------------------------------------------------------------
goto tanya_lagi

:selesai
echo.
echo Menutup sistem dan mematikan server AI...
taskkill /f /im llama-server.exe >nul 2>&1
echo Selesai. Sampai jumpa!
timeout /t 2 >nul
