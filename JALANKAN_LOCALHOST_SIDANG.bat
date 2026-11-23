@echo off
title BMI Analyzer - Localhost Sidang Skripsi (23.11.5538)
cd /d "%~dp0"
echo ===============================================================
echo   Menjalankan BMI Analyzer di http://localhost:8501
echo   Browser akan terbuka otomatis dalam beberapa detik.
echo   (Biarkan jendela hitam ini tetap terbuka selama demo)
echo ===============================================================
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --server.address localhost --browser.gatherUsageStats false
pause
