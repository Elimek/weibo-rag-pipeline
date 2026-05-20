@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 🧠 微博记忆库 — 网页版启动中...
echo.
echo 浏览器将自动打开 http://localhost:8787
echo 关闭此窗口即可停止服务
echo.
python app.py
pause
