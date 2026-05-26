@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 🧠 微博记忆库 v2 — Gradio 网页版
echo.
echo 浏览器将自动打开 http://localhost:7860
echo 支持「去年今日」语义搜索和 Critic 可信度审核
echo 关闭此窗口即可停止服务
echo.
python gradio_app.py
pause
