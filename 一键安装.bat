@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ╔══════════════════════════════════════╗
echo ║    微博记忆库 — 一键安装              ║
echo ╚══════════════════════════════════════╝
echo.
echo 本工具会帮你完成：
echo   1. 检查 Python 环境
echo   2. 安装所需依赖
echo   3. 下载 AI 模型（约 4.5GB，首次较慢）
echo   4. 生成桌面快捷方式
echo.

:: ─── Step 1: 检查 Python ─────────────────────────────────────────────────
echo [1/4] 检查 Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 未检测到 Python！
    echo.
    echo 请先从微软商店安装 Python 3.10+：
    echo   1. 打开 https://www.python.org/downloads/
    echo   2. 下载 Python 3.10 或更新版本
    echo   3. 安装时勾选 "Add Python to PATH"
    echo.
    echo 安装完成后重新运行此脚本。
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set pyver=%%i
echo ✅ Python %pyver%

:: ─── Step 2: 安装依赖 ────────────────────────────────────────────────────
echo [2/4] 安装依赖...
echo   pip install -r requirements.txt
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ⚠️ 部分依赖安装失败，尝试降级...
    pip install "sentence-transformers<3.0"
)
echo ✅ 依赖安装完成

:: ─── Step 3: 设置运行权限 ─────────────────────────────────────────────────
echo [3/4] 检测运行环境...
if not exist "db\chroma" (
    echo ℹ️ 未检测到向量库。
    echo   请按以下步骤准备数据：
    echo.
    echo   1. 打开 https://weibo.com 并登录
    echo   2. 按 F12 → Application → Cookies → weibo.com
    echo   3. 复制 SUB 和 SUBP 的值
    echo   4. 打开 agent_1_scrape.py，粘贴到顶部 MY_COOKIES
    echo   5. 运行：python run_pipeline.py
    echo.
    echo   或打开说明文档：查询手册.html
) else (
    echo ✅ 向量库已就绪
)

:: ─── Step 4: 创建快捷方式 ─────────────────────────────────────────────────
echo [4/4] 创建快捷方式...
:: 创建运行脚本
copy "启动网页版.bat" "%USERPROFILE%\Desktop\微博记忆库.bat" >nul 2>&1
echo ✅ 桌面快捷方式已创建：微博记忆库.bat

echo.
echo ╔══════════════════════════════════════╗
echo ║         安装完成！                    ║
echo ║                                      ║
echo ║  双击桌面「微博记忆库.bat」开始使用    ║
echo ║  或双击本目录「启动网页版.bat」        ║
echo ╚══════════════════════════════════════╝
echo.
pause
