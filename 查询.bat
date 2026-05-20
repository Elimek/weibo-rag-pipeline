@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ╔══════════════════════════════════════╗
echo ║    微博知识库 · 随时查               ║
echo ╚══════════════════════════════════════╝
echo.
echo 直接输入问题 = 语义搜索
echo 特殊命令：
echo   :date 2025-06-15       查某天的微博
echo   :hour 22               锁定到晚上10点
echo   :topic 求职             查某话题
echo   :year 2025             只看某年
echo   :clear                 清除所有筛选
echo   :quit                  退出
echo.
echo 第一次使用？先看 查询手册.html
echo.
python agent_4_query.py -i
pause
