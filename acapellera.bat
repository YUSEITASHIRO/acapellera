@echo off
title Acapellera
rem Acapellera: drop a song on the page that opens, CHIMERA sings it a cappella. Ctrl+C stops it.
rem   acapellera.bat            the worker runs on this PC
rem   acapellera.bat --remote   the worker runs on the server in acapellera\remote.json (see acapellera\README.md)
python "%~dp0acapellera\launch.py" %* <nul
pause
