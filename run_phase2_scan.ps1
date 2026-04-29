$ErrorActionPreference = "Stop"
$poetryVenv = (poetry env info -p 2>$null)
$pythonExe = Join-Path $poetryVenv "Scripts\python.exe"
& $pythonExe sector_scan.py --phase 2 --top-n 3 --llm-top 3 --output quant_archive/2026-04/scan_full_phase2.md
