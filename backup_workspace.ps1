# OpenClaw 工作区一键备份脚本
# 双击运行，即可在桌面生成带日期的备份文件

$date = Get-Date -Format "yyyy-MM-dd"
$backupDir = [Environment]::GetFolderPath("Desktop")
$backupPath = Join-Path $backupDir "openclaw_workspace_$date.zip"

$workspacePath = "$env:USERPROFILE\.qclaw\workspace-agent-b9c8dcea"

if (Test-Path $workspacePath) {
    Compress-Archive -Path $workspacePath -DestinationPath $backupPath -Force
    Write-Host "备份完成: $backupPath"
} else {
    Write-Host "工作区路径不存在: $workspacePath"
}

# 保持窗口
Read-Host "按回车退出"
