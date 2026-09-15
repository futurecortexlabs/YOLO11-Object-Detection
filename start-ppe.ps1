$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PORT = '7862'
& "$PSScriptRoot/.venv/Scripts/python.exe" "$PSScriptRoot/app.py"
