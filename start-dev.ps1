param()

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$frontend = Join-Path $root "templates"

Write-Host "Starting DataMind backend on http://127.0.0.1:5000"
Start-Process -FilePath "python" -ArgumentList "app.py" -WorkingDirectory $root

Write-Host "Starting DataMind frontend on http://127.0.0.1:5173"
Start-Process -FilePath "npm" -ArgumentList @("run", "dev", "--", "--host", "127.0.0.1", "--port", "5173") -WorkingDirectory $frontend

Write-Host ""
Write-Host "Open:"
Write-Host "  http://127.0.0.1:5173"
Write-Host "Backend:"
Write-Host "  http://127.0.0.1:5000"
