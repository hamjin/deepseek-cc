param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$Install,
    [switch]$Reload
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$EnvFile = Join-Path $ProjectRoot ".env"

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host "Creating virtual environment at .venv"
    python -m venv (Join-Path $ProjectRoot ".venv")
}

if ($Install) {
    Write-Host "Installing project dependencies"
    & $Python -m pip install -e "$ProjectRoot[dev]"
}

if (-not (Test-Path -LiteralPath $EnvFile)) {
    Write-Warning ".env not found. Defaults will be used unless you copy .env.example to .env."
}

$UvicornArgs = @(
    "-m",
    "uvicorn",
    "app.main:app",
    "--host",
    $BindHost,
    "--port",
    "$Port"
)

if ($Reload) {
    $UvicornArgs += "--reload"
}

Write-Host "Starting DeepSeek Anthropic Thinking Repair Proxy"
Write-Host "URL: http://${BindHost}:$Port"
Write-Host "Stop: Ctrl+C"

Push-Location $ProjectRoot
try {
    & $Python @UvicornArgs
}
finally {
    Pop-Location
}
