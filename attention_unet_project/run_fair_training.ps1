param(
  [int]$Epochs = 20,
  [string]$RunName = "attention_unet_fair20_fixed_postprocess_seed42",
  [int]$MaxTrainSamples = 0,
  [int]$MaxValSamples = 0
)

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $Root
$RunDir = Join-Path $Root "runs\$RunName"
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Dataset = Join-Path $RepoRoot "Dataset"
$OutLog = Join-Path $RunDir "train.out.log"
$ErrLog = Join-Path $RunDir "train.err.log"

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
Set-Location $Root

$ArgsList = @(
  "-m", "attention_unet.train",
  "--data-dir", "$Dataset",
  "--epochs", "$Epochs",
  "--batch-size", "4",
  "--lr", "0.001",
  "--base-channels", "16",
  "--image-size", "256,256",
  "--augment", "basic",
  "--loss", "combo",
  "--loss-dice-weight", "1.0",
  "--loss-bce-weight", "1.0",
  "--loss-focal-weight", "0.5",
  "--scheduler", "plateau",
  "--threshold-min", "0.12",
  "--threshold-max", "0.50",
  "--threshold-steps", "20",
  "--early-stopping-patience", "8",
  "--postprocess",
  "--post-open-iters", "1",
  "--output-dir", "$RunDir"
)

if ($MaxTrainSamples -gt 0) {
  $ArgsList += @("--max-train-samples", "$MaxTrainSamples")
}
if ($MaxValSamples -gt 0) {
  $ArgsList += @("--max-val-samples", "$MaxValSamples")
}

function Quote-CmdArg([string]$Value) {
  return '"' + ($Value -replace '"', '\"') + '"'
}

$CommandLine = (Quote-CmdArg $Python) + " " + (($ArgsList | ForEach-Object { Quote-CmdArg $_ }) -join " ") +
  " 1> " + (Quote-CmdArg $OutLog) + " 2> " + (Quote-CmdArg $ErrLog)

Write-Host "Starting Attention U-Net training..."
Write-Host "Run dir: $RunDir"
Write-Host "Stdout log: $OutLog"
Write-Host "Stderr/progress log: $ErrLog"
Write-Host "To watch progress in another PowerShell:"
Write-Host "  Get-Content `"$ErrLog`" -Wait"

& cmd.exe /d /c $CommandLine
$ExitCode = $LASTEXITCODE

if ($ExitCode -ne 0) {
  Write-Error "Training exited with code $ExitCode. Check logs above."
  exit $ExitCode
}

Write-Host "Training completed successfully."
