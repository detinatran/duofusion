# Windows: fit + evaluate every experiment, then print the comparison table.
#   powershell -ExecutionPolicy Bypass -File run_experiments.ps1
#   powershell -ExecutionPolicy Bypass -File run_experiments.ps1 baseline full
# "Continue": PowerShell 5.1 turns any stderr line of a native program (e.g. a
# numpy warning) into an error record, which "Stop" would treat as fatal.
# Failures are detected from the exit code instead.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$py = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }
$names = if ($args.Count -gt 0) { $args } else { @("baseline","c1_tr","c15_units","c152_delay","c2_delay_only","full_ref","c3_reffree","full","full_aif_ref","full_aif_wm","full_aif","full_idreg","full_idvox","full_aif_idreg","full_aif_idvox","full_idvox_moment","full_idvox_osvd","full_aif_idvox_moment","full_aif_idvox_osvd") }
foreach ($n in $names) {
    & $py scripts\run_cohort.py "configs\$n.yaml"
    if ($LASTEXITCODE -ne 0) { Write-Output "run_cohort failed for $n"; exit 1 }
    & $py scripts\evaluate.py "configs\$n.yaml"
    if ($LASTEXITCODE -ne 0) { Write-Output "evaluate failed for $n"; exit 1 }
}
& $py scripts\compare.py --save outputs\comparison.md
