# convert_to_svg.ps1
# Converts all .mmd files in the current directory to Word-compatible SVGs
# Requirements: Node.js, @mermaid-js/mermaid-cli (mmdc)
#   Install once: npm install -g @mermaid-js/mermaid-cli

$ErrorActionPreference = "Stop"

# Check mmdc is available
if (-not (Get-Command mmdc -ErrorAction SilentlyContinue)) {
    Write-Error "mmdc not found. Run: npm install -g @mermaid-js/mermaid-cli"
    exit 1
}

$mmdFiles = Get-ChildItem -Filter "*.mmd" -File
if ($mmdFiles.Count -eq 0) {
    Write-Host "No .mmd files found in current directory." -ForegroundColor Yellow
    exit 0
}

$success = 0
$failed  = 0

foreach ($file in $mmdFiles) {
    $out = [System.IO.Path]::ChangeExtension($file.FullName, ".svg")
    Write-Host "Converting: $($file.Name) -> $([System.IO.Path]::GetFileName($out))" -ForegroundColor Cyan

    try {
        # --backgroundColor white -> explicit white bg Word can render
        mmdc `
            --input  $file.FullName `
            --output $out `
            --backgroundColor white 2>&1 | Out-Null

        if (Test-Path $out) {
            Write-Host "  OK" -ForegroundColor Green
            $success++
        } else {
            Write-Host "  FAILED (no output file)" -ForegroundColor Red
            $failed++
        }
    } catch {
        Write-Host "  ERROR: $_" -ForegroundColor Red
        $failed++
    }
}

Write-Host ""
Write-Host "Done. Success: $success  Failed: $failed" -ForegroundColor White
