<#
.SYNOPSIS
Local part of the repo rename (see repo-rename-procedure.md).

.DESCRIPTION
Renames the old pipeline's folder to old-tower-processing and the rewrite's folder to
ttu-tower-processing, swaps the matching Claude project directories, and repoints the old
clone's origin to old-tower-processing.git.

Dry run unless -Execute is given. -Rollback reverses a completed run.
Run from a copy outside both repos, with VS Code, terminals, Jupyter kernels and the Claude
desktop app closed.
#>
param(
    [switch]$Execute,
    [switch]$Rollback
)

$ErrorActionPreference = 'Stop'

$Code     = 'C:\Users\ellwalke\Code'
$Projects = 'C:\Users\ellwalke\.claude\projects'
$GitHub   = 'https://github.com/Intergalactyc'

$OldOriginBefore = "$GitHub/ttu-tower-processing.git"
$OldOriginAfter  = "$GitHub/old-tower-processing.git"

# Order matters: each repo folder / project dir frees its name before the next one takes it.
$Moves = @(
    @{ From = "$Code\ttu-tower-processing"; To = "$Code\old-tower-processing" },
    @{ From = "$Code\new-tower-processing"; To = "$Code\ttu-tower-processing" },
    @{ From = "$Projects\C--Users-ellwalke-Code-ttu-tower-processing"; To = "$Projects\C--Users-ellwalke-Code-old-tower-processing" },
    @{ From = "$Projects\C--Users-ellwalke-Code-new-tower-processing"; To = "$Projects\C--Users-ellwalke-Code-ttu-tower-processing" }
)

function Fail([string]$Message) {
    Write-Host "ABORT: $Message" -ForegroundColor Red
    exit 1
}

# Windows PowerShell 5.1 turns redirected native stderr into terminating errors under 'Stop'
function Invoke-Git {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = & git @args 2>&1 | ForEach-Object { "$_" }
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prev
    }
    [pscustomobject]@{ Code = $code; Out = ($out -join "`n").Trim() }
}

function Move-All($Pairs) {
    $done = New-Object System.Collections.ArrayList
    foreach ($m in $Pairs) {
        if (-not $Execute) {
            Write-Host "would move $($m.From) -> $($m.To)"
            continue
        }
        try {
            if (Test-Path -LiteralPath $m.To) { throw "destination already exists: $($m.To)" }
            Move-Item -LiteralPath $m.From -Destination $m.To
            [void]$done.Add($m)
            Write-Host "moved $($m.From) -> $($m.To)"
        } catch {
            Write-Host "FAILED: $($_.Exception.Message)" -ForegroundColor Red
            for ($i = $done.Count - 1; $i -ge 0; $i--) {
                Move-Item -LiteralPath $done[$i].To -Destination $done[$i].From
                Write-Host "undid $($done[$i].From) -> $($done[$i].To)"
            }
            Fail "no moves left in place; close whatever holds the folder and rerun"
        }
    }
}

function Set-Origin([string]$Repo, [string]$Url) {
    if (-not $Execute) {
        Write-Host "would set origin of $Repo -> $Url"
        return
    }
    [void](Invoke-Git -C $Repo remote set-url origin $Url)
    $r = Invoke-Git -C $Repo remote get-url origin
    if ($r.Out -ne $Url) {
        Write-Host "WARNING: origin of $Repo is '$($r.Out)'; set it to $Url manually" -ForegroundColor Yellow
    } else {
        Write-Host "origin of $Repo -> $Url"
    }
}

if ($Rollback) {
    $Pairs = @()
    for ($i = $Moves.Count - 1; $i -ge 0; $i--) {
        $Pairs += @{ From = $Moves[$i].To; To = $Moves[$i].From }
    }
    $OldRepoAfterMoves = "$Code\ttu-tower-processing"
    $OriginAfterMoves  = $OldOriginBefore
} else {
    $Pairs = $Moves
    $OldRepoAfterMoves = "$Code\old-tower-processing"
    $OriginAfterMoves  = $OldOriginAfter
}

# --- preflight ---

$cwd = (Get-Location).Path
foreach ($m in $Pairs) {
    if ($cwd.StartsWith($m.From, [System.StringComparison]::OrdinalIgnoreCase)) {
        Fail "current directory is inside $($m.From); cd outside it first"
    }
    if (-not (Test-Path -LiteralPath $m.From)) { Fail "missing: $($m.From)" }
}

if ($Rollback) {
    foreach ($p in "$Code\new-tower-processing", "$Projects\C--Users-ellwalke-Code-new-tower-processing") {
        if (Test-Path -LiteralPath $p) { Fail "already exists: $p" }
    }
    $r = Invoke-Git -C "$Code\ttu-tower-processing" remote
    if ($r.Out) {
        Write-Host "WARNING: the rewrite already has a remote ($($r.Out)), so step 6 ran. Delete or rename the new GitHub repo, and rename old-tower-processing back on GitHub, by hand." -ForegroundColor Yellow
    }
} else {
    foreach ($p in "$Code\old-tower-processing", "$Projects\C--Users-ellwalke-Code-old-tower-processing") {
        if (Test-Path -LiteralPath $p) { Fail "already exists: $p" }
    }
    $r = Invoke-Git -C "$Code\ttu-tower-processing" remote get-url origin
    if ($r.Code -ne 0 -or $r.Out -ne $OldOriginBefore) {
        Fail "old repo origin is '$($r.Out)', expected $OldOriginBefore"
    }
    $r = Invoke-Git -C "$Code\ttu-tower-processing" status --porcelain
    if ($r.Out) { Fail "old repo has uncommitted changes" }
    $r = Invoke-Git -C "$Code\new-tower-processing" remote
    if ($r.Out) { Fail "the rewrite already has a remote ($($r.Out)); this procedure assumes none" }
    $r = Invoke-Git -C "$Code\ttu-tower-processing" tag --list v1.0.0
    if (-not $r.Out) { Write-Host "warning: old repo has no v1.0.0 tag (procedure step 0)" -ForegroundColor Yellow }
    $r = Invoke-Git ls-remote $OldOriginAfter HEAD
    if ($r.Code -ne 0) { Fail "can't reach $OldOriginAfter - rename the repo on GitHub first (step 2)" }
}

if (-not $Execute) { Write-Host "DRY RUN - pass -Execute to apply" -ForegroundColor Cyan }

# --- moves, then origin ---

Move-All $Pairs
Set-Origin $OldRepoAfterMoves $OriginAfterMoves

if ($Execute -and -not $Rollback) {
    Write-Host ""
    Write-Host "Done. Next: steps 4-8 of repo-rename-procedure.md (consumers, venvs, GitHub repo + push, memory pass, verify)." -ForegroundColor Green
}
