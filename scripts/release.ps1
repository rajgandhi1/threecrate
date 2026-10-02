param(
    [Parameter(Mandatory = $true)]
    [string]$Version
)

# Bump every crate, the umbrella crate and the docs to $Version, and scaffold
# release notes. Run it with `just release-prep <version>`, then review the diff.
#
# The current version is read from [workspace.package] in Cargo.toml, so the
# script works for any release without editing it first.

$ErrorActionPreference = 'Stop'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

function Read-Text([string]$Path) {
    return [System.IO.File]::ReadAllText($Path, $utf8NoBom)
}

# Write as UTF-8 without a BOM, keeping the file's line endings and any
# non-ASCII characters (Set-Content in Windows PowerShell would not).
function Write-Text([string]$Path, [string]$Text) {
    [System.IO.File]::WriteAllText($Path, $Text, $utf8NoBom)
}

function Update-File([string]$Path, [string]$Pattern, [string]$Replacement) {
    $content = Read-Text $Path
    $updated = [regex]::Replace($content, $Pattern, $Replacement)
    if ($updated -ne $content) {
        Write-Text $Path $updated
        Write-Host "  updated $($Path.Substring($root.Length + 1))"
    }
}

$workspaceToml = Join-Path $root 'Cargo.toml'
$workspaceText = Read-Text $workspaceToml
$match = [regex]::Match($workspaceText, '(?ms)^\[workspace\.package\][^\[]*?^version\s*=\s*"([^"]+)"')
if (-not $match.Success) {
    throw 'Could not find the version in [workspace.package] of Cargo.toml'
}
$old = $match.Groups[1].Value
if ($old -eq $Version) {
    throw "The workspace is already at $Version"
}
$oldRe = [regex]::Escape($old)
Write-Host "Bumping $old -> $Version"

# 1. Workspace version (all crates inherit it, and so does the PyPI package).
Update-File $workspaceToml "(?ms)(^\[workspace\.package\][^\[]*?^version\s*=\s*`")$oldRe(`")" "`${1}$Version`${2}"

# 2. The umbrella crate lives outside the workspace and has its own version.
$umbrellaToml = Join-Path $root 'threecrate-umbrella\Cargo.toml'
Update-File $umbrellaToml "(?ms)(^\[package\][^\[]*?^version\s*=\s*`")$oldRe(`")" "`${1}$Version`${2}"

# 3. Version pins on internal dependencies in every manifest, e.g.
#    threecrate-core = { path = "../threecrate-core", version = "0.8.0" }
$manifests = Get-ChildItem -Path $root -Filter Cargo.toml -Recurse |
    Where-Object { $_.FullName -notmatch '[\\/]target[\\/]' }
foreach ($manifest in $manifests) {
    Update-File $manifest.FullName "(?m)^(threecrate[-\w]*\s*=\s*\{[^\r\n]*version\s*=\s*`")$oldRe(`")" "`${1}$Version`${2}"
}

# 4. Docs that tell users which version to install or report.
$docs = @(
    'README.md',
    'docs\installation.md',
    '.github\ISSUE_TEMPLATE\bug_report.md'
)
foreach ($doc in $docs) {
    Update-File (Join-Path $root $doc) $oldRe $Version
}

# 5. Release notes scaffold.
$releaseNotes = Join-Path $root "RELEASE_NOTES_v$Version.md"
if (-not (Test-Path -LiteralPath $releaseNotes)) {
    $notes = @"
# v$Version Release Notes

## Highlights

- TODO: summarize the major changes.

## Changes that may affect you

- TODO: list behavior or API changes.

## Crates

All crates and the PyPI package are bumped to ``$Version``.
"@
    Write-Text $releaseNotes ($notes + "`n")
    Write-Host "Created $($releaseNotes.Substring($root.Length + 1))"
} else {
    Write-Host "Release notes already exist: $releaseNotes"
}

# 6. Anything still mentioning the old version needs a manual look.
Write-Host ''
Write-Host "Remaining mentions of $old (check these by hand):"
$leftover = & git -C $root grep -n -F "$old" -- '*.toml' '*.md' ':!Cargo.lock' ':!RELEASE_NOTES_v*' 2>$null
if ($leftover) { $leftover | ForEach-Object { Write-Host "  $_" } } else { Write-Host '  none' }

Write-Host ''
Write-Host 'Done. Review the diff, fill in the release notes, and open a PR.'
