param(
    [Parameter(Mandatory = $true)]
    [string]$Version
)

$ErrorActionPreference = 'Stop'

$root = Resolve-Path (Join-Path $PSScriptRoot '..')
$releaseNotes = Join-Path $root.Path "RELEASE_NOTES_v$Version.md"

function Replace-InFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Pattern,
        [Parameter(Mandatory = $true)]
        [string]$Replacement
    )

    $content = Get-Content -LiteralPath $Path -Raw
    $updated = [regex]::Replace($content, $Pattern, $Replacement)
    if ($updated -ne $content) {
        Set-Content -LiteralPath $Path -Value $updated -NoNewline
    }
}

Write-Host "Bumping workspace to $Version"

Replace-InFile -Path (Join-Path $root.Path 'Cargo.toml') -Pattern 'version = "0\.7\.1"' -Replacement "version = `"$Version`""
Replace-InFile -Path (Join-Path $root.Path 'threecrate-umbrella\Cargo.toml') -Pattern '0\.7\.1' -Replacement $Version
Replace-InFile -Path (Join-Path $root.Path 'examples\Cargo.toml') -Pattern '0\.7\.0' -Replacement $Version
Replace-InFile -Path (Join-Path $root.Path 'README.md') -Pattern '0\.7\.1' -Replacement $Version
Replace-InFile -Path (Join-Path $root.Path '.github\ISSUE_TEMPLATE\bug_report.md') -Pattern '0\.7\.1' -Replacement $Version

if (-not (Test-Path -LiteralPath $releaseNotes)) {
@"
# v$Version Release Notes

## Highlights

- TODO: summarize the major robotics improvements.
- TODO: summarize any API changes or notable fixes.

## Crates

All crates are bumped to `$Version`.

## Release checklist

- [ ] Fill in the highlights above.
- [ ] Run `just release-prep $Version`.
- [ ] Create the GitHub release tag `v$Version`.
"@ | Set-Content -LiteralPath $releaseNotes
    Write-Host "Created $releaseNotes"
} else {
    Write-Host "Release notes already exist: $releaseNotes"
}

Write-Host "Done. Review the diff, then run the release checks and create the GitHub release."
