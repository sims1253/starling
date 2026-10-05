# Run a packaged Windows executable outside its build environment
# (RUNTIME.md "Windows archive check"). Windows PowerShell 5.1 compatible.
#
# The archive is extracted into a fresh directory and every child process runs
# with a scrubbed PATH: the Windows system directories plus -RuntimePath only.
# A CUDA archive therefore resolves cudart/cuBLAS from the documented runtime
# DLL directories, never from an installed toolkit or the build tree. Installed
# SDKs elsewhere on the machine are not removed, so this is a scrubbed-PATH
# check, not a clean-VM check; the loaded-module report below shows which DLL
# files the process actually used.
#
# Startup (always): checksum, version, backend flavor, ABI.
# Inference (-Gguf/-Audio/-Expected): starts the server on a Parakeet GGUF,
# requires /health to report a device of the archive's backend, posts one
# 16 kHz mono PCM16 WAV, and requires exactly the expected transcript with no
# accelerator-rejected node (STARLING_SCHED_DEBUG=1, #184).
#
# Example (CUDA, runtime DLLs from NVIDIA's redistributable archives):
#   powershell -ExecutionPolicy Bypass -File check-windows-archive.ps1 `
#     -Archive starling-serve-windows-cuda.zip -Version 0.1.0 -Abi 8 `
#     -RuntimePath C:\cuda\cudart\bin\x64,C:\cuda\cublas\bin\x64 `
#     -Gguf parakeet-tdt-0.6b-v3-q8_0.gguf -Audio jfk.wav -Expected "..."
param(
    [Parameter(Mandatory = $true)][string]$Archive,
    [Parameter(Mandatory = $true)][string]$Version,
    [Parameter(Mandatory = $true)][string]$Abi,
    [string[]]$RuntimePath = @(),
    [string]$Gguf,
    [string]$Audio,
    [string]$Expected,
    [int]$Port = 18188
)
$ErrorActionPreference = 'Stop'
# powershell -File passes "a,b" as one string; accept ',' or ';' separators.
$RuntimePath = @($RuntimePath | ForEach-Object { $_ -split '[;,]' } | Where-Object { $_ } |
    ForEach-Object { (Resolve-Path $_).Path.TrimEnd('\') })

$archivePath = (Resolve-Path $Archive).Path
$name = [IO.Path]::GetFileNameWithoutExtension($archivePath)
if ($name -notmatch '^starling-serve-windows-(cuda|vulkan|cpu)$') {
    throw "Unexpected archive name: $name"
}
$flavor = $Matches[1]
$deviceIs = @{ cuda = '^CUDA\d+$'; vulkan = '^Vulkan\d+$'; cpu = '^(?i:cpu)$' }[$flavor]

$work = Join-Path ([IO.Path]::GetTempPath()) ("starling-check-" + [guid]::NewGuid())
New-Item -ItemType Directory $work | Out-Null
Expand-Archive -LiteralPath $archivePath -DestinationPath $work
$exe = Join-Path $work "$name.exe"
if (-not (Test-Path (Join-Path $work 'RUNTIME.md'))) { throw 'RUNTIME.md missing from archive' }
$want = ((Get-Content (Join-Path $work "$name.sha256") -Raw).Trim() -split '\s+')[0].ToLower()
$have = (Get-FileHash $exe -Algorithm SHA256).Hash.ToLower()
if ($have -ne $want) { throw "Checksum mismatch: $have != $want" }
"checksum OK: $have"

$system = "$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\Wbem"
$env:PATH = (@($system) + $RuntimePath) -join ';'
"PATH=$env:PATH"

$versionText = & $exe --version
if ($LASTEXITCODE -ne 0) { throw "--version exited $LASTEXITCODE" }
$versionText
if ($versionText -notcontains "starling-serve $Version") { throw "Expected version line: starling-serve $Version" }
if ($versionText -notcontains "backend: $flavor") { throw "Expected backend line: backend: $flavor" }
$abiText = (& $exe --abi-version | Out-String).Trim()
"ABI: $abiText"
if ($abiText -ne $Abi) { throw "Expected ABI $Abi, got $abiText" }

if (-not $Gguf) { "startup check passed (no inference requested)"; exit 0 }
if (-not $Audio -or -not $Expected) { throw '-Gguf needs -Audio and -Expected' }

$env:STARLING_SCHED_DEBUG = '1'
$log = Join-Path $work 'server.log'
$err = Join-Path $work 'server.err.log'
$proc = Start-Process -FilePath $exe -PassThru -NoNewWindow `
    -RedirectStandardOutput $log -RedirectStandardError $err `
    -ArgumentList @('--model', 'parakeet', '--gguf', "`"$((Resolve-Path $Gguf).Path)`"",
                    '--host', '127.0.0.1', '--port', "$Port")
$base = "http://127.0.0.1:$Port"
try {
    $health = $null
    for ($i = 0; $i -lt 600; $i++) {
        if ($proc.HasExited) { break }
        try { $health = Invoke-RestMethod "$base/health" -TimeoutSec 5 } catch { $health = $null }
        if ($health -and $health.loaded) { break }
        Start-Sleep -Milliseconds 500
    }
    if (-not $health -or -not $health.loaded) {
        Get-Content $log, $err -ErrorAction SilentlyContinue | Select-Object -Last 40
        throw 'Model did not load'
    }
    "health: " + ($health | ConvertTo-Json -Compress)
    if ($health.backend -notmatch $deviceIs) { throw "Expected a $flavor device, got $($health.backend)" }

    # curl.exe ships in System32 on Windows 10 1803+ and Windows 11.
    $response = & curl.exe -sS -w "`n%{http_code}" -F model=parakeet `
        -F "file=@$((Resolve-Path $Audio).Path);type=audio/wav" "$base/v1/audio/transcriptions"
    $status = $response[-1]
    $body = ($response[0..($response.Count - 2)] -join "`n")
    "HTTP $status $body"
    if ($status -ne '200') { throw "Transcription failed with HTTP $status" }
    $text = ($body | ConvertFrom-Json).text
    if ($text -cne $Expected) { throw "Transcript mismatch: '$text' != '$Expected'" }
    if ($proc.HasExited) { throw 'Server died during transcription' }

    # Which runtime and driver DLLs the server actually loaded, and from where.
    $modules = (Get-Process -Id $proc.Id).Modules |
        Where-Object { $_.ModuleName -match '^(vcomp|cudart|cublas|nvcuda|vulkan-1|nvoglv|amdvlk|igvk)' } |
        Sort-Object ModuleName
    $modules | ForEach-Object { "loaded: $($_.FileName) $($_.FileVersionInfo.FileVersion)" }
    if ($flavor -eq 'cuda') {
        # The CUDA runtime (cudart) is linked statically on Windows; cuBLAS
        # is the delay-loaded DLL dependency (it loads cublasLt itself).
        foreach ($dll in 'cublas64_', 'cublasLt64_') {
            $hit = $modules | Where-Object { $_.ModuleName -like "$dll*" }
            if (-not $hit) { throw "$dll*.dll was not loaded" }
            $dir = Split-Path $hit[0].FileName -Parent
            if ($RuntimePath -notcontains $dir) { throw "$($hit[0].FileName) is outside -RuntimePath" }
        }
    }
} finally {
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force; $proc.WaitForExit() }
}
if (Select-String -Path $log, $err -SimpleMatch '[sched-dbg]' -Quiet) {
    Select-String -Path $log, $err -SimpleMatch '[sched-dbg]'
    throw "The $flavor backend rejected graph nodes (#184)"
}
"inference: exact transcript on $($health.backend)"
