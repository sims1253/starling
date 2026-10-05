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
# The extraction directory is deleted when the check passes and kept, with its
# path printed, when it fails, so a failed run's logs stay inspectable.
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

# One extraction directory per run. On success it is removed below; on any
# failure the catch block keeps it (extracted archive, server logs) for
# inspection and prints its path.
$work = Join-Path ([IO.Path]::GetTempPath()) ("starling-check-" + [guid]::NewGuid())
New-Item -ItemType Directory $work | Out-Null
try {
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

    # Inference takes all three of -Gguf/-Audio/-Expected or none of them: a
    # caller who passes only some asked for the hardware verification, and a
    # silent startup-only pass would be a false success.
    if (($Gguf -or $Audio -or $Expected) -and -not ($Gguf -and $Audio -and $Expected)) {
        throw 'Inference requires all of -Gguf, -Audio, and -Expected'
    }
    if (-not $Gguf) {
        "startup check passed (no inference requested)"
    } else {
        # A server already listening on the port could answer for the one under test.
        if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
            throw "Port $Port is already in use; pass a free -Port"
        }
        $log = Join-Path $work 'server.log'
        $err = Join-Path $work 'server.err.log'
        # PS 5.1's Start-Process joins -ArgumentList elements with spaces
        # without quoting them, so a path with spaces must carry its own
        # quotes — and a trailing backslash before that closing quote would
        # be read as escaping it under Windows argv parsing ("C:\dir\"
        # arrives as C:\dir"). Resolve-Path output for a file never ends in
        # a separator, so the quotes are safe as-is; the input TrimEnd just
        # accepts a caller-supplied trailing separator instead of letting
        # Resolve-Path reject it, and the output TrimEnd is belt-and-braces.
        $ggufPath = (Resolve-Path $Gguf.TrimEnd('\')).Path.TrimEnd('\')
        $base = "http://127.0.0.1:$Port"
        # The server child inherits STARLING_SCHED_DEBUG from this process
        # at Start-Process. Capture the caller's value and set ours right
        # before the try, so every failure from here on (Start-Process
        # included) restores it in the finally instead of leaking scheduler
        # debug into a session that ran this script in-process.
        $schedDebugBefore = $env:STARLING_SCHED_DEBUG
        $env:STARLING_SCHED_DEBUG = '1'
        # Null before the try: a dot-sourced rerun must never stop a $proc
        # left over from an earlier run.
        $proc = $null
        try {
            $proc = Start-Process -FilePath $exe -PassThru -NoNewWindow `
                -RedirectStandardOutput $log -RedirectStandardError $err `
                -ArgumentList @('--model', 'parakeet', '--gguf', "`"$ggufPath`"",
                                '--host', '127.0.0.1', '--port', "$Port")
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
            $owners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
                Select-Object -ExpandProperty OwningProcess -Unique)
            if ($owners.Count -ne 1 -or $owners[0] -ne $proc.Id) {
                throw "Port $Port is not served by the process under test (pid $($proc.Id); listeners: $($owners -join ','))"
            }
            "health: " + ($health | ConvertTo-Json -Compress)
            if ($health.backend -notmatch $deviceIs) { throw "Expected a $flavor device, got $($health.backend)" }

            # curl.exe ships in System32 on Windows 10 1803+ and Windows 11.
            $response = & curl.exe -sS --connect-timeout 10 --max-time 600 -w "`n%{http_code}" -F model=parakeet `
                -F "file=@$((Resolve-Path $Audio).Path);type=audio/wav" "$base/v1/audio/transcriptions"
            if ($LASTEXITCODE -ne 0) { throw "curl.exe failed with exit code $LASTEXITCODE" }
            $status = $response[-1]
            # 0..-1 would duplicate the status line if there were no body
            # lines; curl's -w newline makes that unreachable today (an empty
            # body still yields two lines), but the guard keeps the slice
            # honest.
            $body = if ($response.Count -gt 1) { $response[0..($response.Count - 2)] -join "`n" } else { '' }
            "HTTP $status $body"
            if ($status -ne '200') { throw "Transcription failed with HTTP $status" }
            $text = ($body | ConvertFrom-Json).text
            if ($text -cne $Expected) { throw "Transcript mismatch: '$text' != '$Expected'" }
            if ($proc.HasExited) { throw 'Server died during transcription' }

            # Which runtime and driver DLLs the server actually loaded, and from
            # where. The process can exit between the check above and this query, or
            # module access can be denied; name the actual failure instead of letting
            # an unrelated runtime error escape.
            try {
                $modules = (Get-Process -Id $proc.Id -ErrorAction Stop).Modules |
                    Where-Object { $_.ModuleName -match '^(vcomp|cudart|cublas|nvcuda|vulkan-1|nvoglv|amdvlk|igvk)' } |
                    Sort-Object ModuleName
            } catch {
                throw "Could not enumerate the server's loaded modules (pid $($proc.Id)): $($_.Exception.Message)"
            }
            $modules | ForEach-Object { "loaded: $($_.FileName) $($_.FileVersionInfo.FileVersion)" }
            if ($flavor -eq 'vulkan') {
                # RUNTIME.md's windows-vulkan prerequisites are "Vulkan loader
                # and vendor Vulkan driver"; the archive bundles no loader, so
                # vulkan-1.dll must come from -RuntimePath (if given) or the
                # Windows system directories. One resolved from an installed
                # SDK elsewhere (C:\VulkanSDK\...) would mean the scrubbed PATH
                # did not hold. The vendor ICD (nvoglv/amdvlk/igvk) is
                # deliberately unconstrained: the loader finds it through the
                # driver's own registry configuration, outside every PATH
                # directory.
                $hit = $modules | Where-Object { $_.ModuleName -like 'vulkan-1*' }
                if (-not $hit) { throw 'vulkan-1.dll was not loaded' }
                $dir = Split-Path $hit[0].FileName -Parent
                if (($RuntimePath -notcontains $dir) -and (($system -split ';') -notcontains $dir)) {
                    throw "$($hit[0].FileName) resolved outside -RuntimePath and the system directories"
                }
            }
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
            # Restore the environment FIRST (it cannot throw): a Stop-Process
            # race — the server exiting between the HasExited check and the
            # stop — must not skip the restore or mask the original error.
            if ($null -eq $schedDebugBefore) {
                Remove-Item Env:STARLING_SCHED_DEBUG -ErrorAction SilentlyContinue
            } else {
                $env:STARLING_SCHED_DEBUG = $schedDebugBefore
            }
            if ($proc -and -not $proc.HasExited) {
                Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
                $proc.WaitForExit()
            }
        }
        if (Select-String -Path $log, $err -SimpleMatch '[sched-dbg]' -Quiet) {
            Select-String -Path $log, $err -SimpleMatch '[sched-dbg]'
            throw "The $flavor backend rejected graph nodes (#184)"
        }
        "inference: exact transcript on $($health.backend)"
    }
} catch {
    "check failed; work directory kept for inspection: $work"
    throw
}
# A cleanup failure (e.g. antivirus briefly holding the extracted exe) must
# not flip an already-passed check to a failed exit; warn and leave the dir.
try {
    Remove-Item -LiteralPath $work -Recurse -Force
} catch {
    Write-Warning "could not remove ${work}: $($_.Exception.Message)"
}
