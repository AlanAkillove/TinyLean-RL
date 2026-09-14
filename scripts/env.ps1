# PowerShell equivalent of scripts/env.sh for local development on Windows.
$env:TINYLEAN_ROOT = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:HF_HOME = if ($env:HF_HOME) { $env:HF_HOME } else { Join-Path $env:TINYLEAN_ROOT ".cache\huggingface" }
$env:HUGGINGFACE_HUB_CACHE = if ($env:HUGGINGFACE_HUB_CACHE) { $env:HUGGINGFACE_HUB_CACHE } else { Join-Path $env:HF_HOME "hub" }
$env:HF_HUB_DOWNLOAD_TIMEOUT = if ($env:HF_HUB_DOWNLOAD_TIMEOUT) { $env:HF_HUB_DOWNLOAD_TIMEOUT } else { "300" }
$env:HF_HUB_ETAG_TIMEOUT = if ($env:HF_HUB_ETAG_TIMEOUT) { $env:HF_HUB_ETAG_TIMEOUT } else { "60" }
$env:XDG_CACHE_HOME = if ($env:XDG_CACHE_HOME) { $env:XDG_CACHE_HOME } else { Join-Path $env:TINYLEAN_ROOT ".cache" }
$env:UV_CACHE_DIR = if ($env:UV_CACHE_DIR) { $env:UV_CACHE_DIR } else { Join-Path $env:TINYLEAN_ROOT ".cache\uv" }
$env:TORCH_HOME = if ($env:TORCH_HOME) { $env:TORCH_HOME } else { Join-Path $env:TINYLEAN_ROOT ".cache\torch" }
$env:TINYLEAN_MODEL_ROOT = if ($env:TINYLEAN_MODEL_ROOT) { $env:TINYLEAN_MODEL_ROOT } else { Join-Path $env:TINYLEAN_ROOT "models\weights" }
$env:TINYLEAN_DATA_ROOT = if ($env:TINYLEAN_DATA_ROOT) { $env:TINYLEAN_DATA_ROOT } else { Join-Path $env:TINYLEAN_ROOT "data\raw" }
$env:LEAN_SERVER_API_URL = if ($env:LEAN_SERVER_API_URL) { $env:LEAN_SERVER_API_URL } else { "http://127.0.0.1:8000" }
$env:PYTHONPATH = "$(Join-Path $env:TINYLEAN_ROOT 'src');$env:PYTHONPATH"

@(
  $env:HF_HOME, $env:HUGGINGFACE_HUB_CACHE, $env:UV_CACHE_DIR,
  $env:TORCH_HOME, $env:TINYLEAN_MODEL_ROOT, $env:TINYLEAN_DATA_ROOT,
  (Join-Path $env:TINYLEAN_ROOT "runs")
) | ForEach-Object { New-Item -ItemType Directory -Force -Path $_ | Out-Null }

Write-Host "TinyLean-RL environment loaded: $env:TINYLEAN_ROOT"
