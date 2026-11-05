# ARC Mamba prompt-boundary pin patch (vLLM 0.30.0)

- `install_mamba_pin_patch.py VENV RECEIPT.json` appends a guarded monkeypatch to
  `vllm/v1/core/single_type_kv_cache_manager.py` inside the unpacked venv. It fails closed
  if the vLLM version or any upstream anchor text differs, or if it is already installed.
- Runtime switches: `ARC_MAMBA_PIN=0` disables it; `ARC_MAMBA_PIN_MAX_PER_REQ` (default 1)
  sets how many hashed snapshots each request may pin per Mamba group (newest kept).
- `test_mamba_pin_patch.py`: anchors + compile check against the saved upstream copy
  (`research/arc_pi_kv_20261019/upstream`) and ref-count logic on stubs. No GPU or vLLM import.

## deployment.py integration (not applied to any live package)

After `command("unpack_environment", ...)` and before `spawn("server", ...)`:

```python
command("mamba_pin_patch", [str(python), str(HERE / "install_mamba_pin_patch.py"),
        str(venv), str(output / "MAMBA_PIN_PATCH.json")], startup_deadline)
```

After the server is ready, require `ARC_MAMBA_PIN init` in `server.log` (it is logged by
each EngineCore MambaManager) and record the count of lines as the number of Mamba groups.

## Capacity note

Each pinned block holds one KV page. With G Mamba groups, the pins cost up to
`running_requests x G x 1600` token-equivalents (for example 24 x 3 x 1600 = 115K, about 9.7%
of 1.18M). The client KV watermark sees the pins as usage. The exact G comes from the
`ARC_MAMBA_PIN init` count. That makes raising KV to 12 GiB more worthwhile.
