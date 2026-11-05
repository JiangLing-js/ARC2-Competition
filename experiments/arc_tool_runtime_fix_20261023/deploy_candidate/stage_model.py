"""Copy the attached model once onto the Kaggle VM disk before TP workers start."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import sys
import time


def stage_model(source, destination, receipt):
    source, destination, receipt = map(Path, (source, destination, receipt))
    index = json.loads((source / "model.safetensors.index.json").read_text())
    names = sorted(set(index["weight_map"].values())) + [
        "model.safetensors.index.json", "config.json", "chat_template.jinja",
        "tokenizer.json", "tokenizer_config.json", "generation_config.json",
        "EXPORT_COMPLETE.json",
    ]
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def copy(name):
        before = time.monotonic()
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        # Sequential buffered reads avoid mmap/page-fault access against FUSE.
        # These writes also populate the VM's ordinary shared OS page cache.
        with (source / name).open("rb") as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
        return {"file": name, "bytes": target.stat().st_size,
                "seconds": time.monotonic() - before}

    files = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(copy, name) for name in names]):
            item = future.result()
            files.append(item)
            print("MODEL_FILE_STAGED", json.dumps(item), flush=True)
    result = {"source": str(source), "destination": str(destination),
              "files": files, "bytes": sum(item["bytes"] for item in files),
              "seconds": time.monotonic() - started, "workers": 8,
              "buffer_bytes": 16 * 1024 * 1024,
              "model_bytes_on_local_pc": 0, "load_strategy": "lazy"}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    print("MODEL_STAGE_COMPLETE", json.dumps({k: v for k, v in result.items() if k != "files"}),
          flush=True)
    return result


if __name__ == "__main__":
    stage_model(*sys.argv[1:])
