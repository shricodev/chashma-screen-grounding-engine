import glob
import os
import subprocess
import sys
import zipfile
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
STAMP = "__STAMP__"  # kaggle_train.sh fills this in
PINS = ["transformers==5.22.1", "peft==0.21.2", "accelerate==1.15.0"]


def sh(*cmd: str) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def find(pattern: str) -> Path | None:
    hits = sorted(glob.glob(str(INPUT / "**" / pattern), recursive=True))
    return Path(hits[0]) if hits else None


def locate_data() -> Path:
    if meta := find("dataset/meta.json"):
        return meta.parent.parent
    out = WORK / "data"
    for archive in glob.glob(str(INPUT / "**" / "*.zip"), recursive=True):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out / Path(archive).stem)
    if (out / "dataset" / "meta.json").exists():
        return out
    raise FileNotFoundError("dataset/meta.json not found in /kaggle/input")


def main() -> None:
    if not STAMP.startswith("__") and find(f".stamp-{STAMP}.txt") is None:
        raise RuntimeError(f"/kaggle/input is not the data version pushed with stamp {STAMP}")
    wheel = find("chashma-*.whl")
    if wheel is None:
        raise FileNotFoundError("chashma wheel not found in /kaggle/input")
    sh(sys.executable, "-m", "pip", "install", "-q", str(wheel), *PINS)
    # kaggle has an old torchao preinstalled and peft breaks with it
    sh(sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao")

    import yaml

    data = locate_data()
    cfg = yaml.safe_load((data / "configs" / "train.yaml").read_text())
    cfg.update(
        dataset=str(data / "dataset"),
        image_root=str(data / "captures"),
        output_dir=str(WORK / Path(cfg["output_dir"]).name),
        num_workers=4,
        keep_checkpoints=False,
    )
    config = WORK / "train.yaml"
    config.write_text(yaml.safe_dump(cfg))
    # 2 t4s but dataparallel breaks idefics3, so just one
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    sh("nvidia-smi")
    sh("chashma", "train", "--config", str(config))


if __name__ == "__main__":
    main()
