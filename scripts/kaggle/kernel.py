import glob
import os
import subprocess
import sys
import zipfile
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
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
    wheel = find("chashma-*.whl")
    if wheel is None:
        raise FileNotFoundError("chashma wheel not found in /kaggle/input")
    sh(sys.executable, "-m", "pip", "install", "-q", str(wheel), *PINS)

    import yaml

    data = locate_data()
    cfg = yaml.safe_load((data / "configs" / "train_256m.yaml").read_text())
    cfg.update(
        dataset=str(data / "dataset"),
        image_root=str(data / "captures"),
        output_dir=str(WORK / "chashma-256m"),
        num_workers=4,
        batch_size=4,
        grad_accum=2,
        keep_checkpoints=False,
    )
    config = WORK / "train.yaml"
    config.write_text(yaml.safe_dump(cfg))
    # 2 t4s but dataparallel breaks idefics3, so just one
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    sh("nvidia-smi")
    sh("chashma", "train", "--config", str(config))


if __name__ == "__main__":
    main()
