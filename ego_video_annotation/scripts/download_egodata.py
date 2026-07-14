#!/usr/bin/env python3
"""Download the assessment videos from ModelScope to a reproducible location."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 ModelScope egodata 数据集")
    parser.add_argument("--dataset-id", default="ly985211/egodata")
    parser.add_argument("--output", default="data/egodata")
    args = parser.parse_args()
    try:
        from modelscope import dataset_snapshot_download
    except ImportError as exc:
        raise SystemExit("缺少 modelscope；请执行 pip install -r requirements.txt") from exc
    target = Path(args.output).resolve()
    print(f"[info] 下载 {args.dataset_id} → {target}")
    dataset_snapshot_download(dataset_id=args.dataset_id, local_dir=str(target))
    print("[done] 下载完成")


if __name__ == "__main__":
    main()
