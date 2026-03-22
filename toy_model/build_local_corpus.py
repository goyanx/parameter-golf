from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a larger local toy corpus from available project text sources.")
    p.add_argument("--out", default="data/local_slice_corpus.txt", help="Output path relative to toy_model/")
    p.add_argument("--target-bytes", type=int, default=2_000_000, help="Approx output size in bytes")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def _iter_docs_selected(data_root: Path) -> Iterable[str]:
    docs_path = data_root / "docs_selected.jsonl"
    if not docs_path.exists():
        return
    with docs_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text") or row.get("content") or row.get("body")
            if isinstance(text, str) and text.strip():
                yield text.strip()


def _iter_repo_docs(repo_root: Path) -> Iterable[str]:
    patterns = ["README.md", "memory.md", "checklist.md", "toy_model/*.md", "records/**/README.md"]
    seen: set[Path] = set()
    for pattern in patterns:
        for path in repo_root.glob(pattern):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for para in text.split("\n\n"):
                para = para.strip()
                if len(para) >= 40:
                    yield para


def main() -> None:
    args = parse_args()
    toy_root = Path(__file__).resolve().parent
    repo_root = toy_root.parent
    data_root = repo_root / "data"

    chunks: List[str] = []
    chunks.extend(_iter_docs_selected(data_root))
    chunks.extend(_iter_repo_docs(repo_root))

    if not chunks:
        raise RuntimeError("No local text sources found for corpus building")

    rng = random.Random(args.seed)
    rng.shuffle(chunks)

    out_path = toy_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    emitted: List[str] = []
    total = 0
    cursor = 0
    while total < args.target_bytes:
        chunk = chunks[cursor % len(chunks)]
        cursor += 1
        emitted.append(chunk)
        total += len(chunk.encode("utf-8")) + 2

    payload = "\n\n".join(emitted) + "\n"
    out_path.write_text(payload, encoding="utf-8")

    print(
        json.dumps(
            {
                "out_path": str(out_path),
                "source_chunks": len(chunks),
                "target_bytes": args.target_bytes,
                "written_bytes": len(payload.encode("utf-8")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
