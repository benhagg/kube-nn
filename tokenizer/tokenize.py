"""Tokenize text and write train/validation NumPy shards for train_gpt2.py."""

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import tiktoken


input_path = Path(os.getenv("INPUT_PATH", "/input.txt"))
output_dir = Path(os.getenv("OUTPUT_DIR", "/data"))
encoding_name = os.getenv("TOKENIZER_ENCODING", "gpt2")
shard_size = int(os.getenv("SHARD_SIZE", "100000"))
validation_fraction = float(os.getenv("VALIDATION_FRACTION", "0.1"))
shard_prefix = os.getenv("SHARD_PREFIX", "dataset")
partition_key = os.getenv("PARTITION_KEY", "")

if not input_path.is_file():
    raise FileNotFoundError(f"Input text file does not exist: {input_path}")
if shard_size < 1:
    raise ValueError("SHARD_SIZE must be positive")
if not 0 < validation_fraction < 1:
    raise ValueError("VALIDATION_FRACTION must be between 0 and 1")

output_dir.mkdir(parents=True, exist_ok=True)

source_bytes = input_path.read_bytes()
source_hash = hashlib.sha256(source_bytes).hexdigest()
manifest_path = output_dir / ".partition.json"
shard_name_pattern = re.compile(r"^.+_(?:train|val)_\d+\.npy$")


def existing_partition_matches() -> bool:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False

    expected_settings = {
        "source_sha256": source_hash,
        "encoding": encoding_name,
        "shard_size": shard_size,
        "validation_fraction": validation_fraction,
        "shard_prefix": shard_prefix,
        "partition_key": partition_key,
    }
    if any(manifest.get(key) != value for key, value in expected_settings.items()):
        return False

    expected_shards = set(manifest.get("shards", []))
    present_shards = {
        path.name for path in output_dir.iterdir()
        if path.is_file() and shard_name_pattern.match(path.name)
    }
    return (
        bool(expected_shards)
        and expected_shards == present_shards
        and all((output_dir / name).is_file() and (output_dir / name).stat().st_size > 0
                for name in expected_shards)
        and any("_train_" in name for name in expected_shards)
        and any("_val_" in name for name in expected_shards)
    )


if existing_partition_matches():
    print("Dataset partitions match the input and tokenizer settings; skipping tokenization.")
    raise SystemExit(0)

# Remove the marker before replacing shards so trainer Pods continue waiting
# until the new partition has been written completely.
manifest_path.unlink(missing_ok=True)
for existing_shard in output_dir.iterdir():
    if existing_shard.is_file() and shard_name_pattern.match(existing_shard.name):
        existing_shard.unlink()

text = source_bytes.decode("utf-8")
encoding = tiktoken.get_encoding(encoding_name)
tokens = np.asarray(encoding.encode_ordinary(text), dtype=np.uint16)

if len(tokens) < 2:
    raise ValueError("Input must produce at least two tokens")

validation_count = max(1, int(len(tokens) * validation_fraction))
train_tokens = tokens[:-validation_count]
validation_tokens = tokens[-validation_count:]


written_shards: list[str] = []


def write_shards(split: str, split_tokens: np.ndarray) -> None:
    for shard_index, start in enumerate(range(0, len(split_tokens), shard_size)):
        shard = split_tokens[start : start + shard_size]
        path = output_dir / f"{shard_prefix}_{split}_{shard_index:06d}.npy"
        np.save(path, shard)
        written_shards.append(path.name)
        print(f"wrote {path} ({len(shard):,} tokens)")


write_shards("train", train_tokens)
write_shards("val", validation_tokens)

manifest = {
    "source_sha256": source_hash,
    "encoding": encoding_name,
    "shard_size": shard_size,
    "validation_fraction": validation_fraction,
    "shard_prefix": shard_prefix,
    "partition_key": partition_key,
    "shards": written_shards,
}
temporary_manifest = manifest_path.with_suffix(".json.tmp")
temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
temporary_manifest.replace(manifest_path)
