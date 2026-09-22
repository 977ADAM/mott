#!/usr/bin/env python
"""Скачивание чеченских датасетов NM-development с Hugging Face.

Кладёт parquet-файлы в data/raw/ (каталог в .gitignore).

Запуск:
    uv run python scripts/download_data.py
    uv run python scripts/download_data.py --only nmd_171k_train
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

from datasets import load_dataset

RAW = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"

# (repo, config, split, имя_файла, описание)
SPECS: list[tuple[str, str, str, str, str]] = [
    (
        "NM-development/nmd-ce-ru-171k-v0",
        "default",
        "train",
        "nmd_171k_train",
        "171 224 пары ce-ru; ~75% — словарные статьи (медиана 2 слова)",
    ),
    (
        "NM-development/nmd-ce-ru-171k-v0",
        "benchmark",
        "test",
        "nmd_171k_benchmark",
        "360 пар ce-ru-en; словарь + стихи Библии",
    ),
    (
        "NM-development/wmt24pp-ce",
        "default",
        "train",
        "wmt24pp_ce",
        "998 новостных en->ce; есть строка-canary и флаг is_bad_source",
    ),
    (
        "NM-development/ce_ru_officials",
        "default",
        "train",
        "ce_ru_officials",
        "официальные тексты ce-ru",
    ),
    (
        "NM-development/ce_ru_toponyms",
        "default",
        "train",
        "ce_ru_toponyms",
        "топонимы ce-ru",
    ),
    (
        "NM-development/smol_ce",
        "default",
        "train",
        "smol_ce",
        "smoldoc.parquet — документный чеченский корпус",
    ),
]


def download(spec: tuple[str, str, str, str, str], force: bool = False) -> bool:
    repo, config, split, name, desc = spec
    out = RAW / f"{name}.parquet"

    if out.exists() and not force:
        print(f"  [skip] {name}.parquet уже есть ({out.stat().st_size / 1e6:.1f} МБ)")
        return True

    print(f"  [{name}] {repo} / {config} / {split}")
    print(f"           {desc}")
    t0 = time.time()
    try:
        ds = load_dataset(repo, config, split=split)
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] {repo} ({config}/{split}): {type(exc).__name__}: {exc}")
        return False

    ds.to_parquet(str(out))
    print(
        f"  [ok]   {len(ds):>7,} строк -> {out.name} "
        f"({out.stat().st_size / 1e6:.2f} МБ, {time.time() - t0:.1f} с)"
    )
    print(f"         поля: {ds.column_names}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", action="append", help="скачать только эти имена")
    ap.add_argument("--force", action="store_true", help="перекачать даже если файл есть")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)

    specs = SPECS
    if args.only:
        wanted = set(args.only)
        specs = [s for s in SPECS if s[3] in wanted]
        missing = wanted - {s[3] for s in specs}
        if missing:
            print(f"неизвестные имена: {sorted(missing)}", file=sys.stderr)
            print(f"доступные: {[s[3] for s in SPECS]}", file=sys.stderr)
            return 2

    print(f"Скачивание в {RAW}\n")
    ok = sum(download(s, force=args.force) for s in specs)

    print(f"\nготово: {ok}/{len(specs)}")
    total = sum(f.stat().st_size for f in RAW.glob("*.parquet"))
    print(f"всего на диске: {total / 1e6:.1f} МБ")
    return 0 if ok == len(specs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
