#!/usr/bin/env python
"""Выборка чеченских предложений из религиозных источников — на суд носителя.

Задача выборки: понять РЕГИСТР чеченского в Библии и Коране. Аргумент «модель заговорит
церковнославянским» относится к русской стороне пары; чеченская сторона написана
переводчиками с именами, и вопрос — приемлемый ли это современный чеченский или архаика.
Судить об этом может только носитель, поэтому скрипт готовит материал, а не вывод.

Запуск:
    uv run python scripts/sample_for_review.py
    uv run python scripts/sample_for_review.py --per-source 15 --seed 1
"""

from __future__ import annotations

import argparse
import pathlib

import pandas as pd

from mott.sources import is_bible, is_quran, normalize_source, words

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "review" / "religious_sample.md"


def pick(df: pd.DataFrame, mask, n: int, rng) -> pd.DataFrame:
    pool = df[mask]
    if pool.empty:
        return pool
    return pool.sample(n=min(n, len(pool)), random_state=rng)


def render(rows: pd.DataFrame, start: int, names: dict) -> list[str]:
    lines: list[str] = []
    for i, (_, r) in enumerate(rows.iterrows(), start=start):
        names.setdefault(r["src"], len(names) + 1)
        lines.append(f"**{i}. [{names[r['src']]}]**")
        lines.append(f"- ce: {r['ce']}")
        lines.append(f"- ru: {r['ru']}")
        lines.append("")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-source", type=int, default=10, help="сколько брать из Корана и из Библии")
    ap.add_argument("--min-words", type=int, default=7)
    ap.add_argument("--max-words", type=int, default=40, help="отсечь нечитаемо длинные стихи")
    ap.add_argument("--seed", type=int, default=20260923)
    args = ap.parse_args()

    path = RAW / "nmd_171k_train.parquet"
    if not path.exists():
        print(f"нет {path} — сначала scripts/download_data.py")
        return 1

    df = pd.read_parquet(path, columns=["ce", "ru", "source"])
    df["src"] = df["source"].map(normalize_source)
    df["w"] = df["ce"].map(words)

    in_range = df["w"].between(args.min_words, args.max_words)
    quran = in_range & df["src"].map(is_quran)
    bible = in_range & df["src"].map(is_bible)

    print(f"доступно в диапазоне {args.min_words}-{args.max_words} слов:")
    print(f"  Коран:  {quran.sum():,}")
    print(f"  Библия: {bible.sum():,}")

    names: dict[str, int] = {}
    lines = [
        "# Выборка религиозных текстов — на оценку носителя",
        "",
        f"Случайная выборка (seed={args.seed}), предложения длиной "
        f"{args.min_words}–{args.max_words} слов. Источник указан номером в квадратных скобках.",
        "",
        "**Вопрос один: это приемлемый современный чеченский или архаика?**",
        "Отмечай каждое как `ок` / `архаика` / `непонятно` — либо просто скажи общее впечатление.",
        "",
        "> Сначала идёт чеченский, под ним русская сторона пары — она здесь только для контекста.",
        "",
        "---",
        "",
        "## Коран (переводы Магомедова и Ибрагимова)",
        "",
    ]
    q = pick(df, quran, args.per_source, args.seed)
    lines += render(q, 1, names)

    lines += ["---", "", "## Библия (Институт перевода Библии)", ""]
    b = pick(df, bible, args.per_source, args.seed)
    lines += render(b, len(q) + 1, names)

    lines += ["---", "", "## Расшифровка источников", ""]
    for src, num in sorted(names.items(), key=lambda kv: kv[1]):
        lines.append(f"- [{num}] {src}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nзаписано: {OUT}  ({len(q)} из Корана + {len(b)} из Библии)")

    # То же самое в stdout для быстрого просмотра
    print("\n" + "=" * 70)
    for line in lines[8:]:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
