#!/usr/bin/env python
"""Этап 3: фильтрация nmd-ce-ru-171k-v0 до пригодного пула предложений.

Отсеивает:
  1. Короткие словарные статьи (по умолчанию <= 5 слов, т.е. берём >5).
  2. Религиозные тексты — по решению носителя: Библия вырезается, Коран остаётся.
  3. num2words — не язык, а проговаривание чисел прописью.

Затем нормализует пробелы и промежутки вокруг знаков. Кавычки НЕ трогает:
проверка показала, что `"` в Коране — настоящий знак препинания, а не повреждение
(см. mott/text.py).

Оставшийся пул пишется в data/pool_nmd.parquet и идёт в конструктор chat-пар.

Запуск:
    uv run python scripts/filter_dictionary.py                    # Коран оставить
    uv run python scripts/filter_dictionary.py --religion drop    # вырезать всё религиозное
    uv run python scripts/filter_dictionary.py --religion keep    # оставить всё
"""

from __future__ import annotations

import argparse
import pathlib

import pandas as pd

from mott.sources import (
    chechen_translator,
    edition_rank,
    is_bible,
    is_junk,
    is_religious,
    normalize_source,
)
from mott.text import artifact_rates, clean

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "pool_nmd.parquet"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-words", type=int, default=5,
                    help="минимум слов в чеченской части (по умолчанию 5, т.е. берём >5)")
    ap.add_argument("--religion", choices=["quran", "drop", "keep"], default="quran",
                    help="quran (по умолчанию) — оставить только Коран; "
                         "drop — вырезать всё религиозное; keep — оставить всё")
    ap.add_argument("--translation", choices=["magomedov", "ibragimov", "both"],
                    default="magomedov",
                    help="какой чеченский перевод Корана брать. По умолчанию ОДИН "
                         "(Магомедов): два перевода дают разные цели на одном русском входе")
    ap.add_argument("--keep-junk", action="store_true", help="НЕ фильтровать num2words")
    ap.add_argument("--no-clean", action="store_true", help="не нормализовать пробелы")
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    args = ap.parse_args()

    src_path = RAW / "nmd_171k_train.parquet"
    if not src_path.exists():
        print(f"нет {src_path} — сначала scripts/download_data.py")
        return 1

    df = pd.read_parquet(src_path, columns=["ce", "ru", "source"])
    total = len(df)
    print(f"вход: {total:,} строк\n")

    df["src"] = df["source"].map(normalize_source)
    df["ce_w"] = df["ce"].fillna("").str.split().str.len()
    df["relig"] = df["src"].map(is_religious)
    df["bible"] = df["src"].map(is_bible)

    steps: list[tuple[str, int]] = [("вход", total)]

    # 1) длина
    df = df[df["ce_w"] > args.min_words].copy()
    steps.append((f"ce > {args.min_words} слов", len(df)))

    # 2) религия
    if args.religion == "drop":
        m = ~df["relig"]
        df = df[m].copy()
        steps.append((f"минус всё религиозное (-{(~m).sum():,})", len(df)))
    elif args.religion == "quran":
        m = ~df["bible"]
        df = df[m].copy()
        steps.append((f"минус Библия, Коран оставлен (-{(~m).sum():,})", len(df)))

    # 3) num2words
    if not args.keep_junk:
        m = ~df["src"].map(is_junk)
        df = df[m].copy()
        steps.append((f"минус num2words (-{(~m).sum():,})", len(df)))

    # 4) один чеченский перевод Корана (не-коранические строки остаются)
    if args.translation != "both":
        tr = df["src"].map(chechen_translator)
        m = tr.isna() | tr.eq(args.translation)
        df = df[m].copy()
        steps.append((f"один перевод Корана: {args.translation} (-{(~m).sum():,})", len(df)))

    print("=== ВОРОНКА ===")
    for i, (name, n) in enumerate(steps):
        delta = "" if i == 0 else f"  ({n - steps[i - 1][1]:+,})"
        print(f"  {name:<36} {n:>7,}{delta}")

    if df.empty:
        print("\nпул пуст — ослабь фильтры")
        return 1

    # 4) нормализация текста
    if not args.no_clean:
        before = artifact_rates(df["ce"].fillna(""))
        df["ce"] = df["ce"].map(clean)
        df["ru"] = df["ru"].map(clean)
        after = artifact_rates(df["ce"])
        steps.append(("нормализация пробелов", len(df)))
        print("\n  артефакты в ce, до -> после:")
        for key in before:
            print(f"    {key:<32} {before[key]:>5.1f}% -> {after[key]:>5.1f}%")

    # 5) пустые и дубли.
    # Дедуп по ce: у каждого аята 2 чеченских перевода x 3 русские редакции.
    # Сортируем по предпочтению редакции ДО дедупа, иначе останется та, что лежит
    # первой в файле (Adel — худшая).
    before = len(df)
    df = df[df["ce"].str.strip().ne("") & df["ru"].str.strip().ne("")]
    df["_rank"] = df["src"].map(edition_rank)
    df = (df.sort_values("_rank", kind="stable")
            .drop_duplicates(subset=["ce"], keep="first")
            .drop(columns=["_rank"]))
    print(f"  минус пустые и дубли по ce            {len(df):>7,}  ({len(df) - before:+,})")

    print(f"\n=== ПУЛ: {len(df):,} строк, медиана {df['ce_w'].median():.0f} слов ===")
    print("\n  по источникам:")
    for src, n in df["src"].value_counts().head(12).items():
        print(f"    {n:>6,}  {src[:62]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df[["ce", "ru", "src"]].to_parquet(args.out, index=False)
    print(f"\nзаписано: {args.out} ({args.out.stat().st_size / 1e6:.2f} МБ)")

    print("\n  примеры после чистки:")
    for _, r in df.sample(min(4, len(df)), random_state=3).iterrows():
        print(f"    ce: {r['ce'][:96]}")
        print(f"    ru: {r['ru'][:96]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
