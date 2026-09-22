#!/usr/bin/env python
"""Замер токенизации чеченского против русского и английского + проверка chat-шаблона.

Отвечает на вопрос из SPEC §6: насколько неэффективно Qwen3-токенизатор ест чеченский
текст. Если раздутие > 1.5-2x против русского — это влияет на --max-seq-length и бюджет
токенов.

Запуск:
    uv run python scripts/check_tokens.py
    uv run python scripts/check_tokens.py --n 5000
"""

from __future__ import annotations

import argparse
import pathlib

import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MODEL = "mlx-community/Qwen3-4B-4bit"

# Палочка и характерные чеченские диграфы.
PALOCHKA_UPPER = "\u04c0"  # Ӏ
PALOCHKA_LOWER = "\u04cf"  # ӏ
SPECIAL_LETTERS = {
    "палочка Ӏ (U+04C0)": PALOCHKA_UPPER,
    "палочка ӏ (U+04CF)": PALOCHKA_LOWER,
    "аь": "аь", "оь": "оь", "уь": "уь", "юь": "юь", "яь": "яь",
    "гӏ": "гӏ", "кӏ": "кӏ", "пӏ": "пӏ", "тӏ": "тӏ",
    "хӏ": "хӏ", "цӏ": "цӏ", "чӏ": "чӏ", "кх": "кх", "къ": "къ", "хь": "хь",
}


def measure(tok, texts: list[str], label: str) -> dict:
    """Считает метрики эффективности токенизации для набора текстов."""
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return {}

    n_tok = 0
    n_chr = 0
    n_wrd = 0
    for t in texts:
        n_tok += len(tok.encode(t, add_special_tokens=False))
        n_chr += len(t)
        n_wrd += len(t.split())

    return {
        "label": label,
        "n": len(texts),
        "chars": n_chr,
        "tokens": n_tok,
        "chars_per_token": n_chr / n_tok,
        "tokens_per_char": n_tok / n_chr,
        "tokens_per_word": n_tok / n_wrd,
    }


def check_template(tok) -> None:
    """Проверяет, что шаблон сам добавляет пустой блок <think> (SPEC §2.1)."""
    messages = [
        {"role": "user", "content": "Муха хьо?"},
        {"role": "assistant", "content": "Дика ву, баркалла."},
    ]
    rendered = tok.apply_chat_template(messages, tokenize=False)
    print("  Рендер train-примера (add_generation_prompt=False):")
    for line in rendered.split("\n"):
        print(f"    | {line}")
    print(f"\n  содержит '<think>'     : {'<think>' in rendered}")
    print(f"  содержит пустой блок  : {'<think>\\n\\n</think>' in rendered or '<think>' in rendered}")

    gen = tok.apply_chat_template(
        messages[:1], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    print("\n  Рендер промпта генерации (enable_thinking=False):")
    for line in gen.split("\n"):
        print(f"    | {line}")
    print(f"\n  генерация и обучение дают один и тот же префикс: "
          f"{gen.rstrip() in rendered or rendered.startswith(gen.rstrip())}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=3000, help="сколько строк брать из корпуса")
    ap.add_argument("--min-words", type=int, default=6,
                    help="брать только предложения длиннее N слов")
    args = ap.parse_args()

    print(f"Загрузка токенизатора {MODEL} ...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    print(f"  vocab_size={tok.vocab_size:,}  model_max_length={tok.model_max_length:,}\n")

    print("=" * 72)
    print("1) ПРОВЕРКА CHAT-ШАБЛОНА")
    print("=" * 72)
    check_template(tok)

    print()
    print("=" * 72)
    print("2) ЭФФЕКТИВНОСТЬ ТОКЕНИЗАЦИИ")
    print("=" * 72)

    results = []

    # Английский — эталон «родного» для токенизатора языка.
    en_path = RAW / "wmt24pp_ce.parquet"
    if en_path.exists():
        rows = pq.read_table(en_path).to_pydict()
        en_texts = [s for s in rows.get("source", []) if s][: args.n]
        r = measure(tok, en_texts, "en (WMT24++ news)")
        if r:
            results.append(r)

    # Чеченский и русский из параллельного корпуса.
    p = RAW / "nmd_171k_train.parquet"
    if not p.exists():
        print(f"\n  [!] нет {p} — сначала запусти scripts/download_data.py")
    else:
        rows = pq.read_table(p).to_pydict()
        ce_all = rows.get("ce", [])
        ru_all = rows.get("ru", [])

        pairs = [
            (c, r) for c, r in zip(ce_all, ru_all)
            if c and r and len(c.split()) >= args.min_words
        ][: args.n]

        if not pairs:
            print(f"\n  [!] не нашлось пар длиннее {args.min_words} слов")
        else:
            print(f"\n  выборка: {len(pairs):,} пар длиннее {args.min_words} слов "
                  f"(из {len(ce_all):,} строк корпуса)")
            for r in (
                measure(tok, [c for c, _ in pairs], "ce (предложения)"),
                measure(tok, [r for _, r in pairs], "ru (предложения)"),
            ):
                if r:
                    results.append(r)

    if not results:
        return 1

    base = next((r for r in results if r["label"].startswith("ru")), results[0])
    print()
    print(f"  {'набор':<22} {'строк':>7} {'симв/ток':>9} {'ток/симв':>9} "
          f"{'ток/слово':>10} {'vs ru':>8}")
    print("  " + "-" * 70)
    for r in results:
        ratio = r["tokens_per_char"] / base["tokens_per_char"]
        print(f"  {r['label']:<22} {r['n']:>7,} {r['chars_per_token']:>9.2f} "
              f"{r['tokens_per_char']:>9.3f} {r['tokens_per_word']:>10.2f} {ratio:>7.2f}x")

    ce = next((r for r in results if r["label"].startswith("ce")), None)
    if ce:
        ratio = ce["tokens_per_char"] / base["tokens_per_char"]
        print()
        if ratio >= 2.0:
            verdict = "СИЛЬНОЕ раздутие — обязательно учесть в --max-seq-length"
        elif ratio >= 1.5:
            verdict = "заметное раздутие — учесть в бюджете токенов"
        elif ratio >= 1.15:
            verdict = "умеренное раздутие — приемлемо"
        else:
            verdict = "раздутия практически нет"
        print(f"  ВЕРДИКТ: чеченский ест в {ratio:.2f}x больше токенов на символ, чем русский.")
        print(f"           {verdict}")

    print()
    print("=" * 72)
    print("3) ПАЛОЧКА И ДИГРАФЫ")
    print("=" * 72)
    print(f"  {'символ':<22} {'кодпоинты':<22} {'токенов':>8}")
    print("  " + "-" * 56)
    for name, s in SPECIAL_LETTERS.items():
        ids = tok.encode(s, add_special_tokens=False)
        cps = " ".join(f"U+{ord(c):04X}" for c in s)
        print(f"  {name:<22} {cps:<22} {len(ids):>8}")
    pal = tok.encode(PALOCHKA_UPPER, add_special_tokens=False)
    digraphs = [len(tok.encode(s, add_special_tokens=False))
                for s in SPECIAL_LETTERS.values() if len(s) == 2]
    single = sum(1 for c in digraphs if c == 1)

    print()
    print(f"  Палочка: {len(pal)} токен(а) на символ — "
          f"{'символ токенизатору знаком' if len(pal) == 1 else 'символ токенизатору НЕ знаком'}.")
    print(f"  Диграфы: {single}/{len(digraphs)} кодируются одним токеном, "
          f"в среднем {sum(digraphs) / len(digraphs):.2f} токена на 2 буквы.")
    print()
    if single == 0:
        print("  ВЫВОД: проблема НЕ в палочке — она кодируется штатно. Проблема в том, что")
        print("  ни один чеченский диграф не является единицей словаря BPE: каждая пара букв")
        print("  стоит 2 токена. Отсюда и раздутие, измеренное в секции 2.")
    else:
        print("  ВЫВОД: часть диграфов представлена в словаре BPE одним токеном.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
