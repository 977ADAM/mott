#!/usr/bin/env python
"""Этап 8: оценка адаптера — сравнение с базовой моделью глазами носителя.

Перплексия говорит, что модель стала лучше предсказывать чеченский текст, но не
говорит, ГРАМОТНО ли она его порождает. Это решает только носитель, поэтому скрипт
готовит материал для чтения, а не вывод.

Генерирует ответы базовой модели и адаптера на одни и те же промпты и пишет
data/review/eval.md.

Запуск:
    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py --n 15 --adapter ./adapters/mott-v0.1-best
    uv run python scripts/evaluate.py --prompt "Муха хьо?"
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "review" / "eval.md"
MODEL = "mlx-community/Qwen3-4B-4bit"


def build_prompt(tokenizer, user_text: str, system: str | None = None) -> str:
    """Тот же шаблон, что при обучении: enable_thinking=False, без <think> в данных."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_text})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", type=pathlib.Path, default=ROOT / "adapters" / "mott-v0.1-best")
    ap.add_argument("--n", type=int, default=12, help="сколько промптов взять из test.jsonl")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--prompt", action="append", help="свой промпт (можно несколько)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    prompts: list[tuple[str, str]] = []  # (метка, текст промпта)

    test_path = DATA / "test.jsonl"
    if test_path.exists():
        rows = [json.loads(l) for l in test_path.open(encoding="utf-8") if l.strip()]
        random.Random(args.seed).shuffle(rows)
        for r in rows[: args.n]:
            prompts.append(("test", r["messages"][0]["content"]))

    for p in args.prompt or []:
        prompts.append(("свой", p))

    # Свободные промпты носителя, если он их положил.
    free = DATA / "review" / "prompts.txt"
    if free.exists():
        for line in free.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(("свободный", line))

    if not prompts:
        print("нет промптов")
        return 1

    print(f"загружаю базовую модель {MODEL} ...")
    base, tok = load(MODEL)

    print(f"загружаю адаптер {args.adapter} ...")
    tuned, _ = load(MODEL, adapter_path=str(args.adapter))

    sampler = make_sampler(temp=0.0)  # детерминированно: сравниваем модели, а не сэмплинг
    lines = [
        "# Оценка адаптера — на чтение носителю",
        "",
        f"Адаптер: `{args.adapter.relative_to(ROOT)}` · seed={args.seed} · temp=0",
        "",
        "Слева базовая Qwen3-4B, справа она же с адаптером. Вопрос один:",
        "**стало ли по-чеченски грамотнее и естественнее?**",
        "",
        "---",
        "",
    ]

    for i, (kind, text) in enumerate(prompts, 1):
        print(f"  [{i}/{len(prompts)}] {kind}: {text[:60]}...")
        p = build_prompt(tok, text)
        b = generate(base, tok, prompt=p, max_tokens=args.max_tokens, sampler=sampler).strip()
        t = generate(tuned, tok, prompt=p, max_tokens=args.max_tokens, sampler=sampler).strip()

        lines += [
            f"### {i}. ({kind})",
            "",
            f"**Промпт:** {text}",
            "",
            f"- **база:** {b}",
            f"- **адаптер:** {t}",
            "",
            "| грамматика 1-5 | орфография 1-5 | естественность 1-5 | по теме 1-5 |",
            "|---|---|---|---|",
            "|  |  |  |  |",
            "",
        ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nзаписано: {OUT}")
    print("Заполни таблицы — это и есть главная метрика качества (SPEC §5.2).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
