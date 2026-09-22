#!/usr/bin/env python
"""Этап 4: конструктор chat-датасета для mlx_lm.lora.

Собирает data/train.jsonl, data/valid.jsonl, data/test.jsonl в формате `chat`:

    {"messages": [{"role": "user", "content": "..."},
                  {"role": "assistant", "content": "..."}]}

Источники:
  * data/pool_nmd.parquet      — Корана (Магомедов/Kuliev) + проза + словари;
  * data/raw/smol_ce.parquet   — 825 выровненных ce/ru/en;
  * data/raw/ce_ru_officials.parquet, ce_ru_toponyms.parquet — термины;
  * data/raw/wmt24pp_ce.parquet — en->ce (в ru-задачах не участвует);
  * data/manual/instructions.jsonl — рукописные пары носителя (если есть).

Запуск:
    uv run python scripts/build_dataset.py --dry-run     # показать, не писать
    uv run python scripts/build_dataset.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
CONFIG = ROOT / "configs" / "tasks.yaml"

# Слова, которые ломают chat-шаблон Qwen3: он режет assistant-сообщение по </think>.
FORBIDDEN_IN_CONTENT = ("</think>", "<think>", "<|im_start|>", "<|im_end|>")


def load_tasks(lang: str) -> dict[str, str]:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    missing: list[str] = []
    for key, variants in cfg.items():
        val = variants.get(lang)
        if val == "FILL_ME":
            missing.append(key)
            continue
        out[key] = val
    if missing:
        print(
            f"шаблоны для языка '{lang}' не заполнены: {missing}\n"
            f"Их должен дать носитель в {CONFIG.relative_to(ROOT)}. "
            f"Либо запусти с --lang ru.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return out


def valid_pair(ce: str, ru: str) -> bool:
    """Проверки из чек-листа SPEC §3.6, применимые автоматически."""
    if not ce or not ru:
        return False
    if not ce.strip() or not ru.strip():
        return False
    for bad in FORBIDDEN_IN_CONTENT:
        if bad in ce or bad in ru:
            return False
    # Однобуквенные «переводы» и совпадение языков — мусор.
    if len(ce.strip()) < 3 or len(ru.strip()) < 3:
        return False
    return True


def make_examples(df: pd.DataFrame, tasks: dict[str, str], rng: random.Random) -> list[dict]:
    """Превращает пары ce/ru в chat-примеры. Одна пара -> один пример, направление случайно."""
    system = tasks.get("system")
    out: list[dict] = []
    for _, r in df.iterrows():
        ce, ru = str(r["ce"]).strip(), str(r["ru"]).strip()
        if not valid_pair(ce, ru):
            continue
        if rng.random() < 0.5:
            template, text, answer = tasks["translate_ru_ce"], ru, ce
        else:
            template, text, answer = tasks["translate_ce_ru"], ce, ru

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": template.format(text=text)})
        messages.append({"role": "assistant", "content": answer})
        out.append({"messages": messages, "src": r.get("src", r.get("origin", "?"))})

    return out


def load_pool() -> pd.DataFrame:
    path = DATA / "pool_nmd.parquet"
    if not path.exists():
        print(f"нет {path} — сначала scripts/filter_dictionary.py", file=sys.stderr)
        raise SystemExit(1)
    df = pd.read_parquet(path)
    df["origin"] = "nmd_171k"
    return df[["ce", "ru", "src", "origin"]]


def load_extra() -> pd.DataFrame:
    """Остальные датасеты. Имена полей у них РАЗНЫЕ — см. AGENTS.md, грабли №5."""
    frames = []

    p = RAW / "smol_ce.parquet"
    if p.exists():
        d = pd.read_parquet(p)
        frames.append(pd.DataFrame({
            "ce": d["che_Cyrl"], "ru": d["rus_Cyrl"],
            "src": "smol_ce", "origin": "smol_ce",
        }))

    for name, src in [("ce_ru_officials", "officials"), ("ce_ru_toponyms", "toponyms")]:
        p = RAW / f"{name}.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            frames.append(pd.DataFrame({
                "ce": d["ce_text"], "ru": d["ru_text"],
                "src": name, "origin": "terms",
            }))

    if not frames:
        return pd.DataFrame(columns=["ce", "ru", "src", "origin"])
    return pd.concat(frames, ignore_index=True)


def filter_by_length(examples: list[dict], tokenizer, max_tokens: int) -> tuple[list[dict], int]:
    """Выкидывает примеры длиннее max_tokens.

    Зачем: mlx-lm паддит ВЕСЬ батч до самого длинного примера в нём
    (`trainer.py`: `max_length_in_batch = 1 + 32 * ((max(lengths)+31)//32)`).
    Один пример на 680 токенов в батче из 8 раздувает батч до 5 440 позиций
    и валит обучение по памяти. Замер: ~10 МБ на позицию токена, бюджет ~14 ГБ
    => предел ~1 400 позиций.

    Отсекать лучше, чем давать mlx-lm молча обрезать: обрезанный ответ учит
    модель обрывать фразу на середине.
    """
    kept, dropped = [], 0
    for e in examples:
        text = tokenizer.apply_chat_template(e["messages"], tokenize=False)
        n = len(tokenizer.encode(text, add_special_tokens=False))
        if n > max_tokens:
            dropped += 1
            continue
        e["_ntok"] = n
        kept.append(e)
    return kept, dropped


def validate_jsonl(path: pathlib.Path) -> tuple[int, list[str]]:
    """Возвращает (число строк, список проблем)."""
    problems: list[str] = []
    n = 0
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                problems.append(f"{path.name}:{lineno} невалидный JSON: {exc}")
                continue
            msgs = obj.get("messages")
            if not isinstance(msgs, list) or len(msgs) < 2:
                problems.append(f"{path.name}:{lineno} нет пары messages")
                continue
            if msgs[0]["role"] == "system" and any(m["role"] == "system" for m in msgs[1:]):
                problems.append(f"{path.name}:{lineno} system не первый/не один")
            for m in msgs:
                for bad in FORBIDDEN_IN_CONTENT:
                    if bad in str(m.get("content", "")):
                        problems.append(f"{path.name}:{lineno} запрещённая подстрока {bad}")
                if "\n" in str(m.get("content", "")):
                    problems.append(f"{path.name}:{lineno} перенос строки внутри content")
    return n, problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", choices=["ru", "ce"], default="ru",
                    help="язык формулировки инструкций (чеченский заполняет носитель)")
    ap.add_argument("--n-train", type=int, default=1200, help="примеров в train (без рукописных)")
    ap.add_argument("--n-valid", type=int, default=120)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--quran-share", type=float, default=0.6,
                    help="доля Корана в выборке; остальное — проза/термины")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="выкинуть примеры длиннее N токенов (0 = не фильтровать). "
                         "См. filter_by_length(): длинные примеры раздувают паддинг батча")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tasks = load_tasks(args.lang)
    rng = random.Random(args.seed)

    pool = load_pool()
    extra = load_extra()
    print(f"пул: {len(pool):,} строк | прочие источники: {len(extra):,} строк")

    # Балансировка: иначе Коран задавит всё остальное.
    quran = pool[pool["src"].str.contains("Quran", case=False, na=False)]
    other = pool[~pool["src"].str.contains("Quran", case=False, na=False)]
    total = args.n_train + args.n_valid + args.n_test
    n_quran = min(int(total * args.quran_share), len(quran))
    n_other = min(total - n_quran, len(other) + len(extra))

    picked = pd.concat([
        quran.sample(n=n_quran, random_state=args.seed),
        other.sample(n=min(n_other, len(other)), random_state=args.seed),
    ], ignore_index=True)

    still = total - len(picked)
    if still > 0 and len(extra):
        picked = pd.concat(
            [picked, extra.sample(n=min(still, len(extra)), random_state=args.seed)],
            ignore_index=True,
        )

    # Дедуп по ce перед сплитом — защита от утечки train/test (SPEC §3.5).
    before = len(picked)
    picked = picked.drop_duplicates(subset=["ce"])
    if len(picked) < before:
        print(f"  дедуп по ce до сплита: -{before - len(picked):,}")

    examples = make_examples(picked, tasks, rng)
    rng.shuffle(examples)
    print(f"примеров собрано: {len(examples):,}")

    if args.max_tokens:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("mlx-community/Qwen3-4B-4bit")
        examples, dropped = filter_by_length(examples, tok, args.max_tokens)
        lens = sorted(e["_ntok"] for e in examples)
        n = len(lens)
        print(f"  отсечено длиннее {args.max_tokens} токенов: {dropped}")
        print(f"  длины: медиана {lens[n // 2]}, p90 {lens[int(n * .9)]}, max {lens[-1]}")
        # Худший батч при батче 2 = 2 x max. ~10 МБ на позицию.
        worst = 2 * lens[-1]
        print(f"  худший батч при --batch-size 2: {worst:,} позиций "
              f"(~{worst * 0.010:.1f} ГБ активаций)")

    n_test, n_valid = args.n_test, args.n_valid
    test = examples[:n_test]
    valid = examples[n_test:n_test + n_valid]
    train = examples[n_test + n_valid:]

    manual = DATA / "manual" / "instructions.jsonl"
    manual_n = 0
    if manual.exists():
        with manual.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                train.insert(0, {"messages": obj["messages"], "src": "manual"})
                manual_n += 1
        print(f"рукописных пар носителя добавлено: {manual_n}")

    print(f"\ntrain={len(train):,}  valid={len(valid):,}  test={len(test):,}")

    if args.dry_run:
        print("\n[dry-run] ничего не записано. Примеры:")
        for e in train[:3]:
            for m in e["messages"]:
                print(f"    {m['role']:<9} {m['content'][:88]}")
            print()
        return 0

    # Пишем только messages — лишние ключи mlx-lm игнорирует, но чище без них.
    for name, rows in [("train", train), ("valid", valid), ("test", test)]:
        path = DATA / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for e in rows:
                fh.write(json.dumps({"messages": e["messages"]}, ensure_ascii=False) + "\n")
        n, problems = validate_jsonl(path)
        status = "OK" if not problems else f"{len(problems)} ПРОБЛЕМ"
        print(f"  {path.relative_to(ROOT)}  {n:,} строк  [{status}]")
        for p in problems[:5]:
            print(f"      {p}")

    print("\nготово. Обучение:")
    print("  uv run mlx_lm.lora --model mlx-community/Qwen3-4B-4bit --train --data ./data ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
