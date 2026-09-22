"""Классификация источников чеченских данных.

Общая логика для scripts/filter_dictionary.py и scripts/sample_for_review.py.
Живёт в пакете, а не в скриптах, намеренно: если фильтр и выборка для проверки
разъедутся, носитель будет судить не о том тексте, который реально попадёт в датасет.
"""

from __future__ import annotations

import re

# Религиозные тексты: Библия (Синодальный перевод / Институт перевода Библии)
# и Коран (переводы Магомедова и Ибрагимова, каждый в трёх русских редакциях:
# Адель, Кулиев, Османов).
RELIGIOUS_RE = re.compile(
    r"quran|bible|synodal|ibragimov|magomedov|kuliev|osmanov|adel", re.IGNORECASE
)

# num2words / baltoslav — проговаривание чисел прописью («семьсот двенадцать целых
# пять сотых»). Формально это текст, по сути — не речь.
JUNK_RE = re.compile(r"num2words|baltoslav", re.IGNORECASE)

# Ссылка на стих в конце источника: "Quran ..., 80:19" или "... 12:3-5".
_VERSE_RE = re.compile(r"[,\s]+\d+:\d+(-\d+)?\s*$")


def normalize_source(source: object) -> str:
    """Убирает ссылку на стих, чтобы источники группировались корректно."""
    text = str(source).strip()
    text = _VERSE_RE.sub("", text)
    return re.sub(r"[\s,]+$", "", text).strip()


def is_religious(source: object) -> bool:
    return bool(RELIGIOUS_RE.search(str(source)))


def is_junk(source: object) -> bool:
    return bool(JUNK_RE.search(str(source)))


def is_quran(source: object) -> bool:
    return "quran" in str(source).lower()


def is_bible(source: object) -> bool:
    s = str(source).lower()
    return "bible" in s or "synodal" in s


def words(text: object) -> int:
    return len(str(text).split())


# Порядок предпочтения русских редакций Корана, от лучшей к худшей.
#
# В nmd_171k каждый аят лежит в 6 вариантах: 2 чеченских перевода (Магомедов,
# Ибрагимов) x 3 русские редакции. Русские редакции НЕ равноценны:
#   Kuliev  — официальный «перевод смыслов», грамотный русский;
#   Osmanov — научный перевод, тоже чистый, но местами без точек;
#   Adel    — черновой/машинный: «Аллах - нет бога , кроме Него,  - Живой, Сущий ;»
#             (пробелы перед запятыми, рваные тире).
# При дедупликации по `ce` надо выбирать Kuliev, а не то, что лежит первым в файле.
# Именно на этом легко потерять качество: дедуп «по порядку» оставлял Adel.
RUSSIAN_EDITION_PREFERENCE = ("kuliev", "osmanov", "adel")


def edition_rank(source: object) -> int:
    """Чем меньше — тем предпочтительнее русская редакция. Не Коран — в конце."""
    s = str(source).lower()
    for i, name in enumerate(RUSSIAN_EDITION_PREFERENCE):
        if name in s:
            return i
    return len(RUSSIAN_EDITION_PREFERENCE)


# Чеченские переводы Корана в nmd_171k. Их два, и они дают РАЗНЫЙ чеченский текст
# для одного и того же русского аята. Для задачи ru->ce это противоречивые цели,
# поэтому решением носителя берём ровно один перевод (Магомедова).
CHECHEN_TRANSLATORS = ("magomedov", "ibragimov")


def chechen_translator(source: object) -> str | None:
    """Возвращает имя чеченского переводчика Корана или None для не-коранических строк."""
    s = str(source).lower()
    for name in CHECHEN_TRANSLATORS:
        if name in s:
            return name
    return None
