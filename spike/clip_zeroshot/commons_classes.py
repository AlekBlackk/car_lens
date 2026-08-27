"""Состав Commons-выборки, зафиксированный §5.2 спеки ДО отбора фотографий.

Обиходные имена сюда не годятся: «классика» в каталоге — четыре класса, «Нива» — три
(`niva_2121`, `niva_2131`, `niva_travel`), и по такому описанию G3 и G4 невычислимы.
Подменять класс другим при недоборе фотографий запрещено — недобор фиксируется отчётом.
"""

from __future__ import annotations

# Порог G4: если модель не знает этих имён — метод мёртв на нашем рынке.
CORE_G4 = ("lada__2107", "lada__granta", "lada__niva_2121", "uaz__patriot")

# Контроль G2: имена, которые модель заведомо знает. Провал здесь означает битую
# выборку, а не провал метода.
CONTROL = ("toyota__camry", "volkswagen__polo", "kia__rio")

# Кандидаты §5.2 дизайна каталога: схлопывать их решает confusion-замер на M3, а не
# впечатление. Оба участника пары обязаны быть в выборке, иначе срез невычислим.
REBADGE_PAIRS = (
    ("renault__duster", "nissan__terrano"),
    ("lada__niva_2121", "lada__niva_travel"),
    ("lada__niva_travel", "chevrolet__niva"),
    ("lada__largus", "renault__logan_mcv"),
    ("datsun__on_do", "lada__granta"),
)

COMMONS_CLASSES = (
    # ядро, порог G4
    "lada__2107",
    "lada__granta",
    "lada__niva_2121",
    "uaz__patriot",
    # партнёры по ребейдж-парам §5.2
    "lada__niva_travel",
    "chevrolet__niva",
    "lada__largus",
    "renault__logan_mcv",
    "renault__duster",
    "nissan__terrano",
    "datsun__on_do",
    # отечественные прочие
    "uaz__bukhanka",
    "gaz__gazel",
    # китайские
    "grp__coolray",
    "grp__atlas_pro",
    "grp__tiggo7",
    "grp__jac_js4",
    "haval__jolion",
    "chery__tiggo_4",
    "changan__cs35_plus",
    "omoda__c5",
    "exeed__txl",
    # контроль
    "toyota__camry",
    "volkswagen__polo",
    "kia__rio",
)

# 25 всего, из них контроль 3 — значит RU + CN ровно 22, как требует порог G3.
RU_CN = tuple(c for c in COMMONS_CLASSES if c not in CONTROL)
