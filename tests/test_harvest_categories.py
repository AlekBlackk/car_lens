import json
from collections.abc import Mapping

from pipeline.harvest.categories import MAX_SUBCAT_REQUESTS, expand
from pipeline.harvest.commons import HttpResponse


class ЗаписанныйТранспорт:
    """Подставной транспорт: отдаёт ответ по имени запрошенной категории."""

    def __init__(self, дерево: dict[str, list[str]]) -> None:
        self.дерево = дерево
        self.вызовы: list[str] = []

    def get(self, url: str, params: Mapping[str, str]) -> HttpResponse:
        заголовок = params["gcmtitle"].removeprefix("Category:")
        self.вызовы.append(заголовок)
        страницы = {
            str(индекс): {"pageid": индекс, "ns": 14, "title": f"Category:{имя}"}
            for индекс, имя in enumerate(self.дерево.get(заголовок, []), start=1)
        }
        тело = {"query": {"pages": страницы}} if страницы else {"batchcomplete": ""}
        return HttpResponse(200, json.dumps(тело).encode("utf-8"))


def test_плоская_категория_возвращает_саму_себя() -> None:
    транспорт = ЗаписанныйТранспорт({"Kia Rio (DC)": []})

    итог = expand(транспорт, "Kia Rio (DC)", sleep=lambda _: None)

    assert итог.categories == ("Kia Rio (DC)",)
    assert итог.requests == 1


def test_поколения_берутся_родитель_первым() -> None:
    транспорт = ЗаписанныйТранспорт({"Kia Rio": ["Kia Rio (DC)", "Kia Rio (QB)"]})

    итог = expand(транспорт, "Kia Rio", sleep=lambda _: None)

    assert итог.categories == ("Kia Rio", "Kia Rio (DC)", "Kia Rio (QB)")


def test_стоп_маркер_отсекает_чужой_домен() -> None:
    транспорт = ЗаписанныйТранспорт(
        {
            "Ford Focus": [
                "European Ford Focus",
                "Ford Focus in competition",
                "Ford Focus taxis",
                "Crashed Ford Focus",
            ]
        }
    )

    итог = expand(транспорт, "Ford Focus", sleep=lambda _: None)

    assert итог.categories == ("Ford Focus", "European Ford Focus")
    assert итог.rejected_stopword == 3


def test_потомок_без_имени_родителя_не_берётся() -> None:
    транспорт = ЗаписанныйТранспорт({"Lada Priora": ["Lada Kalina", "Lada Priora sedan"]})

    итог = expand(транспорт, "Lada Priora", sleep=lambda _: None)

    assert итог.categories == ("Lada Priora", "Lada Priora sedan")


def test_exclude_режет_поколение_и_считается() -> None:
    транспорт = ЗаписанныйТранспорт(
        {"Hyundai Accent": ["Hyundai Accent (RB)", "Hyundai Accent (LC)"]}
    )

    итог = expand(
        транспорт, "Hyundai Accent", exclude=["Hyundai Accent (RB)"], sleep=lambda _: None
    )

    assert итог.categories == ("Hyundai Accent", "Hyundai Accent (LC)")
    assert итог.rejected_exclude == 1


def test_include_пробивает_правило_допуска() -> None:
    транспорт = ЗаписанныйТранспорт({"Hyundai Solaris": []})

    итог = expand(
        транспорт,
        "Hyundai Solaris",
        include=["Hyundai Accent (RB)", "Solaris HS"],
        sleep=lambda _: None,
    )

    assert итог.categories == ("Hyundai Solaris", "Hyundai Accent (RB)", "Solaris HS")


def test_глубина_два_спускается_через_промежуточную() -> None:
    транспорт = ЗаписанныйТранспорт(
        {"Ford Focus": ["European Ford Focus"], "European Ford Focus": ["European Ford Focus Mk1"]}
    )

    итог = expand(транспорт, "Ford Focus", depth=2, sleep=lambda _: None)

    assert итог.categories == ("Ford Focus", "European Ford Focus", "European Ford Focus Mk1")


def test_глубина_один_не_спускается_глубже() -> None:
    транспорт = ЗаписанныйТранспорт(
        {"Ford Focus": ["European Ford Focus"], "European Ford Focus": ["European Ford Focus Mk1"]}
    )

    итог = expand(транспорт, "Ford Focus", depth=1, sleep=lambda _: None)

    assert итог.categories == ("Ford Focus", "European Ford Focus")
    assert транспорт.вызовы == ["Ford Focus"]


def test_глубина_ноль_не_ходит_вовсе() -> None:
    транспорт = ЗаписанныйТранспорт({"Lada Priora": ["Lada Priora sedan"]})

    итог = expand(транспорт, "Lada Priora", depth=0, sleep=lambda _: None)

    assert итог.categories == ("Lada Priora",)
    assert транспорт.вызовы == []


def test_бюджет_обрывает_обход_и_отмечается() -> None:
    дерево = {"Дерево": [f"Дерево {индекс}" for индекс in range(30)]}
    дерево |= {
        f"Дерево {индекс}": [f"Дерево {индекс} {буква}" for буква in "аб"] for индекс in range(30)
    }
    транспорт = ЗаписанныйТранспорт(дерево)

    итог = expand(транспорт, "Дерево", depth=2, sleep=lambda _: None)

    assert итог.requests == MAX_SUBCAT_REQUESTS
    assert итог.budget_exhausted is True
