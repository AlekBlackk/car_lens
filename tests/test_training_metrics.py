"""Тесты метрик закрытого множества (M-1.2, §4.4 дизайна)."""

import pytest

from training.metrics import aggregate

# Классов восемь, а не три: при трёх классах список top-5 — перестановка всего
# множества, истинный класс попадает в него по построению, и `top5_model`
# оказывается равен 1.0 независимо от поведения кода. Тот же дефект чинился в
# Task 7 для `tests/test_training_model.py`.
ИНДЕКС = {
    0: "acura__rl",
    1: "acura__tl",
    2: "audi__a4",
    3: "audi__tt_rs",
    4: "bmw__m3",
    5: "bmw__x5",
    6: "chevrolet__corvette",
    7: "chevrolet__express",
}

# Марка приходит отдельной картой, из каталога, а не выводится из `class_id`:
# отсюда и человеческое написание, которого в идентификаторе нет.
МАРКИ: dict[int, str | None] = {
    0: "Acura",
    1: "Acura",
    2: "Audi",
    3: "Audi",
    4: "BMW",
    5: "BMW",
    6: "Chevrolet",
    7: "Chevrolet",
}

# Каталог с платформенными группами §13.2. У одномарочной группы марка есть,
# у разнобрендовой её нет вовсе — ровно те два случая, которых префикс
# `class_id` описать не может.
ИНДЕКС_С_ГРУППАМИ = {
    0: "grp__kei_duo",
    1: "toyota__corolla",
    2: "grp__coolray",
    3: "grp__kei_trio",
    4: "bmw__x5",
}
МАРКИ_С_ГРУППАМИ: dict[int, str | None] = {
    0: "Toyota",
    1: "Toyota",
    2: None,
    3: None,
    4: "BMW",
}

# Выборка на все четыре исхода: верный top1; промах top1 при попадании в top5 и
# промахе по марке; полный промах, где истинного класса в top5 нет вовсе; промах
# top1 при верной марке.
ПРЕДСКАЗАНИЯ = [
    [0, 1, 2, 3, 4],
    [1, 2, 3, 4, 5],
    [6, 7, 0, 1, 2],
    [7, 6, 5, 4, 3],
]
ЦЕЛИ = [0, 5, 4, 6]


def test_марка_берётся_из_каталога_а_не_из_идентификатора() -> None:
    """Одномарочная группа и обычная модель той же марки — попадание по марке.

    `class_id` группы начинается с `grp__`, поэтому по префиксу марки вышли бы
    разными и верный ответ по марке считался бы промахом. Настоящая марка
    приходит из каталога (§13.2), где у такой группы она одна на всех
    участников.
    """
    метрики = aggregate(
        [[0, 1, 2, 3, 4]],
        [1],
        idx_to_class_id=ИНДЕКС_С_ГРУППАМИ,
        idx_to_brand=МАРКИ_С_ГРУППАМИ,
    )
    assert метрики.top1_model == pytest.approx(0.0)
    assert метрики.top1_brand == pytest.approx(1.0)


def test_две_разнобрендовые_группы_не_считаются_одной_маркой() -> None:
    """Пустая марка не совпадает ни с чем, включая другую пустую.

    Две разные разнобрендовые группы — это две группы, а не «одна марка»: по
    префиксу у обеих вышло бы `grp`, и промах засчитался бы попаданием.
    """
    метрики = aggregate(
        [[2, 0, 1, 3, 4]],
        [3],
        idx_to_class_id=ИНДЕКС_С_ГРУППАМИ,
        idx_to_brand=МАРКИ_С_ГРУППАМИ,
    )
    assert метрики.top1_model == pytest.approx(0.0)
    assert метрики.top1_brand == pytest.approx(0.0)


def test_группа_без_марки_и_настоящая_марка_не_совпадают() -> None:
    """Проверяются оба направления: пустая марка не совпадает и с непустой."""
    группа_вместо_модели = aggregate(
        [[2, 0, 1, 3, 4]],
        [4],
        idx_to_class_id=ИНДЕКС_С_ГРУППАМИ,
        idx_to_brand=МАРКИ_С_ГРУППАМИ,
    )
    модель_вместо_группы = aggregate(
        [[4, 0, 1, 2, 3]],
        [2],
        idx_to_class_id=ИНДЕКС_С_ГРУППАМИ,
        idx_to_brand=МАРКИ_С_ГРУППАМИ,
    )
    assert группа_вместо_модели.top1_brand == pytest.approx(0.0)
    assert модель_вместо_группы.top1_brand == pytest.approx(0.0)


def test_несогласованные_карты_класса_и_марки_отвергаются() -> None:
    """Марка приходит второй картой, и разъехаться она может только с первой.

    Молча это дало бы `KeyError` из середины цикла — на классе, которому не
    повезло встретиться в выборке, — или вовсе не дало бы ничего.
    """
    with pytest.raises(ValueError, match="idx_to_brand"):
        aggregate([[0, 1, 2, 3, 4]], [0], idx_to_class_id=ИНДЕКС, idx_to_brand={0: "Acura"})


def test_метрики_на_смешанной_выборке() -> None:
    """Все три метрики строго между нулём и единицей и посчитаны вручную.

    Именно этот тест ловит подмену любой из трёх проверок попадания на `True`:
    ни одно из трёх чисел не равно 1.0.
    """
    метрики = aggregate(ПРЕДСКАЗАНИЯ, ЦЕЛИ, idx_to_class_id=ИНДЕКС, idx_to_brand=МАРКИ)
    assert метрики.top1_model == pytest.approx(0.25)
    assert метрики.top5_model == pytest.approx(0.75)
    assert метрики.top1_brand == pytest.approx(0.5)


def test_истинного_класса_нет_в_top5() -> None:
    """Единственная строка, где ответа нет в списке: `top5_model` обязан быть нулём."""
    метрики = aggregate([[1, 2, 3, 4, 5]], [0], idx_to_class_id=ИНДЕКС, idx_to_brand=МАРКИ)
    assert метрики.top1_model == pytest.approx(0.0)
    assert метрики.top5_model == pytest.approx(0.0)


def test_идеальные_предсказания() -> None:
    метрики = aggregate(
        [[0, 1, 2, 3, 4], [5, 4, 3, 2, 1]], [0, 5], idx_to_class_id=ИНДЕКС, idx_to_brand=МАРКИ
    )
    assert метрики.top1_model == pytest.approx(1.0)
    assert метрики.top5_model == pytest.approx(1.0)
    assert метрики.top1_brand == pytest.approx(1.0)


def test_ошибка_внутри_марки_портит_модель_но_не_марку() -> None:
    # Ответ acura__tl вместо acura__rl: модель неверна, марка верна.
    метрики = aggregate([[1, 0, 2, 3, 4]], [0], idx_to_class_id=ИНДЕКС, idx_to_brand=МАРКИ)
    assert метрики.top1_model == pytest.approx(0.0)
    assert метрики.top1_brand == pytest.approx(1.0)
    assert метрики.top5_model == pytest.approx(1.0)


def test_top5_считается_по_всему_списку() -> None:
    # Истинный класс на последнем месте списка: top1 промах, top5 попадание.
    метрики = aggregate([[5, 4, 3, 2, 0]], [0], idx_to_class_id=ИНДЕКС, idx_to_brand=МАРКИ)
    assert метрики.top1_model == pytest.approx(0.0)
    assert метрики.top5_model == pytest.approx(1.0)


def test_recall_по_классам() -> None:
    метрики = aggregate(
        [[0, 1, 2, 3, 4], [1, 0, 2, 3, 4], [6, 0, 1, 2, 3]],
        [0, 0, 6],
        idx_to_class_id=ИНДЕКС,
        idx_to_brand=МАРКИ,
    )
    assert метрики.per_class_recall["acura__rl"] == pytest.approx(0.5)
    assert метрики.per_class_recall["chevrolet__corvette"] == pytest.approx(1.0)
    # Класс, которого нет в выборке, в recall не попадает: делить не на что.
    assert "acura__tl" not in метрики.per_class_recall


def test_пустая_выборка_отвергается() -> None:
    with pytest.raises(ValueError, match="пустая выборка"):
        aggregate([], [], idx_to_class_id=ИНДЕКС, idx_to_brand=МАРКИ)


def test_несовпадение_длин_отвергается() -> None:
    with pytest.raises(ValueError, match="длин"):
        aggregate([[0, 1, 2, 3, 4]], [0, 1], idx_to_class_id=ИНДЕКС, idx_to_brand=МАРКИ)


def test_неизвестный_индекс_отвергается() -> None:
    with pytest.raises(KeyError):
        aggregate([[9, 1, 2, 3, 4]], [9], idx_to_class_id=ИНДЕКС, idx_to_brand=МАРКИ)


def test_неинъективный_индекс_отвергается() -> None:
    """Два индекса с одним `class_id` склеили бы recall и уменьшили число классов."""
    двойник = {**ИНДЕКС, 7: "chevrolet__corvette"}
    with pytest.raises(ValueError, match="неинъективен"):
        aggregate([[0, 1, 2, 3, 4]], [0], idx_to_class_id=двойник, idx_to_brand=МАРКИ)


@pytest.mark.parametrize("ширина", [1, 3, 4])
def test_ширина_меньше_пяти_отвергается(ширина: int) -> None:
    """Список из трёх индексов — это top-3, и считать его как top-5 нельзя."""
    with pytest.raises(ValueError, match="top5 требует 5"):
        aggregate([list(range(ширина))], [0], idx_to_class_id=ИНДЕКС, idx_to_brand=МАРКИ)


def test_ширина_больше_пяти_допустима_но_считается_по_первым_пяти() -> None:
    """Длинный список — законный вход, но шестое место в top-5 не попадает.

    Истинный класс стоит седьмым: `top5_model` обязан быть нулём, иначе метрика
    под именем top-5 считает top-N по фактической длине списка.
    """
    метрики = aggregate([[1, 2, 3, 4, 5, 6, 0]], [0], idx_to_class_id=ИНДЕКС, idx_to_brand=МАРКИ)
    assert метрики.top1_model == pytest.approx(0.0)
    assert метрики.top5_model == pytest.approx(0.0)
