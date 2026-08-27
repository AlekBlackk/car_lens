import numpy as np

from spike.clip_zeroshot.run_commons import собрать_итог


def test_срезы_считаются_по_своим_классам() -> None:
    """Контроль, RU+CN и четвёрка G4 меряются раздельно: смешать их значит спрятать провал."""
    индекс = {"toyota__camry": 0, "lada__granta": 1, "uaz__patriot": 2}
    метки = np.array([0, 0, 1, 1, 2, 2])
    попадания = np.array([True, True, True, False, False, False])

    итог = собрать_итог(попадания, метки, индекс, {"top1": 0.8, "top3": 0.9})

    assert итог.control_top3 == 1.0
    assert итог.ru_cn_top3 == 0.25
    assert итог.core_top3["lada__granta"] == 0.5
    assert итог.core_top3["uaz__patriot"] == 0.0
    assert итог.stanford_top1 == 0.8


def test_класс_без_фотографий_в_срез_не_попадает() -> None:
    индекс = {"toyota__camry": 0, "lada__granta": 1}
    метки = np.array([0])
    попадания = np.array([True])

    итог = собрать_итог(попадания, метки, индекс, {"top1": 0.8, "top3": 0.9})

    assert "lada__granta" not in итог.core_top3
    assert итог.control_top3 == 1.0
