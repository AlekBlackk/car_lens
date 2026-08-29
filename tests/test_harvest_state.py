# tests/test_harvest_state.py
from pathlib import Path

from pipeline.harvest.state import (
    ACCEPTED_TARGET,
    SEEN_BUDGET,
    HarvestState,
    is_class_done,
    load_state,
    save_state,
)


def test_отсутствующий_файл_даёт_пустое_состояние(tmp_path: Path) -> None:
    состояние = load_state(tmp_path / "state.json")
    assert состояние.seen == {} and состояние.accepted == {}
    assert состояние.done_items == set()


def test_состояние_переживает_запись_и_чтение(tmp_path: Path) -> None:
    путь = tmp_path / "state.json"
    save_state(
        путь,
        HarvestState(
            seen={"lada__granta": 42},
            accepted={"lada__granta": 3},
            tokens={"lada__granta|category:Lada Granta": "page|1"},
            done_items={"commons:12345"},
        ),
    )
    состояние = load_state(путь)
    assert состояние.seen["lada__granta"] == 42
    assert состояние.tokens["lada__granta|category:Lada Granta"] == "page|1"
    assert состояние.done_items == {"commons:12345"}


def test_запись_атомарна(tmp_path: Path) -> None:
    """Оборванная запись состояния хуже её отсутствия: она путает бюджеты молча."""
    путь = tmp_path / "state.json"
    save_state(путь, HarvestState({}, {}, {}, set()))
    assert not list(tmp_path.glob("*.tmp"))


def test_класс_закрыт_по_бюджету_просмотра() -> None:
    состояние = HarvestState({"lada__granta": SEEN_BUDGET}, {"lada__granta": 2}, {}, set())
    assert is_class_done(состояние, "lada__granta")


def test_класс_закрыт_по_набранным_фото() -> None:
    состояние = HarvestState({"lada__granta": 5}, {"lada__granta": ACCEPTED_TARGET}, {}, set())
    assert is_class_done(состояние, "lada__granta")


def test_недобранный_класс_не_закрыт() -> None:
    состояние = HarvestState({"lada__granta": 199}, {"lada__granta": 14}, {}, set())
    assert not is_class_done(состояние, "lada__granta")
