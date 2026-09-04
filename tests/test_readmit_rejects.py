"""Привод переприёма: ядро, порядок фиксации и CLI.

Ни сети, ни весов: `HarvestDeps` приходит заглушками из `tests/test_harvest_run.py`, а
`build_deps` в тестах CLI подменяется — `open-clip-torch` живёт только в группе `train`
(`pyproject.toml:33`), и настоящая сборка уронила бы обычный `pytest`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.crop import Detection
from pipeline.harvest.cache import cache_path, digest
from pipeline.harvest.dedup import DedupIndex
from pipeline.harvest.manifest import HarvestRow, append_row, load_manifest
from pipeline.harvest.readmit import CachedSource
from pipeline.harvest.rejects import Reject, append_reject_row
from pipeline.harvest.run import HarvestDeps
from pipeline.harvest.sources import Candidate, HarvestSource
from pipeline.harvest.state import HarvestState, load_state, save_state
from scripts.readmit_rejects import build_parser, commit, main, readmit, sha256_by_page
from tests.test_harvest_run import _jpeg, _зависимости, _кандидат


class ПадающийДетектор:
    """Роняет прогон на заданном вызове: модель обрывается посреди цикла."""

    def __init__(self, боксы: list[Detection], падать_на: int) -> None:
        self.боксы = боксы
        self.падать_на = падать_на
        self.вызовы = 0

    def detect(self, image: np.ndarray) -> list[Detection]:
        self.вызовы += 1
        if self.вызовы >= self.падать_на:
            raise RuntimeError("детектор оборвался")
        return self.боксы


def _в_кэш(root: Path, seed: int) -> str:
    """Положить кадр в кэш под его настоящим sha256 и вернуть этот sha256."""
    данные = _jpeg(seed=seed)
    sha = digest(данные)
    путь = cache_path(root, "commons", sha, "jpg")
    путь.parent.mkdir(parents=True, exist_ok=True)
    путь.write_bytes(данные)
    return sha


def _строка(file: str, pageid: str) -> HarvestRow:
    return HarvestRow(
        file=file,
        class_id="lifan__x60",
        source="commons",
        source_item_id=pageid,
        page_url="https://commons.wikimedia.org/wiki/File:X.jpg",
        file_url="https://upload.wikimedia.org/X.jpg",
        author="Иван Петров",
        license="CC BY 4.0",
        license_tier=2,
        checked="2026-09-04",
        basis="category:Lifan X60",
        sha256=file.removesuffix(".jpg").rjust(64, "0"),
        phash=1,
        clip_rank=1,
        clip_margin=0.0,
        plates=1,
        faces=0,
        bbox_x=1.0,
        bbox_y=2.0,
        bbox_w=300.0,
        bbox_h=200.0,
        bbox_area_frac=0.4,
        det_conf=0.9,
        audit="ok",
    )


def _прежняя() -> HarvestRow:
    return _строка("старая.jpg", "222")


def _новая() -> HarvestRow:
    return _строка("новая.jpg", "333")


def test_ядро_копит_строки_и_не_пишет_манифест(tmp_path: Path) -> None:
    первый, второй = _в_кэш(tmp_path, seed=1), _в_кэш(tmp_path, seed=2)
    источник = CachedSource("commons", tmp_path, {"111": первый, "222": второй})
    зависимости = _зависимости(tmp_path, источник=источник)
    кандидаты = [_кандидат(source_item_id="111"), _кандидат(source_item_id="222")]

    итог = readmit(кандидаты, зависимости, DedupIndex())

    assert итог.reasons == {"accepted": 2}
    assert len(итог.rows) == 2
    assert not зависимости.manifest_path.exists()


def test_обрыв_в_цикле_оставляет_манифест_без_изменений(tmp_path: Path) -> None:
    sha = _в_кэш(tmp_path, seed=1)
    машина = Detection(x=100.0, y=100.0, w=700.0, h=500.0, conf=0.9, cls=2)
    зависимости = _зависимости(
        tmp_path,
        источник=CachedSource("commons", tmp_path, {"111": sha, "222": sha}),
        vehicle_detector=ПадающийДетектор([машина], падать_на=2),
    )
    append_row(зависимости.manifest_path, _прежняя())
    было = зависимости.manifest_path.read_bytes()

    with pytest.raises(RuntimeError):
        readmit(
            [_кандидат(source_item_id="111"), _кандидат(source_item_id="222")],
            зависимости,
            DedupIndex(),
        )

    assert зависимости.manifest_path.read_bytes() == было
    assert not (tmp_path / "state.json").exists()


def _фиксация(tmp_path: Path, rows: list[HarvestRow], state: HarvestState) -> None:
    """Позвать `commit` со стандартной раскладкой путей внутри `tmp_path`."""
    commit(
        rows,
        state,
        manifest_path=tmp_path / "harvest_manifest.csv",
        state_path=tmp_path / "state.json",
        backup_path=tmp_path / "state.json.bak.readmit",
        verdicts_tmp=tmp_path / "verdicts.dedup.readmit.csv",
        verdicts_path=tmp_path / "verdicts.dedup.csv",
    )


def test_фиксация_дописывает_строки_не_переписывая_чужие_байты(tmp_path: Path) -> None:
    манифест = tmp_path / "harvest_manifest.csv"
    append_row(манифест, _прежняя())
    было = манифест.read_bytes()

    _фиксация(tmp_path, [_новая()], HarvestState({}, {"lifan__x60": 2}, {}, {"commons:222"}))

    стало = манифест.read_bytes()
    assert стало.startswith(было)
    assert len(load_manifest(манифест)) == 2
    assert not (tmp_path / "harvest_manifest.readmit.csv").exists()
    assert load_state(tmp_path / "state.json").accepted == {"lifan__x60": 2}


def test_копия_состояния_снимается_один_раз(tmp_path: Path) -> None:
    состояние = tmp_path / "state.json"
    состояние.write_text('{"seen": {}, "accepted": {}, "tokens": {}, "done_items": []}', "utf-8")
    первое = состояние.read_bytes()

    for набрано in (1, 2):
        _фиксация(tmp_path, [], HarvestState({}, {"lifan__x60": набрано}, {}, set()))

    # Второй прогон не затирает точку возврата уже пропатченным состоянием.
    assert (tmp_path / "state.json.bak.readmit").read_bytes() == первое


def test_карта_sha256_пропускает_кандидата_без_оригинала(tmp_path: Path) -> None:
    карта = tmp_path / "reject_index.csv"
    карта.write_text(
        "class_id,source_item_id,sha256\nlifan__x60,111,abc\nlifan__x60,222,\n", encoding="utf-8"
    )

    assert sha256_by_page(карта) == {"111": "abc"}


def test_манифест_фиксируется_раньше_состояния(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Обрыв на записи состояния оставляет манифест уже дописанным.

    Это не дефект, а выбранная сторона: состояние выводится из манифеста, поэтому повторный
    прогон досчитает его сам. Обратный порядок оставил бы состояние, обещающее строки,
    которых в манифесте нет.
    """
    манифест = tmp_path / "harvest_manifest.csv"
    append_row(манифест, _прежняя())

    def падение(path: Path, state: HarvestState) -> None:
        raise OSError("диск кончился")

    monkeypatch.setattr("scripts.readmit_rejects.save_state", падение)

    with pytest.raises(OSError):
        _фиксация(tmp_path, [_новая()], HarvestState({}, {"lifan__x60": 2}, {}, set()))

    assert len(load_manifest(манифест)) == 2
    assert not (tmp_path / "state.json").exists()


class ПодставнойПоиск:
    """Подставной `CommonsSource`: отдаёт кандидатов по заявкам, в сеть не ходит."""

    def __init__(self, transport: object) -> None:
        self.заявки: dict[str, tuple[str, str]] = {}

    def lookup(self, pages: dict[str, tuple[str, str]]) -> tuple[Candidate, ...]:
        self.заявки = dict(pages)
        return tuple(
            _кандидат(class_id=class_id, source_item_id=pageid, basis=основание)
            for pageid, (class_id, основание) in pages.items()
        )


@pytest.fixture
def стенд(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Журнал, карта и кэш на диске; сеть и сборка зависимостей подменены.

    `build_deps` подменяется, а не вызывается: `open-clip-torch` живёт только в группе
    `train`, и настоящая сборка уронила бы обычный `pytest`.
    """
    корень = tmp_path / "harvest"
    корень.mkdir()
    манифест = tmp_path / "harvest_manifest.csv"
    страницы = {"111": _в_кэш(корень, seed=1), "222": _в_кэш(корень, seed=2)}
    for pageid in страницы:
        append_reject_row(
            корень / "reject_log.csv",
            Reject(
                class_id="lada__granta",
                basis="category:Lada Granta",
                source="commons",
                source_item_id=pageid,
                reason="redaction_failed",
                checked="2026-09-01",
            ),
        )
    (корень / "reject_index.csv").write_text(
        "class_id,source_item_id,sha256\n"
        + "".join(f"lada__granta,{pageid},{sha}\n" for pageid, sha in страницы.items()),
        encoding="utf-8",
    )

    monkeypatch.setattr("scripts.readmit_rejects.UrllibTransport", lambda contact: object())
    monkeypatch.setattr("scripts.readmit_rejects.CommonsSource", ПодставнойПоиск)

    def подставная_сборка(source: HarvestSource, expand: object, **прочее: object) -> HarvestDeps:
        return _зависимости(корень, источник=source, manifest_path=манифест)

    monkeypatch.setattr("scripts.readmit_rejects.build_deps", подставная_сборка)
    return {"корень": корень, "манифест": манифест, "состояние": корень / "state.json"}


def _аргументы(стенд: dict[str, Path], *хвост: str) -> list[str]:
    return [
        "--contact",
        "mail@example.com",
        "--data-root",
        str(стенд["корень"]),
        "--manifest",
        str(стенд["манифест"]),
        *хвост,
    ]


def test_сухой_прогон_ничего_не_пишет(
    стенд: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(_аргументы(стенд)) == 0

    assert not стенд["манифест"].exists()
    assert not стенд["состояние"].exists()
    вывод = capsys.readouterr().out
    assert "сухой прогон" in вывод
    assert "accepted lada__granta: 0 -> 2" in вывод


def test_запись_дописывает_строки_и_патчит_состояние(стенд: dict[str, Path]) -> None:
    save_state(
        стенд["состояние"],
        HarvestState({"lada__granta": 12}, {}, {"k": "t"}, {"commons:9"}),
    )

    assert main(_аргументы(стенд, "--write")) == 0

    строки = load_manifest(стенд["манифест"])
    assert len(строки) == 2
    состояние = load_state(стенд["состояние"])
    assert состояние.accepted == {"lada__granta": 2}
    assert состояние.done_items == {"commons:9", "commons:111", "commons:222"}
    # Стоп-критерий не тронут: переприём ничего не просматривал.
    assert состояние.seen == {"lada__granta": 12}
    assert состояние.tokens == {"k": "t"}
    assert (стенд["корень"] / "state.json.bak.readmit").exists()


def test_регресс_accepted_останавливает_до_записи(стенд: dict[str, Path]) -> None:
    save_state(стенд["состояние"], HarvestState({}, {"kia__rio": 30}, {}, set()))
    было = стенд["состояние"].read_bytes()

    assert main(_аргументы(стенд, "--write")) == 2

    assert not стенд["манифест"].exists()
    assert стенд["состояние"].read_bytes() == было


def test_лимит_режет_число_кандидатов(стенд: dict[str, Path]) -> None:
    assert main(_аргументы(стенд, "--limit", "1", "--write")) == 0
    assert len(load_manifest(стенд["манифест"])) == 1


def test_умолчания_разбора_аргументов() -> None:
    аргументы = build_parser().parse_args(["--contact", "mail@example.com"])

    assert аргументы.reason == "redaction_failed"
    assert аргументы.source == "commons"
    assert аргументы.write is False
    assert аргументы.limit == 0
