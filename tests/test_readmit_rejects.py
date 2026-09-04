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
from pipeline.harvest.state import HarvestState, load_state
from scripts.readmit_rejects import commit, readmit, sha256_by_page
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
