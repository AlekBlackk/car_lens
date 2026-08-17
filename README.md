# CarLens

Распознавание марки и модели автомобиля с камеры телефона, полностью on-device.

Спецификация: [docs/SPEC.md](docs/SPEC.md). Решения на старте:
[docs/superpowers/specs/2026-08-15-carlens-decisions.md](docs/superpowers/specs/2026-08-15-carlens-decisions.md).

## Окружение

Питоновский конвейер работает в WSL2 Ubuntu 24.04, приложение собирается
на Windows (§16.1 спецификации). Драйвер NVIDIA ставится **только на
Windows** — установка драйвера внутри WSL ломает проброс GPU.

Окружение живёт в ext4, а не в дереве репозитория: на `/mnt/c` uv копирует
каждый файл вместо хардлинка из кэша, и пересборка растягивается с минут на
часы. Один раз при настройке машины:

```bash
UV_PROJECT_ENVIRONMENT=~/.venvs/carlens uv sync --group dev --group train
ln -s ~/.venvs/carlens .venv
```

Дальше `uv run` и `pytest` работают как обычно — они находят интерпретатор
по ссылке `.venv`. Если `.venv` вдруг стал настоящим каталогом, ссылку надо
восстановить теми же двумя командами.

Каталог данных задаётся переменной `CARLENS_DATA_ROOT`, по умолчанию
`~/carlens-data`. Держать его в файловой системе WSL, не в `/mnt/c`.

## Проверки

```bash
uv run ruff check . && uv run mypy && uv run pytest
```

## Скелет M-1: данные Stanford Cars

Датасет и веса детектора скачиваются один раз и в git не попадают.

```bash
uv run --group train python -m scripts.fetch_stanford
```

```bash
uv run --group train python -m scripts.build_stanford_splits
```

Второй скрипт пишет `train/val/test.parquet` в `$CARLENS_DATA_ROOT/splits`
и отчёт о прогоне в `$CARLENS_DATA_ROOT/runs/m1b_ingest_<дата>/`.

Каталог скелета `catalog/stanford.yaml` — расходный и к продакшн-каталогу
`canonical.yaml` отношения не имеет (решение Р-7). Он генерируется и
перезаписывается командой:

```bash
uv run python -m scripts.gen_stanford_catalog
```

## Скелет M-1: обучение

```bash
uv run --group train python -m training.train
```

Пишет прогон в `$CARLENS_DATA_ROOT/runs/m1_2_train_<дата>/`: карточку,
метрики по эпохам, манифест и два чекпоинта — `best.pt` по `val top1_model`
и `last.pt`. Конфиг по умолчанию — `training/configs/skeleton_stanford.yaml`.

Санитарный прогон перед полным (три эпохи, десять шагов на эпоху):

```bash
uv run --group train python -m training.train --epochs 3 --limit-batches 10
```

## Скелет M-1: индекс эталонов

Сборка `catalog_index.bin` по §13.2 из чекпоинта обучения:

```bash
uv run --group train python -m training.build_index \
  --checkpoint $CARLENS_DATA_ROOT/runs/m1_2_train_<дата>/best.pt --device cuda
```

Пишет в `runs/m1_2_train_<дата>/index/`: `catalog_index.bin`, спутник
`catalog_meta.json` (порядок классов, sha256 которого лежит в заголовке
бинарника) и `report.json` с размерами и хешами. Индекс — расходный артефакт
прогона рядом с весами, по которым собран, и в git не хранится (решение Р-22);
в приложение он копируется явным шагом сборки на M-1.5.

Оценка индекса — top-1/top-5 по косинусу к центроидам, той же арифметикой
метрик, что мерила голову ArcFace:

```bash
uv run --group train python -m scripts.eval_index \
  --index $CARLENS_DATA_ROOT/runs/m1_2_train_<дата>/index \
  --checkpoint $CARLENS_DATA_ROOT/runs/m1_2_train_<дата>/best.pt --device cuda
```

Кладёт `index_metrics.json` рядом с индексом.

Читалка формата на TypeScript живёт в `app/src/vision/catalogIndex.ts` и
проверяется на коммитнутой фикстуре `tests/fixtures/index/`:

```bash
cd app && npm ci && npm run build && npm test
```

Прочитать этим же кодом настоящий собранный индекс:

```bash
node app/scripts/dump-index.mjs $CARLENS_DATA_ROOT/runs/m1_2_train_<дата>/index
```

Фикстуру пересобирает генератор — после любой правки формата это обязательно,
иначе `tests/test_index_fixture.py` падает:

```bash
uv run python -m scripts.gen_index_fixture
```

## Зафиксированные версии

Точные версии живут в `uv.lock`. Ключевые пины и причины — §13.4
спецификации.

| Пакет | Версия | Почему зафиксирован |
|---|---|---|
| Python | 3.11 | Нижняя граница `litert-torch` — 3.10; 3.11 даёт запас и стабильные колёса |
| torch | 2.12.0 | `litert-torch` 0.9.3 требует `>=2.4.0,<2.13.0` |
| torchvision | 0.27.0 | Парная к torch 2.12 |
| timm | 1.0.28 | Источник весов MobileNetV4 |
| litert-torch | 0.9.3 | Конвертация в TFLite; преемник `ai-edge-torch` |
| Node | 20.18.0 | Требование React Native 0.76 |
