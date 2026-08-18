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

## Скелет M-1: экспорт и паритет

Экспорт живёт в **отдельной** venv: `litert-torch` тянет собственный стек
конвертера (jax, xdsl, torchao), и держать его рядом с обучающим значило бы
разбирать конфликты версий при каждом обновлении любой из половин (§13.4
спецификации). Механизм — переменная `UV_PROJECT_ENVIRONMENT`: тот же проект и
те же пины, но другой каталог venv. Один раз при настройке машины:

```bash
UV_PROJECT_ENVIRONMENT=~/.venvs/carlens-export uv sync --locked --group export
```

Флаг `--group export` обязателен и у `uv run`, а не только у `uv sync`: перед
запуском uv приводит окружение к состоянию проекта, а по умолчанию включает
только группы из `default-groups` (у нас — `dev`). Без флага он вычистил бы
`litert-torch` прямо перед конвертацией. `--locked` запрещает молча переписать
`uv.lock` — тот самый файл, которым §13.4 держит цепочку конвертации.

Конвертация чекпоинта в два файла LiteRT — эталонный fp32 и рабочий INT8:

```bash
UV_PROJECT_ENVIRONMENT=~/.venvs/carlens-export uv run --locked --group export \
  python -m export.to_tflite \
  --checkpoint $CARLENS_DATA_ROOT/runs/m1_2_train_<дата>/best.pt
```

Кладёт в `runs/m1_2_train_<дата>/export/`: `classifier_fp32.tflite`,
`classifier_int8.tflite` и `export_report.json`. В отчёте — версии тулчейна,
состав калибровки, sha256 и размеры моделей, дословные
`input_details`/`output_details` интерпретатора и быстрая самопроверка на 32
кропах. Границы INT8-графа получаются `int8`, а не float32, поэтому `scale` и
`zero_point` из `export_report.json` — не справка, а входные данные для того,
кто будет кормить модель.

Замер паритета: PyTorch против обоих `.tflite` на одной выборке в одном
порядке, метрика — top-1/top-5 по косинусу к центроидам индекса, а не по голове
ArcFace (голова на устройство не едет).

```bash
UV_PROJECT_ENVIRONMENT=~/.venvs/carlens-export uv run --locked --group export \
  python -m export.verify_parity \
  --export $CARLENS_DATA_ROOT/runs/m1_2_train_<дата>/export \
  --index $CARLENS_DATA_ROOT/runs/m1_2_train_<дата>/index \
  --checkpoint $CARLENS_DATA_ROOT/runs/m1_2_train_<дата>/best.pt --device cpu
```

`--device cpu` здесь не экономия, а условие корректности замера. Жёсткий порог
сырого слоя существует, чтобы поймать ошибку конвертации, поэтому база не должна
вносить собственный шум того же порядка, что порог. На CPU обе стороны считают в
честной одинарной точности, и расхождение садится на пол точности представления.
Приёмочный прогон вехи сделан именно так.

Кладёт `parity.json` рядом с моделями: база PyTorch, метрики трёх рантаймов,
сырые и косинусные расхождения, доля смен top-1 и дельты метрик в пунктах
процента.

**Ненулевой код возврата означает провал порога приёмки**, а не сбой прогона:
`parity.json` пишется до выхода, и числа доступны в любом случае. Порогов два —
расхождение fp32-TFLite с PyTorch на ненормированных эмбеддингах (жёсткий,
диагноз «сломан граф или препроцессинг») и просадка `top1_model` у INT8
относительно базы (мягкий, диагноз про качество квантизации). Какой именно
провален, пишется в stderr и в `thresholds.failures`.

**Осторожно: `--device` по умолчанию `cuda`, и на Ampere это даёт ложный провал
жёсткого порога.** Torch включает TF32 для свёрток сам, мантисса TF32 — десять
бит, то есть погрешность самой базы выходит порядка 1e-3 при пороге 1e-3. На
исправном графе порог срабатывает, а stderr при этом утверждает «сломан граф».
Замерено на этом проекте, сплит `val`, один и тот же экспорт, менялся только
флаг: `--device cpu` — 6.2e-06, `--device cuda` с `NVIDIA_TF32_OVERRIDE=0` —
8.1e-04, `--device cuda` по умолчанию — 3.8e-03. Отсюда правило: приёмка меряется
только на CPU, а провал на `cuda` сначала перепроверяется на CPU и лишь потом
считается поломкой конвертации. Смотреть при этом надо на косинус — у исправного
fp32-графа он не опускался ниже 0.999997 ни при одной из трёх баз. То, что дефолт
флага остался `cuda`, — известный долг вехи, а не рекомендация.

Флаг `--embedder tflite:<путь>` у сборщика индекса применяется **только** по
факту провала второго порога: он пересчитывает центроиды на выходах
квантованной модели, чтобы индекс и модель перестали расходиться. Результат
пишется в отдельный каталог, исходный `index/` не трогается, и рядом появляется
`index_manifest.json` с происхождением.

```bash
UV_PROJECT_ENVIRONMENT=~/.venvs/carlens-export uv run --locked --group export \
  python -m training.build_index \
  --checkpoint $CARLENS_DATA_ROOT/runs/m1_2_train_<дата>/best.pt \
  --embedder tflite:$CARLENS_DATA_ROOT/runs/m1_2_train_<дата>/export/classifier_int8.tflite \
  --out $CARLENS_DATA_ROOT/runs/m1_2_train_<дата>/index_int8 --device cpu
```

Всё в `export/` и `index_int8/` — расходный материал прогона рядом с весами, по
которым собрано, и в git не хранится (решение Р-22, как и индекс). `.tflite`
весит десятки мегабайт и меняется при каждом прогоне; в приложение модель
копируется явным шагом сборки на M-1.5.

**Зелёный CI не является свидетельством того, что экспорт работает.** Ни одна
джоба экспорт не запускает: джоб `python` гоняет `pytest -m "not export"`, а
venv экспорта в CI не существует вовсе — там ставится одна группа `dev`, в
которой нет ни torch, ни `litert-torch`. Цепочку конвертации проверяет только
ручной прогон команд выше плюс помеченные тесты, и запускать их надо в venv
экспорта:

```bash
UV_PROJECT_ENVIRONMENT=~/.venvs/carlens-export uv run --locked --group export \
  pytest -m export
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
| ai-edge-litert | 2.1.6 | Рантайм LiteRT, которым считается паритет; без пина уехал бы вслед за `litert-torch` |
| torchao | 0.18.0 | В torch 2.12 PT2E-квантизация живёт там, а не в `torch.ao`; пин держит эти имена на месте |
| Node | 20.18.0 | Требование React Native 0.76 |
