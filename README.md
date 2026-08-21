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
fp32-графа он не опускался ниже 0.999997 ни при одной из трёх баз. Значение по умолчанию
у `--device` — `cpu` именно поэтому; `cuda` остаётся явным выбором того, кто про TF32
знает.

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

## Экспорт детектора M-1.5

Детектор собирается в **третьей** venv, отдельно от классификатора: ultralytics
доустанавливает конвертер сам и прямо в то окружение, где запущен, а venv обучения
и venv экспорта собраны по `uv.lock`, которым §13.4 держит цепочку конвертации.
Один раз при настройке машины:

```bash
UV_PROJECT_ENVIRONMENT=~/.venvs/carlens-yolo uv sync --locked --group yolo
```

Сам экспорт `yolo11n.pt` в fp32-граф на 416:

```bash
UV_PROJECT_ENVIRONMENT=~/.venvs/carlens-yolo uv run --locked --group yolo \
  python -m scripts.export_yolo
```

`--group yolo` обязателен и у `uv run` по той же причине, что `--group export` выше.
Там же причина, по которой ultralytics в каждом прогоне печатает `requirements:
AutoUpdate success`: uv приводит venv к `uv.lock` перед запуском и вычищает
доустановленный конвертер, а ultralytics ставит его заново (секунды из кеша). Это
ожидаемая часть прогона, а не сбой: стек конвертера в lock проекта не попадает.
Флаги `--weights`, `--imgsz` и `--out` меняют веса, размер входа и каталог, но
приёмка вехи считается на умолчаниях — `yolo11n.pt` и 416.

Кладёт в `$CARLENS_DATA_ROOT/detector/` два файла: `detector_fp32.tflite` и
`detector_report.json` — sha256 весов и графа, версия ultralytics и контракт графа
целиком. Рядом с весами при этом остаётся `yolo11n.tflite`: туда пишет сам
ultralytics, а скрипт копирует граф в каталог данных под нужным именем.

Контракт — это не формы тензоров. Два графа с одинаковыми `1×3×416×416` и `1×84×N`
расходятся раскладкой входа, единицами координат и форматом бокса, и каждое
расхождение даёт рамку не на машине при полностью зелёной сверке форм. Поэтому
единицы координат не угадываются по формам, а снимаются зондом: скрипт прогоняет
через граф один серый кадр и смотрит на диапазон координат бокса. Прогон на 8.4.120
дал `input_layout = NCHW`, `output_layout = CN` и `coord_units = normalized`. Вход
здесь `NCHW`, то есть `1×3×416×416`, а не `NHWC`, которого от LiteRT ждут по
умолчанию, — ровно та причина, по которой сторона Kotlin обязана читать раскладку из
`detector_report.json`, а не держать её константой в коде.

Лицензия ultralytics — AGPL-3.0, использование допустимо по решению Р-4, пока проект
личный. `detector_fp32.tflite` — такой же расходный артефакт прогона, что и экспорт
классификатора: в git не идёт (Р-22), в приложение копируется явным шагом сборки.

## Android-модуль M-1.5

Дерево сборки — `app/android/`: библиотека `:vision` (LiteRT, парсер индекса §13.2,
препроцессинг прямо из плоскостей `YUV_420_888`) и демонстрационное приложение `:demo`
на CameraX. Требуется JDK 17 и Android SDK 35; путь к SDK берётся из `ANDROID_HOME` или
`local.properties`.

### Стейджинг ассетов

Модели в git не хранятся (решение Р-22), и без них `:demo` не собирается вовсе —
`checkModelsManifest` роняет сборку с текстом команды. Раскладка ставит четыре файла и
`models_manifest.json` в один каталог `app/android/models/`, откуда их подхватывают оба
модуля: `:demo` — как ассеты приложения, `:vision` — как ассеты инструментальных тестов.

```bash
uv run python -m scripts.stage_assets --run $CARLENS_DATA_ROOT/runs/m1_2_train_<дата> --detector $CARLENS_DATA_ROOT/detector/detector_fp32.tflite --out app/android/models
```

Каталог назначения сносится целиком, поэтому значения по умолчанию у `--out` нет.
Манифест несёт не только имена и sha256, но и контракт детектора §7.1 целиком: раскладку
входа, единицы координат, формат бокса, список классов COCO. Kotlin сверяет каждое поле с
графом при загрузке, потому что два графа с одинаковыми формами тензоров расходятся
раскладкой и дают правдоподобные рамки не на машинах.

Хеши из манифеста вшиваются в `BuildConfig` приложения на этапе сборки и проверяются при
загрузке. Это ответ на вопрос «те ли модели попали в APK, что раскладывались», а не на
вопрос про tampering: эталон лежит в том же артефакте, что и проверяемый файл. Проверка
подписи APK — долг M-1.6.

### Сборка и тесты

JVM-тесты модуля (парсер индекса, геометрия, letterbox, разбор выхода детектора, манифест
— всё, что не требует телефона):

```bash
cd app/android && ./gradlew :vision:testDebugUnitTest
```

Инструментальные тесты — паритет с питоновским конвейером, жизненный цикл кадра, отказ по
подменённому хешу. Нужен телефон и разложенные модели:

```bash
cd app/android && ./gradlew :vision:connectedDebugAndroidTest
```

Эталон паритета для них собирает питоновская сторона в `tests/fixtures/device/`
(двенадцать кадров, эмбеддинги, боксы и `golden.json`) — он коммитится, в отличие от
моделей:

```bash
uv run --group train python -m scripts.gen_device_golden --run $CARLENS_DATA_ROOT/runs/m1_2_train_<дата> --detector $CARLENS_DATA_ROOT/detector/detector_fp32.tflite
```

Демонстрационное приложение — рамка и подпись поверх превью:

```bash
cd app/android && ./gradlew :demo:installDebug
```

### Замер латентности

```bash
cd app/android && ./gradlew :vision:connectedDebugAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=com.carlens.vision.LatencyTest
```

Печатает p50/p95 по стадиям в logcat (`adb logcat -d -s System.out:I | grep ЗАМЕР`) для
делегата по устройству и для принудительного CPU (`AssetSpec(allowGpu = false)`).

Так снимать нельзя, если числа нужны для решений: APK androidTest отладочного варианта
собирается `android:debuggable=true`, и ART в таком процессе ограничивает JIT — стадии на
Kotlin выходят втрое-впятеро медленнее, чем на самом деле. Замер вехи снимается релизным
вариантом:

```bash
cd app/android && ./gradlew :vision:connectedReleaseAndroidTest -Pcarlens.testRelease -Pandroid.testInstrumentationRunnerArguments.class=com.carlens.vision.LatencyTest
```

Флаг `-Pcarlens.testRelease` переключает `testBuildType` модуля `:vision`; без него всё
идёт по-прежнему в отладочном варианте. Числа обоих прогонов на S23 Ultra и выводы —
[`bench/latency.md`](bench/latency.md).

**Зелёный CI не является свидетельством того, что путь на устройстве работает.** Джоба
`android` гоняет только JVM-тесты модуля `:vision`: инструментальные требуют телефона и
разложенных моделей, а эмулятор без GPU не отвечает ни на один вопрос вехи. Паритет с
питоновским конвейером проверяется только ручным прогоном
`./gradlew :vision:connectedDebugAndroidTest`.

### Версии Android-слоя

| Пакет | Версия | Почему зафиксирован |
|---|---|---|
| Gradle | 8.10.2 | Требование React Native Gradle Plugin 0.76.5 (шаблон RN 0.76.5 пинует ровно эту версию в обёртке). Было 8.11.1 в `:vision`/`:demo`; понижение до версии шаблона сделано превентивно, по правилу §3 спеки «при расхождении двигается наша сторона» (см. ruling к Task 2, `.superpowers/sdd/2026-08-20-m1-6-rn-bridge/progress.md`) — падения на 8.11.1 никто не наблюдал и не проверял |
| AGP | 8.6.0 | Требование React Native Gradle Plugin 0.76.5 — версию AGP для всего дерева задаёт сам RNGP. Было 8.7.3; понижение превентивное, той же причины и по тому же правилу §3, что и Gradle выше — конфликта на 8.7.3 никто не наблюдал и не проверял |
| Kotlin | 1.9.24 | Требование React Native Gradle Plugin 0.76.5 — RNGP несёт композитную сборку именно под эту версию. Было 2.0.21; понижение превентивное, по тому же правилу §3 — совместимость RN-моста с Kotlin 2.0.21 никто не проверял, вопрос «можно ли вернуть 2.0.21» остаётся открытым |
| compileSdk / targetSdk | 35 | Диапазон YUV на API 34+ определён явно (`Image.getFormat` и `DYNAMIC_RANGE`), ниже принимается допущением |
| minSdk | 26 | Нижняя граница §5 спецификации; `ImageWriter` и `YUV_420_888` со стабильными шагами доступны с 26 |
| litert / litert-gpu / litert-gpu-api | 1.4.2 | В Google Maven `litert-gpu` заканчивается на 1.4.2: GPU-делегат из линии 2.x вынесен в другие координаты. Ветка берётся целиком — смешивать ядро 2.x с делегатом 1.x нельзя |
| CameraX | 1.4.1 | Пара к compileSdk 35; `ImageAnalysis` отдаёт `YUV_420_888` без промежуточного битмапа |
| kotlinx-serialization-json | 1.6.3 | Сдвинута вниз вместе с Kotlin 1.9.24 — это единственная строка таблицы с проверенной, а не превентивной причиной: версия 1.7.3 требует Kotlin >= 2.0.0-RC1 и **не компилируется** под 1.9.24 (наблюдалось напрямую); 1.6.3 — последний релиз ветки 1.6.x, совместимой с Kotlin 1.9.x |
| JDK | 17 | Требование AGP 8.6; `jvmTarget` модулей тот же |
| react-native | 0.76.5 | Версия каркаса RN-моста (M-1.6); версия шаблона, из которого унаследованы версии Gradle/AGP/Kotlin выше |
| react-native-vision-camera | 4.6.4 | Контракт пропа `outputOrientation` и типа `Frame`/`SharedArray` этой линии API зафиксирован в заметках спайка задачи 1; в новых линиях (5.x/6.x) контракт другой |
| react-native-worklets-core | 1.5.0 | Требование react-native-vision-camera 4 для работы Frame Processors (`Worklets.createRunOnJS`) |

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
