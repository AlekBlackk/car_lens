<h1 align="center">CarLens</h1>

<p align="center">
  Распознавание марки и модели автомобиля с камеры телефона — полностью на устройстве, без сети.
</p>

<p align="center">
  <img alt="Android" src="https://img.shields.io/badge/Android-8.0%2B-3DDC84?logo=android&logoColor=white">
  <img alt="React Native" src="https://img.shields.io/badge/React%20Native-0.76-61DAFB?logo=react&logoColor=black">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
  <img alt="On-device" src="https://img.shields.io/badge/inference-on--device-success">
  <img alt="Status" src="https://img.shields.io/badge/version-0.9.0-blue">
</p>

Наведите камеру на машину — приложение найдёт её в кадре и назовёт марку и модель.
Каталог ориентирован на российский автопарк: **260 классов, 55 марок** — от «копейки» и
Granta до китайских новинок и массовых иномарок. Интернет не нужен, кадры с телефона
никуда не отправляются.

## Как это работает

```
кадр с камеры ──► детектор машин ──► кроп ──► эмбеддинг ──► сравнение с каталогом ──► марка и модель
 (YUV_420_888)     YOLO11n          (letterbox)  MobileNetV4     косинус к центроидам      top-k + отказ
```

- **Детектор** (YOLO11n) находит машину и режет кроп прямо из плоскостей `YUV_420_888`.
- **Классификатор** (MobileNetV4, дообучен на российском пуле) превращает кроп в эмбеддинг.
  Модель не «выбирает из списка», а измеряет расстояние до эталонов каталога (индекс
  центроидов по классам) — это позволяет отказывать на незнакомых машинах.
- **Отказ вместо угадывания.** Если машина не похожа ни на один класс (чужая модель,
  не машина, плохой ракурс), приложение не называет модель наугад.
- **Стабилизация по кадрам.** Решение копится по нескольким кадрам, а не по одному.
- Инференс идёт через LiteRT на GPU телефона, на CPU — как запасной путь.

## Что внутри

| Каталог | Назначение |
|---|---|
| `app/` | Приложение: React Native (интерфейс, камера) + нативный Android-модуль `:vision` на Kotlin |
| `pipeline/` | Конвейер данных на Python: сбор кадров, дедупликация, редакция номеров и лиц, сплиты, индекс |
| `training/` | Обучение и сборка индекса эталонов |
| `export/` | Конвертация в LiteRT и проверка паритета с PyTorch |
| `catalog/` | Каталог марок и моделей, алиасы названий |
| `scripts/` | Запускаемые шаги конвейера |
| `tests/` | Тесты Python-части |
| `spike/` | Ранние эксперименты (CLIP zero-shot) |

## Быстрый старт

Нужны [uv](https://docs.astral.sh/uv/), Python 3.11, Node 20 и JDK 17 с Android SDK 35.

**Python-часть и проверки**

```bash
uv sync --locked --group dev
uv run ruff check . && uv run ruff format --check . && uv run mypy
uv run pytest -m "not export"
```

**Приложение**

```bash
cd app
npm ci
npm run typecheck && npm run lint && npm test
```

**Android-модуль**

```bash
cd app/android
./gradlew :vision:testDebugUnitTest :app:compileDebugKotlin
```

Эти же проверки гоняет CI (`.github/workflows/ci.yml`).

## Обучение и экспорт

Обучение, индекс и экспорт требуют GPU и данных, которых в репозитории нет (см. ниже).
Каталог данных задаётся переменной `CARLENS_DATA_ROOT` (по умолчанию `~/carlens-data`).

```bash
uv run --group train python -m training.train                 # обучение
uv run --group train python -m training.build_index --checkpoint <best.pt>   # индекс эталонов
uv run --group export python -m export.to_tflite --checkpoint <best.pt>      # LiteRT
uv run --group export python -m export.verify_parity ...      # паритет с PyTorch
```

Конвейер рассчитан на Linux (WSL2 Ubuntu 24.04 тоже подходит). Группы зависимостей `train`,
`export` и `yolo` ставятся в отдельные окружения: у них конфликтующие тулчейны.

## Данные, приватность и лицензии

- **В репозитории нет** ни датасетов, ни весов моделей, ни собранных APK — только код,
  каталог и тесты.
- Кадры собираются из открытых источников с учётом лицензий (Wikimedia Commons, список
  ссылок MAD-Cars — CC BY-NC-SA, только некоммерческое использование). На этапе сбора
  **номера и лица на кадрах закрываются** до того, как кадр попадает в обучающий пул.
- Сторонние компоненты и их лицензии перечислены в [`NOTICE.md`](NOTICE.md). Обратите
  внимание: `ultralytics` / YOLO11n распространяются под **AGPL-3.0**, а Stanford Cars и
  CompCars — только для исследований.
- Проект личный и учебный. Лицензия на собственный код не выбрана: по умолчанию все права
  защищены. Если хотите использовать код — откройте issue.
