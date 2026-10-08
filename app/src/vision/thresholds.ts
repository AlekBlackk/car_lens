/**
 * Пороги отказа (§13.5) и таблица калибровки уверенности (§13.6).
 *
 * Оба файла лежат в `app/android/models/` — том же каталоге, куда `scripts/stage_assets.py`
 * раскладывает модели и каталог, и откуда gradle кладёт их в assets `:app`. Второго
 * каталога ассетов в проекте нет намеренно: два источника правды о том, какие пороги
 * уехали в APK, рано или поздно разойдутся.
 *
 * Разбор идёт при импорте модуля, а не при первом использовании: битый ассет обязан
 * уронить запуск с внятным текстом, а не выдать `NaN%` на живой машине.
 */
import сыраяКалибровка from '../../android/models/calibration.json'
import сырыеПороги from '../../android/models/thresholds.json'

export class AssetError extends Error {
  constructor(сообщение: string) {
    super(сообщение)
    this.name = 'AssetError'
  }
}

export type Пороги = {
  simMin: number
  marginMin: number
  minObservations: number
  minShortSidePx: number
  tuned: boolean
  /** Порог отказного прототипа §3 спеки 0.8.2; `null` — прототипов в ассетах нет. */
  rejectMargin: number | null
}

/**
 * `knots` — пары `[sim, вероятность]`, строго возрастающие по обеим координатам.
 *
 * Имя поля ASCII, и это требование границы, а не стиль: константа `CALIBRATION`
 * захватывается воркетом, а ключи объектов с не-ASCII символами до worklet-рантайма не
 * доезжают. Кириллическое `узлы` приезжало `undefined`, и `confidenceFromSim` падал на
 * устройстве с «Cannot convert undefined value to object» — при зелёных тестах в Node.
 */
export type Калибровка = {
  knots: number[][]
  clipLo: number
  clipHi: number
  tuned: boolean
}

const конечное = (значение: unknown, поле: string): number => {
  if (typeof значение !== 'number' || !Number.isFinite(значение)) {
    throw new AssetError(`${поле}: ожидалось конечное число, пришло ${JSON.stringify(значение)}`)
  }
  return значение
}

const вЕдиничномОтрезке = (значение: number, поле: string): number => {
  if (значение < 0 || значение > 1) {
    throw new AssetError(`${поле}: ${значение} вне [0, 1]`)
  }
  return значение
}

export function parseThresholds(документ: unknown): Пороги {
  const поля = документ as Record<string, unknown>
  const наблюдений = конечное(поля.min_observations, 'min_observations')
  if (!Number.isInteger(наблюдений) || наблюдений < 1) {
    throw new AssetError(`min_observations: ожидалось целое ≥ 1, пришло ${наблюдений}`)
  }
  const короткаяСторона = конечное(поля.min_short_side_px, 'min_short_side_px')
  if (короткаяСторона < 0) {
    throw new AssetError(`min_short_side_px: ${короткаяСторона} меньше нуля`)
  }
  // Поле необязательное: у сборок без отказных прототипов его нет, и решение тогда
  // идёт по прежним веткам. Порог в [-1, 1]: это разность двух косинусов.
  let rejectMargin: number | null = null
  if (поля.reject_margin !== undefined) {
    rejectMargin = конечное(поля.reject_margin, 'reject_margin')
    if (rejectMargin < -1 || rejectMargin > 1) {
      throw new AssetError(`reject_margin: ${rejectMargin} вне [-1, 1]`)
    }
  }
  return {
    simMin: вЕдиничномОтрезке(конечное(поля.sim_min, 'sim_min'), 'sim_min'),
    marginMin: вЕдиничномОтрезке(конечное(поля.margin_min, 'margin_min'), 'margin_min'),
    minObservations: наблюдений,
    minShortSidePx: короткаяСторона,
    tuned: поля.tuned === true,
    rejectMargin,
  }
}

export function parseCalibration(документ: unknown): Калибровка {
  const поля = документ as Record<string, unknown>
  const сырые = поля.knots
  if (!Array.isArray(сырые) || сырые.length < 2) {
    const сколько = Array.isArray(сырые) ? String(сырые.length) : 'не массив'
    throw new AssetError(`knots: нужно не меньше двух точек, пришло ${сколько}`)
  }
  const узлы: number[][] = new Array(сырые.length)
  for (let номер = 0; номер < сырые.length; номер++) {
    const пара = сырые[номер]
    if (!Array.isArray(пара) || пара.length !== 2) {
      throw new AssetError(`knots[${номер}]: ожидалась пара чисел`)
    }
    узлы[номер] = [конечное(пара[0], `knots[${номер}][0]`), конечное(пара[1], `knots[${номер}][1]`)]
  }
  for (let номер = 1; номер < узлы.length; номер++) {
    const текущий = узлы[номер]!
    const прошлый = узлы[номер - 1]!
    if (текущий[0]! <= прошлый[0]! || текущий[1]! <= прошлый[1]!) {
      throw new AssetError(`knots[${номер}]: таблица не строго возрастает по обеим координатам`)
    }
  }
  const clip = поля.clip
  if (!Array.isArray(clip) || clip.length !== 2) {
    throw new AssetError('clip: ожидалась пара чисел')
  }
  const clipLo = вЕдиничномОтрезке(конечное(clip[0], 'clip[0]'), 'clip[0]')
  const clipHi = вЕдиничномОтрезке(конечное(clip[1], 'clip[1]'), 'clip[1]')
  if (clipLo > clipHi) {
    throw new AssetError(`clip: нижняя граница ${clipLo} больше верхней ${clipHi}`)
  }
  return { knots: узлы, clipLo, clipHi, tuned: поля.tuned === true }
}

/**
 * Косинусное сходство в калиброванную вероятность. За пределами таблицы — клиппинг по
 * `clip`: §13.6 запрещает показывать «100%», потому что приложение, утверждающее полную
 * уверенность в fine-grained задаче, врёт пользователю.
 *
 * Края таблицы обрабатываются несимметрично, и это не описка. Слева — пол `clipLo`:
 * первый узел стоит на пороге отказа `sim_min`, ниже него изотоническая регрессия не
 * измеряла ничего, и выдавать вероятность первого узла сходству вдвое меньшему значило бы
 * экстраполировать вверх. Справа — значение последнего узла, зажатое `clipHi`: там кривая
 * измерена и уже вышла на насыщение, а `clipHi` мешает ей дойти до единицы.
 */
export function confidenceFromSim(sim: number, калибровка: Калибровка): number {
  'worklet'
  const узлы = калибровка.knots
  const первый = узлы[0]!
  const последний = узлы[узлы.length - 1]!
  let значение: number
  if (sim <= первый[0]!) {
    значение = калибровка.clipLo
  } else if (sim >= последний[0]!) {
    значение = последний[1]!
  } else {
    let номер = 1
    while (узлы[номер]![0]! < sim) {
      номер++
    }
    const левый = узлы[номер - 1]!
    const правый = узлы[номер]!
    const доля = (sim - левый[0]!) / (правый[0]! - левый[0]!)
    значение = левый[1]! + доля * (правый[1]! - левый[1]!)
  }
  if (значение < калибровка.clipLo) return калибровка.clipLo
  if (значение > калибровка.clipHi) return калибровка.clipHi
  return значение
}

export const THRESHOLDS: Пороги = parseThresholds(сырыеПороги)
export const CALIBRATION: Калибровка = parseCalibration(сыраяКалибровка)
