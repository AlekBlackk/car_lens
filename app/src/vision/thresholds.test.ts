import { describe, expect, test } from 'vitest'
import {
  AssetError,
  CALIBRATION,
  THRESHOLDS,
  confidenceFromSim,
  parseCalibration,
  parseThresholds,
} from './thresholds'

const пороги = {
  version: 1,
  tuned: false,
  sim_min: 0.42,
  margin_min: 0.05,
  min_observations: 3,
  min_short_side_px: 120,
}

const калибровка = {
  version: 1,
  method: 'isotonic_piecewise_linear',
  tuned: false,
  knots: [
    [0.4, 0.02],
    [0.5, 0.3],
    [0.6, 0.9],
  ],
  clip: [0.01, 0.98],
}

describe('parseThresholds', () => {
  test('разбирает корректный документ', () => {
    expect(parseThresholds(пороги)).toEqual({
      simMin: 0.42,
      marginMin: 0.05,
      minObservations: 3,
      minShortSidePx: 120,
      tuned: false,
    })
  })

  test('NaN в поле роняет разбор с именем поля', () => {
    expect(() => parseThresholds({ ...пороги, sim_min: NaN })).toThrow(/sim_min/)
  })

  test('Infinity в поле роняет разбор', () => {
    expect(() => parseThresholds({ ...пороги, margin_min: Infinity })).toThrow(AssetError)
  })

  test('дробный min_observations роняет разбор', () => {
    expect(() => parseThresholds({ ...пороги, min_observations: 2.5 })).toThrow(/min_observations/)
  })

  test('нулевой min_observations роняет разбор', () => {
    expect(() => parseThresholds({ ...пороги, min_observations: 0 })).toThrow(/min_observations/)
  })

  test('sim_min вне [0, 1] роняет разбор', () => {
    expect(() => parseThresholds({ ...пороги, sim_min: 1.5 })).toThrow(/sim_min/)
  })
})

describe('parseCalibration', () => {
  test('разбирает корректный документ', () => {
    const итог = parseCalibration(калибровка)
    expect(итог.узлы).toHaveLength(3)
    expect(итог.clipLo).toBe(0.01)
    expect(итог.clipHi).toBe(0.98)
    expect(итог.tuned).toBe(false)
  })

  test('одна точка в knots роняет разбор', () => {
    expect(() => parseCalibration({ ...калибровка, knots: [[0.4, 0.02]] })).toThrow(/knots/)
  })

  test('немонотонные knots роняют разбор', () => {
    const кривые = [
      [0.4, 0.02],
      [0.5, 0.3],
      [0.45, 0.9],
    ]
    expect(() => parseCalibration({ ...калибровка, knots: кривые })).toThrow(/knots/)
  })

  test('немонотонные по вероятности knots роняют разбор', () => {
    const кривые = [
      [0.4, 0.3],
      [0.5, 0.2],
    ]
    expect(() => parseCalibration({ ...калибровка, knots: кривые })).toThrow(/knots/)
  })

  test('clipLo больше clipHi роняет разбор', () => {
    expect(() => parseCalibration({ ...калибровка, clip: [0.9, 0.1] })).toThrow(/clip/)
  })
})

describe('confidenceFromSim', () => {
  const таблица = parseCalibration(калибровка)

  test('точно в узле даёт значение узла', () => {
    expect(confidenceFromSim(0.5, таблица)).toBeCloseTo(0.3, 6)
  })

  test('между узлами интерполирует линейно', () => {
    expect(confidenceFromSim(0.45, таблица)).toBeCloseTo(0.16, 6)
  })

  test('ниже таблицы клиппится по clipLo', () => {
    expect(confidenceFromSim(0.1, таблица)).toBe(0.01)
  })

  test('выше таблицы клиппится по clipHi', () => {
    expect(confidenceFromSim(0.99, таблица)).toBeCloseTo(0.9, 6)
  })

  test('никогда не даёт единицу', () => {
    const крайняя = parseCalibration({
      ...калибровка,
      knots: [
        [0.4, 0.5],
        [0.6, 1.0],
      ],
    })
    expect(confidenceFromSim(1.0, крайняя)).toBe(0.98)
  })
})

describe('ассеты сборки', () => {
  test('настоящий thresholds.json разбирается', () => {
    expect(THRESHOLDS.minObservations).toBeGreaterThanOrEqual(1)
  })

  test('настоящий calibration.json разбирается', () => {
    expect(CALIBRATION.узлы.length).toBeGreaterThanOrEqual(2)
  })

  test('плейсхолдеры честно помечены неоткалиброванными', () => {
    expect(THRESHOLDS.tuned).toBe(false)
    expect(CALIBRATION.tuned).toBe(false)
  })
})
