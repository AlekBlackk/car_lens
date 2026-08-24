import { describe, expect, test } from 'vitest'
import { decide, decisionKey } from './decide'
import { parseCalibration, parseThresholds } from './thresholds'

const пороги = parseThresholds({
  version: 1,
  tuned: false,
  sim_min: 0.42,
  margin_min: 0.05,
  min_observations: 3,
  min_short_side_px: 120,
})

const калибровка = parseCalibration({
  version: 1,
  method: 'isotonic_piecewise_linear',
  tuned: false,
  knots: [
    [0.4, 0.05],
    [1.0, 0.98],
  ],
  clip: [0.01, 0.98],
})

const вход = (правки: Partial<Parameters<typeof decide>[0]> = {}) => ({
  наблюдений: 5,
  короткаяСторона: 300,
  топ1: { classIdx: 1, sim: 0.9 },
  топ2: { classIdx: 2, sim: 0.5 },
  брендыСовпали: false,
  ...правки,
})

describe('decide', () => {
  test('до min_observations — определяю', () => {
    expect(decide(вход({ наблюдений: 2 }), пороги, калибровка).kind).toBe('определяю')
  })

  test('ровно на min_observations уже решает', () => {
    expect(decide(вход({ наблюдений: 3 }), пороги, калибровка).kind).toBe('модель')
  })

  test('мелкий бокс — подойдите ближе', () => {
    expect(decide(вход({ короткаяСторона: 119 }), пороги, калибровка).kind).toBe('ближе')
  })

  test('ровно на min_short_side_px уже решает', () => {
    expect(decide(вход({ короткаяСторона: 120 }), пороги, калибровка).kind).toBe('модель')
  })

  test('sim ниже порога — неизвестно', () => {
    const итог = decide(вход({ топ1: { classIdx: 1, sim: 0.419 }, топ2: null }), пороги, калибровка)
    expect(итог.kind).toBe('неизвестно')
  })

  test('sim ровно на пороге проходит', () => {
    const итог = decide(вход({ топ1: { classIdx: 1, sim: 0.42 }, топ2: null }), пороги, калибровка)
    expect(итог.kind).toBe('модель')
  })

  test('отсутствие топ-1 — неизвестно', () => {
    expect(decide(вход({ топ1: null }), пороги, калибровка).kind).toBe('неизвестно')
  })

  test('малый margin при совпавших брендах — только бренд', () => {
    const итог = decide(
      вход({
        топ1: { classIdx: 1, sim: 0.9 },
        топ2: { classIdx: 2, sim: 0.87 },
        брендыСовпали: true,
      }),
      пороги,
      калибровка,
    )
    expect(итог).toEqual({ kind: 'бренд', classIdx: 1 })
  })

  test('малый margin при разных брендах — неизвестно', () => {
    const итог = decide(
      вход({ топ1: { classIdx: 1, sim: 0.9 }, топ2: { classIdx: 2, sim: 0.87 } }),
      пороги,
      калибровка,
    )
    expect(итог.kind).toBe('неизвестно')
  })

  test('margin ровно на пороге проходит как модель', () => {
    const итог = decide(
      вход({ топ1: { classIdx: 1, sim: 0.9 }, топ2: { classIdx: 2, sim: 0.85 } }),
      пороги,
      калибровка,
    )
    expect(итог.kind).toBe('модель')
  })

  test('решение модели несёт калиброванную уверенность и флаг', () => {
    const итог = decide(вход(), пороги, калибровка)
    expect(итог).toMatchObject({ kind: 'модель', classIdx: 1, calibrated: false })
    expect(итог.kind === 'модель' && итог.confidence).toBeLessThan(0.98)
  })

  test('откалибровано только когда оба ассета tuned', () => {
    const настоящие = parseThresholds({
      version: 1,
      tuned: true,
      sim_min: 0.42,
      margin_min: 0.05,
      min_observations: 3,
      min_short_side_px: 120,
    })
    const полуправда = decide(вход(), настоящие, калибровка)
    expect(полуправда).toMatchObject({ calibrated: false })
  })
})

describe('decisionKey', () => {
  test('разные виды дают разные ключи', () => {
    expect(decisionKey({ kind: 'определяю' })).not.toBe(decisionKey({ kind: 'неизвестно' }))
  })

  test('разные классы одной модели дают разные ключи', () => {
    const первый = decisionKey({
      kind: 'модель',
      classIdx: 1,
      confidence: 0.5,
      calibrated: false,
    })
    const второй = decisionKey({
      kind: 'модель',
      classIdx: 2,
      confidence: 0.5,
      calibrated: false,
    })
    expect(первый).not.toBe(второй)
  })

  test('уверенность на ключ не влияет', () => {
    const первый = decisionKey({
      kind: 'модель',
      classIdx: 1,
      confidence: 0.5,
      calibrated: false,
    })
    const второй = decisionKey({
      kind: 'модель',
      classIdx: 1,
      confidence: 0.9,
      calibrated: false,
    })
    expect(первый).toBe(второй)
  })
})
