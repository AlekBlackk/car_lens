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
    expect(decide(вход({ наблюдений: 2 }), пороги, калибровка).вид).toBe('определяю')
  })

  test('ровно на min_observations уже решает', () => {
    expect(decide(вход({ наблюдений: 3 }), пороги, калибровка).вид).toBe('модель')
  })

  test('мелкий бокс — подойдите ближе', () => {
    expect(decide(вход({ короткаяСторона: 119 }), пороги, калибровка).вид).toBe('ближе')
  })

  test('ровно на min_short_side_px уже решает', () => {
    expect(decide(вход({ короткаяСторона: 120 }), пороги, калибровка).вид).toBe('модель')
  })

  test('sim ниже порога — неизвестно', () => {
    const итог = decide(вход({ топ1: { classIdx: 1, sim: 0.419 }, топ2: null }), пороги, калибровка)
    expect(итог.вид).toBe('неизвестно')
  })

  test('sim ровно на пороге проходит', () => {
    const итог = decide(вход({ топ1: { classIdx: 1, sim: 0.42 }, топ2: null }), пороги, калибровка)
    expect(итог.вид).toBe('модель')
  })

  test('отсутствие топ-1 — неизвестно', () => {
    expect(decide(вход({ топ1: null }), пороги, калибровка).вид).toBe('неизвестно')
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
    expect(итог).toEqual({ вид: 'бренд', classIdx: 1 })
  })

  test('малый margin при разных брендах — неизвестно', () => {
    const итог = decide(
      вход({ топ1: { classIdx: 1, sim: 0.9 }, топ2: { classIdx: 2, sim: 0.87 } }),
      пороги,
      калибровка,
    )
    expect(итог.вид).toBe('неизвестно')
  })

  test('margin ровно на пороге проходит как модель', () => {
    const итог = decide(
      вход({ топ1: { classIdx: 1, sim: 0.9 }, топ2: { classIdx: 2, sim: 0.85 } }),
      пороги,
      калибровка,
    )
    expect(итог.вид).toBe('модель')
  })

  test('решение модели несёт калиброванную уверенность и флаг', () => {
    const итог = decide(вход(), пороги, калибровка)
    expect(итог).toMatchObject({ вид: 'модель', classIdx: 1, откалибровано: false })
    expect(итог.вид === 'модель' && итог.уверенность).toBeLessThan(0.98)
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
    expect(полуправда).toMatchObject({ откалибровано: false })
  })
})

describe('decisionKey', () => {
  test('разные виды дают разные ключи', () => {
    expect(decisionKey({ вид: 'определяю' })).not.toBe(decisionKey({ вид: 'неизвестно' }))
  })

  test('разные классы одной модели дают разные ключи', () => {
    const первый = decisionKey({
      вид: 'модель',
      classIdx: 1,
      уверенность: 0.5,
      откалибровано: false,
    })
    const второй = decisionKey({
      вид: 'модель',
      classIdx: 2,
      уверенность: 0.5,
      откалибровано: false,
    })
    expect(первый).not.toBe(второй)
  })

  test('уверенность на ключ не влияет', () => {
    const первый = decisionKey({
      вид: 'модель',
      classIdx: 1,
      уверенность: 0.5,
      откалибровано: false,
    })
    const второй = decisionKey({
      вид: 'модель',
      classIdx: 1,
      уверенность: 0.9,
      откалибровано: false,
    })
    expect(первый).toBe(второй)
  })
})
