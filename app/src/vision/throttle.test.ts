import { describe, expect, test } from 'vitest'
import { CLASSIFY_INTERVAL_MS, MAX_CROPS, shouldClassify } from './throttle'

const бокс = (conf: number) => ({ x: 0, y: 0, w: 10, h: 10, conf })

describe('shouldClassify', () => {
  test('до истечения интервала не классифицирует', () => {
    expect(shouldClassify(1000, 1000 - CLASSIFY_INTERVAL_MS + 1, [бокс(0.9)])).toEqual([])
  })

  test('после интервала берёт не больше MAX_CROPS по убыванию conf', () => {
    const выбранные = shouldClassify(
      1000, 1000 - CLASSIFY_INTERVAL_MS, [бокс(0.5), бокс(0.9), бокс(0.7)],
    )
    expect(выбранные).toHaveLength(MAX_CROPS)
    expect(выбранные.map((б) => б.conf)).toEqual([0.9, 0.7])
  })

  test('пустой список боксов не классифицирует', () => {
    expect(shouldClassify(1000, 0, [])).toEqual([])
  })
})
