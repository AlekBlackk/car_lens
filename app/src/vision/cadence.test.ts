import { describe, expect, test } from 'vitest'
import { MAX_DUTY, MAX_INTERVAL_MS, nextPipelineInterval } from './cadence'

const БАЗА_МС = 100

describe('nextPipelineInterval', () => {
  test('дешёвый шаг оставляет базовый интервал', () => {
    expect(nextPipelineInterval(38, БАЗА_МС)).toBe(БАЗА_МС)
  })

  test('полный шаг здорового состояния (≈57 мс) интервал не растягивает', () => {
    expect(nextPipelineInterval(57, БАЗА_МС)).toBe(БАЗА_МС)
  })

  test('шаг ровно на границе нагрузки интервал не меняет', () => {
    expect(nextPipelineInterval(БАЗА_МС * MAX_DUTY, БАЗА_МС)).toBe(БАЗА_МС)
  })

  test('дорогой шаг растягивает интервал так, чтобы нагрузка не превышала MAX_DUTY', () => {
    for (const цена of [80, 150, 300, 500]) {
      const интервал = nextPipelineInterval(цена, БАЗА_МС)
      expect(цена / интервал).toBeLessThanOrEqual(MAX_DUTY + 1e-9)
    }
  })

  test('горячий корпус: 300 мс на шаг даёт полсекунды паузы, а не спину к спине', () => {
    expect(nextPipelineInterval(300, БАЗА_МС)).toBeCloseTo(500, 6)
  })

  test('аномально дорогой шаг упирается в потолок', () => {
    expect(nextPipelineInterval(5000, БАЗА_МС)).toBe(MAX_INTERVAL_MS)
  })

  test('интервал никогда не короче базового', () => {
    expect(nextPipelineInterval(0, БАЗА_МС)).toBe(БАЗА_МС)
  })
})
