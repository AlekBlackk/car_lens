import { describe, expect, test, vi } from 'vitest'
import { percentile, НакопительЦеныМоста } from './latency'

describe('percentile', () => {
  test('ближайший ранг, как в LatencyTest на Kotlin', () => {
    // Тот же метод, что в M-1.5: без него числа двух вех несравнимы.
    expect(percentile([1, 2, 3, 4, 5], 50)).toBe(3)
    expect(percentile([1, 2, 3, 4, 5], 95)).toBe(5)
  })

  test('пустой набор роняет, а не возвращает ноль', () => {
    expect(() => percentile([], 50)).toThrow(/пуст/)
  })
})

describe('НакопительЦеныМоста', () => {
  test('до 200 замеров стадии ничего не печатает', () => {
    const лог = vi.spyOn(console, 'log').mockImplementation(() => {})
    const накопитель = new НакопительЦеныМоста()
    for (let номер = 0; номер < 199; номер++) {
      накопитель.добавить('embedAndMatch', 0, 5, 1)
    }
    expect(лог).not.toHaveBeenCalled()
    лог.mockRestore()
  })

  test('на 200-м замере стадии печатает p50/p95 ровно один раз', () => {
    const лог = vi.spyOn(console, 'log').mockImplementation(() => {})
    const накопитель = new НакопительЦеныМоста()
    // Цена каждого замера одинаковая (10 − 0 − 3 = 7): p50 и p95 совпадают, легко сверить.
    for (let номер = 0; номер < 200; номер++) {
      накопитель.добавить('detect', 0, 10, 3)
    }
    expect(лог).toHaveBeenCalledTimes(1)
    expect(лог.mock.calls[0]?.[0]).toMatch(/ЦЕНА_МОСТА detect p50=7(\.0+)? p95=7(\.0+)?/)
    лог.mockRestore()
  })

  test('стадии копятся и печатаются независимо друг от друга', () => {
    const лог = vi.spyOn(console, 'log').mockImplementation(() => {})
    const накопитель = new НакопительЦеныМоста()
    for (let номер = 0; номер < 200; номер++) {
      накопитель.добавить('detect', 0, 10, 3)
    }
    лог.mockClear()
    for (let номер = 0; номер < 199; номер++) {
      накопитель.добавить('matchEmbedding', 0, 2, 1)
    }
    expect(лог).not.toHaveBeenCalled()
    накопитель.добавить('matchEmbedding', 0, 2, 1)
    expect(лог).toHaveBeenCalledTimes(1)
    expect(лог.mock.calls[0]?.[0]).toContain('ЦЕНА_МОСТА matchEmbedding')
    лог.mockRestore()
  })
})
