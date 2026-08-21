import { describe, expect, test, vi } from 'vitest'
import { РАЗМЕР_ВЫБОРКИ, StabilityCollector } from './stability'

const событие = (id: number, мгновенный: number, ema: number, ключ: string) => ({
  id,
  top1Мгновенный: мгновенный,
  top1Ema: ema,
  ключРешения: ключ,
})

describe('StabilityCollector', () => {
  test('первое наблюдение трека сменой не считается', () => {
    const накопитель = new StabilityCollector()
    накопитель.добавить([событие(1, 5, 5, 'модель:5')])
    expect(накопитель.сводка().top1EmaСмен).toBe(0)
    expect(накопитель.сводка().наблюдений).toBe(1)
  })

  test('смена класса считается по каждому ряду отдельно', () => {
    const накопитель = new StabilityCollector()
    накопитель.добавить([событие(1, 5, 5, 'модель:5')])
    накопитель.добавить([событие(1, 6, 5, 'модель:5')])
    expect(накопитель.сводка().top1МгновенныйСмен).toBe(1)
    expect(накопитель.сводка().top1EmaСмен).toBe(0)
  })

  test('decision_churn расходится с top1_churn', () => {
    const накопитель = new StabilityCollector()
    накопитель.добавить([событие(1, 5, 5, 'бренд:5')])
    накопитель.добавить([событие(1, 6, 6, 'бренд:5')])
    expect(накопитель.сводка().top1EmaСмен).toBe(1)
    expect(накопитель.сводка().решенийСмен).toBe(0)
  })

  test('смерть трека не даёт ложной смены', () => {
    const накопитель = new StabilityCollector()
    накопитель.добавить([событие(1, 5, 5, 'модель:5')])
    накопитель.добавить([событие(2, 9, 9, 'модель:9')])
    expect(накопитель.сводка().top1EmaСмен).toBe(0)
    expect(накопитель.сводка().треков).toBe(2)
  })

  test('считает число разных классов и видов решений', () => {
    const накопитель = new StabilityCollector()
    накопитель.добавить([событие(1, 5, 5, 'модель:5')])
    накопитель.добавить([событие(1, 6, 7, 'неизвестно')])
    expect(накопитель.сводка().классовEma).toBe(2)
    expect(накопитель.сводка().видов).toBe(2)
  })

  test('печатает один раз по набору выборки', () => {
    const печать = vi.spyOn(console, 'log').mockImplementation(() => undefined)
    const накопитель = new StabilityCollector()
    for (let номер = 0; номер < РАЗМЕР_ВЫБОРКИ + 10; номер++) {
      накопитель.добавить([событие(1, номер % 3, номер % 2, 'модель:1')])
    }
    const наши = печать.mock.calls.filter((вызов) => String(вызов[0]).startsWith('СТАБИЛЬНОСТЬ'))
    expect(наши).toHaveLength(2)
    печать.mockRestore()
  })

  test('после печати накопитель перестаёт расти', () => {
    const печать = vi.spyOn(console, 'log').mockImplementation(() => undefined)
    const накопитель = new StabilityCollector()
    for (let номер = 0; номер < РАЗМЕР_ВЫБОРКИ + 50; номер++) {
      накопитель.добавить([событие(1, 1, 1, 'модель:1')])
    }
    expect(накопитель.сводка().наблюдений).toBe(РАЗМЕР_ВЫБОРКИ)
    печать.mockRestore()
  })
})
