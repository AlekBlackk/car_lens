import { describe, expect, test } from 'vitest'
import { DimensionError, EMA_ALPHA, emaStep } from './ema'

const вектор = (числа: number[]) => Float32Array.from(числа)
const норма = (числа: number[]) => Math.sqrt(числа.reduce((сумма, ч) => сумма + ч * ч, 0))
const косинус = (а: number[], б: number[]) =>
  а.reduce((сумма, ч, номер) => сумма + ч * б[номер]!, 0) / (норма(а) * норма(б))

describe('emaStep', () => {
  test('инициализация даёт нормализованную копию входа', () => {
    const итог = emaStep(null, вектор([3, 4]), EMA_ALPHA)
    expect(итог.принят).toBe(true)
    expect(итог.вектор![0]).toBeCloseTo(0.6, 6)
    expect(итог.вектор![1]).toBeCloseTo(0.8, 6)
  })

  test('норма результата равна единице после каждого шага', () => {
    let текущий = emaStep(null, вектор([1, 0, 0]), EMA_ALPHA).вектор
    const входы = [вектор([0, 1, 0]), вектор([0, 0, 1]), вектор([1, 1, 0])]
    for (const вход of входы) {
      текущий = emaStep(текущий, вход, EMA_ALPHA).вектор
      expect(норма(текущий!)).toBeCloseTo(1, 6)
    }
  })

  test('повторяющийся одинаковый эмбеддинг оставляет EMA на месте', () => {
    const первый = emaStep(null, вектор([1, 2, 3]), EMA_ALPHA).вектор
    const второй = emaStep(первый, вектор([1, 2, 3]), EMA_ALPHA).вектор
    expect(косинус(первый!, второй!)).toBeCloseTo(1, 6)
  })

  test('шаг двигает EMA в сторону нового вектора', () => {
    const старый = emaStep(null, вектор([1, 0]), EMA_ALPHA).вектор!
    const новый = emaStep(старый, вектор([0, 1]), EMA_ALPHA).вектор!
    expect(косинус(новый, [0, 1])).toBeGreaterThan(косинус(старый, [0, 1]))
  })

  test('чередование двух векторов не разбегается', () => {
    let текущий = emaStep(null, вектор([1, 0]), EMA_ALPHA).вектор
    for (let шаг = 0; шаг < 20; шаг++) {
      текущий = emaStep(текущий, шаг % 2 === 0 ? вектор([0, 1]) : вектор([1, 0]), EMA_ALPHA).вектор
    }
    expect(норма(текущий!)).toBeCloseTo(1, 6)
    expect(косинус(текущий!, [1, 0])).toBeGreaterThan(0)
    expect(косинус(текущий!, [0, 1])).toBeGreaterThan(0)
  })

  test('NaN отвергает шаг и возвращает предыдущее значение', () => {
    const прежний = emaStep(null, вектор([1, 0]), EMA_ALPHA).вектор!
    const итог = emaStep(прежний, вектор([NaN, 0]), EMA_ALPHA)
    expect(итог.принят).toBe(false)
    expect(итог.вектор).toBe(прежний)
  })

  test('Infinity отвергает шаг', () => {
    const прежний = emaStep(null, вектор([1, 0]), EMA_ALPHA).вектор!
    expect(emaStep(прежний, вектор([Infinity, 0]), EMA_ALPHA).принят).toBe(false)
  })

  test('нулевая норма отвергает шаг', () => {
    const прежний = emaStep(null, вектор([1, 0]), EMA_ALPHA).вектор!
    expect(emaStep(прежний, вектор([0, 0]), EMA_ALPHA).принят).toBe(false)
  })

  test('первый же битый эмбеддинг оставляет трек без EMA', () => {
    const итог = emaStep(null, вектор([NaN, NaN]), EMA_ALPHA)
    expect(итог.принят).toBe(false)
    expect(итог.вектор).toBeNull()
  })

  test('несовпадение размерности роняет с внятной ошибкой', () => {
    const прежний = emaStep(null, вектор([1, 0, 0]), EMA_ALPHA).вектор!
    expect(() => emaStep(прежний, вектор([1, 0]), EMA_ALPHA)).toThrow(DimensionError)
  })

  test('почти противоположный вектор при вырожденной сумме берёт свежий', () => {
    const прежний = [1, 0]
    const итог = emaStep(прежний, вектор([-1, 0]), 1)
    expect(итог.принят).toBe(true)
    expect(итог.вектор![0]).toBeCloseTo(-1, 6)
  })
})
