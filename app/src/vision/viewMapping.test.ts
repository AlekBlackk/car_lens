import { describe, expect, test } from 'vitest'
import { mapBoxToView } from './viewMapping'

describe('mapBoxToView', () => {
  test('кадр и вью одного соотношения: масштаб без сдвигов', () => {
    const прямоугольник = mapBoxToView(
      { x: 100, y: 200, w: 50, h: 80, conf: 0.9 },
      1000, 2000, 500, 1000,
    )
    expect(прямоугольник).toEqual({ left: 50, top: 100, width: 25, height: 40 })
  })

  test('вью шире кадра: масштаб по ширине, кадр обрезается сверху и снизу', () => {
    // FILL_CENTER: масштаб по большей стороне, иначе рамка растянется относительно
    // картинки, которую превью показывает без искажения пропорций.
    const прямоугольник = mapBoxToView(
      { x: 0, y: 0, w: 100, h: 100, conf: 1 },
      100, 200, 100, 100,
    )
    expect(прямоугольник.left).toBeCloseTo(0)
    expect(прямоугольник.top).toBeCloseTo(-50)
    expect(прямоугольник.width).toBeCloseTo(100)
    expect(прямоугольник.height).toBeCloseTo(100)
  })

  test('нулевая сторона кадра роняет с обоими числами', () => {
    expect(() => mapBoxToView({ x: 0, y: 0, w: 1, h: 1, conf: 1 }, 0, 200, 100, 100))
      .toThrow(/0.*200/)
  })
})
