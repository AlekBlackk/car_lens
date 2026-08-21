import { describe, expect, test } from 'vitest'
import { classLabel, parseCatalog, sameBrand } from './labels'

const каталог = {
  format_version: 1,
  classes: [
    { model_id: 'audi__a4', brand: 'audi', display: 'Audi A4', platform_group: null },
    { model_id: 'audi__a6', brand: 'audi', display: 'Audi A6', platform_group: null },
    {
      model_id: 'volkswagen__golf',
      brand: 'volkswagen',
      display: 'VW Golf',
      platform_group: null,
    },
  ],
}

describe('parseCatalog', () => {
  test('разбирает классы в порядке индексов', () => {
    const классы = parseCatalog(каталог)
    expect(классы[0]!.display).toBe('Audi A4')
    expect(классы[2]!.brand).toBe('volkswagen')
    expect(классы[0]!.platformGroup).toBeNull()
  })

  test('пустой каталог роняет разбор', () => {
    expect(() => parseCatalog({ classes: [] })).toThrow(/classes/)
  })

  test('класс без display роняет разбор с номером', () => {
    const битый = { classes: [{ model_id: 'x', brand: 'y', platform_group: null }] }
    expect(() => parseCatalog(битый)).toThrow(/classes\[0\]/)
  })
})

describe('classLabel', () => {
  test('индекс за границей каталога даёт null, а не падение', () => {
    expect(classLabel(-1)).toBeNull()
    expect(classLabel(1_000_000)).toBeNull()
  })

  test('первый класс настоящего каталога читается', () => {
    expect(classLabel(0)?.display.length).toBeGreaterThan(0)
  })
})

describe('sameBrand', () => {
  test('индекс за границей не совпадает ни с чем', () => {
    expect(sameBrand(0, 1_000_000)).toBe(false)
  })

  test('класс совпадает сам с собой', () => {
    expect(sameBrand(0, 0)).toBe(true)
  })
})
