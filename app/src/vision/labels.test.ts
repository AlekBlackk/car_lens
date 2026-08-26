import { describe, expect, test } from 'vitest'
import {
  allClasses,
  brandsMatch,
  catalogDigest,
  catalogInfo,
  classLabel,
  parseCatalog,
  parseCatalogInfo,
  sameBrand,
} from './labels'
import { AssetError } from './thresholds'

const каталог = {
  format_version: 1,
  classes: [
    { class_id: 'audi__a4', members: ['audi__a4'], brand: 'audi', display: 'Audi A4' },
    { class_id: 'audi__a6', members: ['audi__a6'], brand: 'audi', display: 'Audi A6' },
    {
      class_id: 'grp__golf',
      members: ['volkswagen__golf', 'skoda__octavia'],
      brand: null,
      display: 'VW Golf / Skoda Octavia',
    },
  ],
}

describe('parseCatalog', () => {
  test('разбирает классы в порядке индексов', () => {
    const классы = parseCatalog(каталог)
    expect(классы[0]!.classId).toBe('audi__a4')
    expect(классы[0]!.display).toBe('Audi A4')
    expect(классы[1]!.brand).toBe('audi')
    expect(классы[0]!.members).toEqual(['audi__a4'])
  })

  test('у разнобрендовой группы brand равен null, а участники на месте', () => {
    const группа = parseCatalog(каталог)[2]!
    expect(группа.brand).toBeNull()
    expect(группа.members).toEqual(['volkswagen__golf', 'skoda__octavia'])
    expect(группа.display).toBe('VW Golf / Skoda Octavia')
  })

  test('пустой каталог роняет разбор', () => {
    expect(() => parseCatalog({ classes: [] })).toThrow(/classes/)
  })

  test('класс без display роняет разбор с номером', () => {
    const битый = { classes: [{ class_id: 'x', members: ['x'], brand: 'y' }] }
    expect(() => parseCatalog(битый)).toThrow(/classes\[0\]\.display/)
  })

  test('мета старого формата роняет разбор на class_id', () => {
    const старая = { classes: [{ model_id: 'x', brand: 'y', display: 'X', platform_group: null }] }
    expect(() => parseCatalog(старая)).toThrow(/classes\[0\]\.class_id/)
  })

  test('brand обязателен: пропуск поля — не то же, что null', () => {
    const битый = { classes: [{ class_id: 'x', members: ['x'], display: 'X' }] }
    expect(() => parseCatalog(битый)).toThrow(/classes\[0\]\.brand/)
  })

  test('класс без участников роняет разбор', () => {
    const битый = { classes: [{ class_id: 'x', members: [], brand: 'y', display: 'X' }] }
    expect(() => parseCatalog(битый)).toThrow(/classes\[0\]\.members/)
  })

  test('участник не строкой роняет разбор с его номером', () => {
    const битый = { classes: [{ class_id: 'x', members: ['x', 7], brand: 'y', display: 'X' }] }
    expect(() => parseCatalog(битый)).toThrow(/classes\[0\]\.members\[1\]/)
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

describe('brandsMatch', () => {
  const классы = parseCatalog(каталог)
  const ауди = классы[0]!
  const другаяАуди = классы[1]!
  const группа = классы[2]!

  test('одна марка у обоих — совпадение', () => {
    expect(brandsMatch(ауди, другаяАуди)).toBe(true)
  })

  test('разные марки не совпадают', () => {
    expect(brandsMatch(ауди, { ...ауди, brand: 'skoda' })).toBe(false)
  })

  test('две разнобрендовые группы — не одна марка', () => {
    // Класс без марки не совпадает и сам с собой: это ответ «марки нет», а не «марка та же».
    expect(brandsMatch(группа, группа)).toBe(false)
    expect(brandsMatch(группа, { ...группа, classId: 'grp__other' })).toBe(false)
  })

  test('класс без марки не совпадает с классом с маркой', () => {
    expect(brandsMatch(группа, ауди)).toBe(false)
    expect(brandsMatch(ауди, группа)).toBe(false)
  })

  test('отсутствующий класс не совпадает ни с чем', () => {
    expect(brandsMatch(null, ауди)).toBe(false)
    expect(brandsMatch(ауди, null)).toBe(false)
    expect(brandsMatch(null, null)).toBe(false)
  })
})

describe('информация о каталоге', () => {
  test('разбирает три поля версии', () => {
    const инфо = parseCatalogInfo({
      format_version: 1,
      generated_at: '2026-08-17T19:42:43Z',
      model_weights_sha256: 'abcdef0123456789abcdef',
      classes: [{ display: 'Audi A5', brand: 'Audi' }],
    })
    expect(инфо).toEqual({
      formatVersion: 1,
      generatedAt: '2026-08-17T19:42:43Z',
      weightsSha: 'abcdef0123456789abcdef',
      классов: 1,
    })
  })

  test('digest склеен из трёх полей и берёт 12 символов sha', () => {
    const инфо = catalogInfo()
    expect(catalogDigest()).toBe(
      `${инфо.formatVersion}:${инфо.weightsSha.slice(0, 12)}:${инфо.классов}`,
    )
  })

  test('digest не пустой и не содержит пробелов', () => {
    expect(catalogDigest().length).toBeGreaterThan(5)
    expect(catalogDigest()).not.toMatch(/\s/)
  })

  test('allClasses отдаёт весь каталог', () => {
    expect(allClasses().length).toBe(catalogInfo().классов)
    expect(allClasses()[0]!.display.length).toBeGreaterThan(0)
  })

  test('битая версия каталога — AssetError', () => {
    expect(() => parseCatalogInfo({ format_version: 'один', classes: [] })).toThrow(AssetError)
  })
})
