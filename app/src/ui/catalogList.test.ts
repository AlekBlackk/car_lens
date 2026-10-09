import { describe, expect, it } from 'vitest'
import { allClasses, type Класс } from '../vision/labels'
import { РАЗДЕЛ_БЕЗ_МАРКИ, подходитПодЗапрос, разделыКаталога } from './catalogList'

const класс = (classId: string, display: string, brand: string | null, members: string[]): Класс => ({
  classId,
  display,
  brand,
  members,
})

const каталог: Класс[] = [
  класс('toyota__camry', 'Toyota Camry', 'Toyota', ['toyota__camry']),
  класс('audi__a4', 'Audi A4', 'Audi', ['audi__a4']),
  класс('grp__coolray', 'Geely Coolray / Belgee X50', null, ['geely__coolray', 'belgee__x50']),
  класс('toyota__rav4', 'Toyota RAV4', 'Toyota', ['toyota__rav4']),
]

describe('подходитПодЗапрос', () => {
  it('пустой и пробельный запрос пропускает всё', () => {
    expect(подходитПодЗапрос(каталог[0]!, '')).toBe(true)
    expect(подходитПодЗапрос(каталог[0]!, '   ')).toBe(true)
  })

  it('ищет по названию и по марке без учёта регистра', () => {
    expect(подходитПодЗапрос(каталог[0]!, 'CAMRY')).toBe(true)
    expect(подходитПодЗапрос(каталог[0]!, ' toyota ')).toBe(true)
    expect(подходитПодЗапрос(каталог[0]!, 'audi')).toBe(false)
  })

  it('группу без марки находит по участнику', () => {
    expect(подходитПодЗапрос(каталог[2]!, 'belgee')).toBe(true)
    expect(подходитПодЗапрос(каталог[2]!, 'x50')).toBe(true)
  })
})

describe('разделыКаталога', () => {
  it('группирует по маркам по алфавиту, раздел без марки — последним', () => {
    const разделы = разделыКаталога(каталог, '')
    expect(разделы.map((раздел) => раздел.название)).toEqual(['Audi', 'Toyota', РАЗДЕЛ_БЕЗ_МАРКИ])
  })

  it('внутри марки сохраняет порядок каталога', () => {
    const toyota = разделыКаталога(каталог, '').find((раздел) => раздел.название === 'Toyota')
    expect(toyota?.data.map((запись) => запись.display)).toEqual(['Toyota Camry', 'Toyota RAV4'])
  })

  it('запрос оставляет только подошедшие классы и не возвращает пустые разделы', () => {
    const разделы = разделыКаталога(каталог, 'rav')
    expect(разделы).toHaveLength(1)
    expect(разделы[0]!.название).toBe('Toyota')
    expect(разделы[0]!.data.map((запись) => запись.classId)).toEqual(['toyota__rav4'])
  })

  it('на запрос без совпадений отдаёт пустой список', () => {
    expect(разделыКаталога(каталог, 'нет такой машины')).toEqual([])
  })

  it('настоящий каталог раскладывается без потерь', () => {
    const все = allClasses()
    const разделы = разделыКаталога(все, '')
    const вРазделах = разделы.flatMap((раздел) => раздел.data.map((запись) => запись.classId))
    expect(вРазделах.sort()).toEqual(все.map((запись) => запись.classId).sort())
  })
})
