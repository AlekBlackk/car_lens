import { describe, expect, test } from 'vitest'
import type { Box } from './plugins'
import { CLASSIFY_INTERVAL_MS, MAX_CROPS, pickForClassify } from './throttle'

const бокс = (conf: number, сторона = 300): Box => ({ x: 0, y: 0, w: сторона, h: сторона, conf })
const кандидат = (id: number, conf: number, classifiedAt: number, сторона = 300) => ({
  id,
  box: бокс(conf, сторона),
  classifiedAt,
})

describe('pickForClassify', () => {
  test('пустой список треков ничего не выбирает', () => {
    expect(pickForClassify([], 1000, 120)).toEqual([])
  })

  test('трек, классифицированный только что, пропускается', () => {
    const треки = [кандидат(1, 0.9, 1000 - CLASSIFY_INTERVAL_MS + 1)]
    expect(pickForClassify(треки, 1000, 120)).toEqual([])
  })

  test('трек ровно на границе интервала классифицируется', () => {
    const треки = [кандидат(1, 0.9, 1000 - CLASSIFY_INTERVAL_MS)]
    expect(pickForClassify(треки, 1000, 120)).toHaveLength(1)
  })

  test('троттлинг раздельный по трекам', () => {
    const треки = [кандидат(1, 0.9, 1000), кандидат(2, 0.8, 0)]
    const выбранные = pickForClassify(треки, 1000, 120)
    expect(выбранные.map((к) => к.id)).toEqual([2])
  })

  test('не больше MAX_CROPS за кадр', () => {
    const треки = [кандидат(1, 0.9, 0), кандидат(2, 0.8, 0), кандидат(3, 0.7, 0), кандидат(4, 0.6, 0)]
    expect(pickForClassify(треки, 1000, 120)).toHaveLength(MAX_CROPS)
  })

  test('порядок отбора — по убыванию conf, затем по id', () => {
    // Единственный выбранный проверяет оба правила разом: трек 3 проигрывает по conf, а
    // трек 2 — по id при равном conf. С `MAX_CROPS = 1` это и есть весь наблюдаемый
    // порядок: остальное отсекается ещё до того, как очередь имеет значение.
    const треки = [кандидат(3, 0.7, 0), кандидат(1, 0.9, 0), кандидат(2, 0.9, 0)]
    expect(pickForClassify(треки, 1000, 120).map((к) => к.id)).toEqual([1])
  })

  test('мелкий бокс не классифицируется вовсе', () => {
    const треки = [кандидат(1, 0.99, 0, 119)]
    expect(pickForClassify(треки, 1000, 120)).toEqual([])
  })

  test('бокс ровно на пороге короткой стороны классифицируется', () => {
    expect(pickForClassify([кандидат(1, 0.99, 0, 120)], 1000, 120)).toHaveLength(1)
  })

  test('короткая сторона считается по меньшей из сторон', () => {
    const узкий = { id: 1, box: { x: 0, y: 0, w: 400, h: 100, conf: 0.9 }, classifiedAt: 0 }
    expect(pickForClassify([узкий], 1000, 120)).toEqual([])
  })

  test('вход не мутируется', () => {
    const треки = [кандидат(3, 0.7, 0), кандидат(1, 0.9, 0)]
    pickForClassify(треки, 1000, 120)
    expect(треки.map((к) => к.id)).toEqual([3, 1])
  })
})
