/**
 * Ветвь «показать только бренд» §6.4 на живом стыке: `pipeline.ts` спрашивает бренды у
 * каталога через `sameBrand`, а `decide` по этому флагу выбирает между «бренд» и
 * «неизвестно». Юнит-тесты `brandsMatch` проверяют само правило, но не то, что конвейер
 * зовёт именно его: подмена вызова на `топ1.classIdx === топ2.classIdx` их не уронила бы.
 *
 * Подменяется не `labels`, а сам ассет, — иначе тест проверял бы мок. Настоящая мета для
 * этого не годится: классов без марки в ней до §11 нет. Поэтому `vi.mock` отдаёт фикстуру,
 * из которой `labels.ts` собирает `CATALOG` при импорте модуля.
 */
import { describe, expect, test, vi } from 'vitest'
import type { Решение } from './decide'
import { allClasses } from './labels'
import { ПУСТОЕ_СОСТОЯНИЕ, stepPipeline, type МостКонвейера } from './pipeline'
import type { Box, EmbedMatchResult, Match } from './plugins'

vi.mock('../../android/models/catalog_meta.json', () => ({
  default: {
    format_version: 1,
    generated_at: '2026-08-25T00:00:00Z',
    model_weights_sha256: 'фикстура',
    classes: [
      {
        class_id: 'grp__coolray',
        members: ['geely__coolray', 'belgee__x50'],
        brand: null,
        display: 'Geely Coolray / Belgee X50',
      },
      {
        class_id: 'grp__golf',
        members: ['volkswagen__golf', 'skoda__octavia'],
        brand: null,
        display: 'VW Golf / Skoda Octavia',
      },
      { class_id: 'audi__a4', members: ['audi__a4'], brand: 'Audi', display: 'Audi A4' },
      { class_id: 'audi__a6', members: ['audi__a6'], brand: 'Audi', display: 'Audi A6' },
      { class_id: 'bmw__x3', members: ['bmw__x3'], brand: 'BMW', display: 'BMW X3' },
    ],
  },
}))

const БОКС: Box = { x: 0, y: 0, w: 300, h: 300, conf: 0.9 }

const топ2 = (первый: number, второй: number): Match[] => [
  { classIdx: первый, sim: 0.9 },
  // Разрыв 0.02 меньше `margin_min` = 0.05: решение уходит в ту самую ветвь §6.4, где всё
  // решает совпадение брендов.
  { classIdx: второй, sim: 0.88 },
]

/** Четыре кадра с шагом 200 мс: наблюдений набирается больше `min_observations` = 3. */
const решениеПоТопам = (топы: Match[]): Решение => {
  const мост: МостКонвейера = {
    embed: (боксы: Box[]): EmbedMatchResult[] =>
      боксы.map(() => ({ embedding: Float32Array.from([1, 0, 0]), matches: топы })),
    matchEma: (вектор: Float32Array): Match[] => (вектор.length > 0 ? топы : []),
  }
  let состояние = ПУСТОЕ_СОСТОЯНИЕ
  let решение: Решение = { kind: 'определяю' }
  for (let кадр = 0; кадр < 4; кадр++) {
    const шаг = stepPipeline(
      состояние,
      { боксы: [БОКС], nowMs: кадр * 200, ширинаКадра: 1920, высотаКадра: 1080 },
      мост,
    )
    состояние = шаг.состояние
    решение = шаг.подписи[0]!.decision
  }
  return решение
}

describe('бренд в решении конвейера', () => {
  test('методика: каталог собран из фикстуры, а не из настоящего ассета', () => {
    // Без этой проверки провалившийся `vi.mock` выглядел бы как зелёные тесты: настоящая
    // мета брендов не теряет, и ветвь «бренд» на ней просто не проверяется.
    expect(allClasses()).toHaveLength(5)
    expect(allClasses()[0]!.brand).toBeNull()
  })

  test('два разнобрендовых класса в топ-2 дают неизвестно, а не бренд', () => {
    expect(решениеПоТопам(топ2(0, 1)).kind).toBe('неизвестно')
  })

  test('две модели одной марки в топ-2 дают бренд', () => {
    // Обратная сторона того же стыка: без неё «неизвестно» выше проходило бы и у конвейера,
    // сравнивающего номера классов вместо брендов.
    const решение = решениеПоТопам(топ2(2, 3))
    expect(решение.kind).toBe('бренд')
    expect(решение.kind === 'бренд' && решение.classIdx).toBe(2)
  })

  test('разные марки в топ-2 дают неизвестно', () => {
    expect(решениеПоТопам(топ2(2, 4)).kind).toBe('неизвестно')
  })
})
