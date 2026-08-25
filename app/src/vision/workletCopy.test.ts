import { describe, expect, it } from 'vitest'
import { копияПодписей } from './workletCopy'
import type { ПодписьТрека } from './pipeline'

/**
 * Прокси границы воркета живёт ровно один синхронный вызов: `createRunOnJS` отдаёт
 * аргументы, у которых после возврата из колбэка вложенные поля читаются как `undefined`,
 * а `Object.keys` показывает огрызок. Проверено на SM-S918B: подпись, положенная в
 * состояние React, доехала до рендера с единственным ключом `decision`.
 *
 * Подделка ниже воспроизводит ровно это: после `убить()` объект отдаёт `undefined` на всё.
 */
const умирающий = <T extends object>(значение: T): { объект: T; убить: () => void } => {
  let жив = true
  const объект = new Proxy(значение, {
    get: (цель, ключ) => (жив ? Reflect.get(цель, ключ) : undefined),
    ownKeys: (цель) => (жив ? Reflect.ownKeys(цель) : []),
  }) as T
  return { объект, убить: () => (жив = false) }
}

describe('копия подписей через границу воркета', () => {
  const подпись = (): ПодписьТрека => ({
    id: 7,
    box: { x: 1, y: 2, w: 3, h: 4, conf: 0.9 },
    decision: { kind: 'модель', classIdx: 3, confidence: 0.8, calibrated: true },
    tops: [
      { classIdx: 3, sim: 0.7 },
      { classIdx: 4, sim: 0.6 },
    ],
  })

  it('копия переживает смерть исходного прокси', () => {
    const источник = умирающий(подпись())
    const боксИсточник = умирающий(источник.объект.box)
    const снимок = копияПодписей([{ ...подпись(), box: боксИсточник.объект }])
    источник.убить()
    боксИсточник.убить()
    expect(снимок[0]!.box).toEqual({ x: 1, y: 2, w: 3, h: 4, conf: 0.9 })
    expect(снимок[0]!.id).toBe(7)
    expect(снимок[0]!.tops).toEqual([
      { classIdx: 3, sim: 0.7 },
      { classIdx: 4, sim: 0.6 },
    ])
  })

  it('вложенные объекты копии не делят ссылок с источником', () => {
    const источник = подпись()
    const [снимок] = копияПодписей([источник])
    expect(снимок!.box).not.toBe(источник.box)
    expect(снимок!.decision).not.toBe(источник.decision)
    expect(снимок!.tops).not.toBe(источник.tops)
    expect(снимок!.tops[0]).not.toBe(источник.tops[0])
  })

  it('решения всех видов переносятся целиком', () => {
    const виды: ПодписьТрека[] = [
      { ...подпись(), decision: { kind: 'определяю' } },
      { ...подпись(), decision: { kind: 'ближе' } },
      { ...подпись(), decision: { kind: 'неизвестно' } },
      { ...подпись(), decision: { kind: 'бренд', classIdx: 5 } },
    ]
    expect(копияПодписей(виды).map((п) => п.decision)).toEqual(виды.map((п) => п.decision))
  })

  it('пустой список копируется пустым', () => {
    expect(копияПодписей([])).toEqual([])
  })
})
