import { describe, expect, test } from 'vitest'
import { classLabel } from './labels'
import { CALIBRATION, THRESHOLDS } from './thresholds'

/**
 * Стража против дефекта, который дважды прошёл мимо всех прежних проверок и оба раза был
 * пойман только на устройстве: `react-native-worklets-core` 1.5.0 переносит между
 * рантаймами значения, но теряет **имена ключей объектов** с не-ASCII символами.
 *
 * `workletClosure.test.ts` рядом проверяет имена захваченных переменных — то есть ключи
 * самого `__closure`. Этого мало: захваченным может быть объект, и тогда его собственные
 * поля едут той же дорогой. Так `CALIBRATION.узлы` приезжал `undefined`, а
 * `confidenceFromSim` падал с «Cannot convert undefined value to object» при зелёном
 * `npm test` — Node про эту границу не знает ничего.
 *
 * Значения при этом целы: строка `'определяю'` доезжает девятью символами, проверено
 * пробой на SM-S918B. Правило касается только имён ключей.
 */
const путиСНеASCII = (значение: unknown, путь = ''): string[] => {
  if (Array.isArray(значение)) {
    return значение.flatMap((элемент, номер) => путиСНеASCII(элемент, `${путь}[${номер}]`))
  }
  if (значение === null || typeof значение !== 'object') return []
  const плохие: string[] = []
  for (const [ключ, поле] of Object.entries(значение as Record<string, unknown>)) {
    for (let позиция = 0; позиция < ключ.length; позиция++) {
      const код = ключ.charCodeAt(позиция)
      if (код < 0x20 || код > 0x7e) {
        плохие.push(`${путь}.${ключ}`)
        break
      }
    }
    плохие.push(...путиСНеASCII(поле, `${путь}.${ключ}`))
  }
  return плохие
}

describe('данные, захватываемые воркетами, переживают границу', () => {
  test('методика: кириллический ключ был бы найден, в том числе вложенный', () => {
    expect(путиСНеASCII({ ключ: 1, key: { поле: 2 } })).toEqual(['.ключ', '.key.поле'])
  })

  test('THRESHOLDS: все ключи ASCII', () => {
    expect(путиСНеASCII(THRESHOLDS)).toEqual([])
  })

  test('CALIBRATION: все ключи ASCII', () => {
    expect(путиСНеASCII(CALIBRATION)).toEqual([])
  })

  test('класс каталога: все ключи ASCII', () => {
    const первый = classLabel(0)
    expect(первый).not.toBeNull()
    expect(путиСНеASCII(первый)).toEqual([])
  })
})
