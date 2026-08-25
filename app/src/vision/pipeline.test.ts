import { describe, expect, test } from 'vitest'
import { EMA_ALPHA } from './ema'
import type { Box, EmbedMatchResult, Match } from './plugins'
import {
  BRIDGE_CONTRACT_ERROR,
  ПУСТОЕ_СОСТОЯНИЕ,
  stepPipeline,
  MAX_HYPOTHESES,
  type МостКонвейера,
  type СостояниеКонвейера,
} from './pipeline'
import { CLASSIFY_INTERVAL_MS, MAX_CROPS } from './throttle'
import { LOST_MAX_MS } from './tracker'

const бокс = (x: number, conf = 0.9, сторона = 300): Box => ({ x, y: 0, w: сторона, h: сторона, conf })

/** Мост, у которого мгновенный топ-1 и топ-1 по EMA намеренно разные. */
const мост = (правки: Partial<МостКонвейера> = {}): МостКонвейера & { вызовыEmbed: Box[][] } => {
  const вызовыEmbed: Box[][] = []
  const базовый = {
    вызовыEmbed,
    embed(боксы: Box[]): EmbedMatchResult[] {
      вызовыEmbed.push(боксы)
      const итог: EmbedMatchResult[] = []
      for (let номер = 0; номер < боксы.length; номер++) {
        итог.push({
          embedding: Float32Array.from([1, 0, 0]),
          matches: [
            { classIdx: 77, sim: 0.95 },
            { classIdx: 78, sim: 0.1 },
          ],
        })
      }
      return итог
    },
    matchEma(вектор: Float32Array): Match[] {
      // Длина проверяется, чтобы параметр не оказался неиспользованным: eslint считает
      // такой аргумент ошибкой, а тестовые файлы линтуются наравне с остальными.
      return вектор.length > 0
        ? [
            { classIdx: 11, sim: 0.9 },
            { classIdx: 12, sim: 0.1 },
          ]
        : []
    },
  }
  return { ...базовый, ...правки }
}

const вход = (боксы: Box[], nowMs: number) => ({
  боксы,
  nowMs,
  ширинаКадра: 1920,
  высотаКадра: 1080,
})

/** Прогнать конвейер n кадров подряд с одним и тем же боксом, шаг по времени — 200 мс. */
const прогнать = (кадров: number, м: МостКонвейера): СостояниеКонвейера => {
  let состояние = ПУСТОЕ_СОСТОЯНИЕ
  for (let кадр = 0; кадр < кадров; кадр++) {
    состояние = stepPipeline(состояние, вход([бокс(0)], кадр * 200), м).состояние
  }
  return состояние
}

describe('stepPipeline', () => {
  test('шаг не мутирует входное состояние', () => {
    const исходное = ПУСТОЕ_СОСТОЯНИЕ
    stepPipeline(исходное, вход([бокс(0)], 1000), мост())
    expect(исходное.tracks).toHaveLength(0)
    expect(исходное.nextId).toBe(1)
  })

  test('первый кадр рождает трек и сразу его классифицирует', () => {
    const м = мост()
    const итог = stepPipeline(ПУСТОЕ_СОСТОЯНИЕ, вход([бокс(0)], 1000), м)
    expect(итог.состояние.tracks).toHaveLength(1)
    expect(итог.состояние.tracks[0]!.observations).toBe(1)
    expect(м.вызовыEmbed).toHaveLength(1)
  })

  test('до min_observations подпись — определяю', () => {
    const итог = stepPipeline(ПУСТОЕ_СОСТОЯНИЕ, вход([бокс(0)], 1000), мост())
    expect(итог.подписи[0]!.decision.kind).toBe('определяю')
  })

  test('решение считается по EMA, а не по мгновенным matches', () => {
    const состояние = прогнать(4, мост())
    const итог = stepPipeline(состояние, вход([бокс(0)], 4 * 200), мост())
    const решение = итог.подписи[0]!.decision
    expect(решение.kind).toBe('модель')
    expect(решение.kind === 'модель' && решение.classIdx).toBe(11)
  })

  test('троттлинг: два кадра подряд — одна классификация', () => {
    const м = мост()
    const первый = stepPipeline(ПУСТОЕ_СОСТОЯНИЕ, вход([бокс(0)], 1000), м)
    stepPipeline(первый.состояние, вход([бокс(0)], 1000 + CLASSIFY_INTERVAL_MS - 1), м)
    expect(м.вызовыEmbed).toHaveLength(1)
  })

  test('два трека в одном кадре классифицируются по очереди, а не оба разом', () => {
    // `MAX_CROPS = 1`: за кадр считается один кроп, поэтому второй трек ждёт следующего
    // кадра. Проверяется именно очередь, а не «второй не считается вовсе» — потерять
    // вторую машину эта правка не должна.
    const м = мост()
    const первый = stepPipeline(ПУСТОЕ_СОСТОЯНИЕ, вход([бокс(0), бокс(1000)], 1000), м)
    expect(первый.состояние.tracks).toHaveLength(2)
    expect(м.вызовыEmbed[0]).toHaveLength(1)

    const второй = stepPipeline(первый.состояние, вход([бокс(0), бокс(1000)], 1010), м)
    expect(м.вызовыEmbed[1]).toHaveLength(1)
    // Первым кадром посчитан один трек, вторым — другой: у обоих есть наблюдение.
    for (const трек of второй.состояние.tracks) {
      expect(трек.observations).toBe(1)
    }
  })

  test('за кадр не больше MAX_CROPS кропов при четырёх треках', () => {
    const м = мост()
    const боксы = [бокс(0), бокс(400), бокс(800), бокс(1200)]
    const итог = stepPipeline(ПУСТОЕ_СОСТОЯНИЕ, вход(боксы, 1000), м)
    expect(итог.состояние.tracks).toHaveLength(4)
    expect(м.вызовыEmbed[0]!.length).toBeLessThanOrEqual(MAX_CROPS)
  })

  test('EMA переживает кадр без классификации', () => {
    const м = мост()
    const первый = stepPipeline(ПУСТОЕ_СОСТОЯНИЕ, вход([бокс(0)], 1000), м)
    const второй = stepPipeline(первый.состояние, вход([бокс(0)], 1010), м)
    expect(второй.состояние.tracks[0]!.ema).toEqual(первый.состояние.tracks[0]!.ema)
    expect(второй.состояние.tracks[0]!.observations).toBe(1)
  })

  test('EMA не течёт между треками', () => {
    const м = мост({
      embed(боксы: Box[]): EmbedMatchResult[] {
        const итог: EmbedMatchResult[] = []
        for (let номер = 0; номер < боксы.length; номер++) {
          итог.push({
            embedding: номер === 0 ? Float32Array.from([1, 0]) : Float32Array.from([0, 1]),
            matches: [{ classIdx: номер, sim: 0.9 }],
          })
        }
        return итог
      },
    })
    const итог = stepPipeline(ПУСТОЕ_СОСТОЯНИЕ, вход([бокс(0), бокс(1000)], 1000), м)
    expect(итог.состояние.tracks[0]!.ema).not.toEqual(итог.состояние.tracks[1]!.ema)
  })

  test('трек, вернувшийся внутри LOST_MAX_MS, продолжает свою EMA', () => {
    const м = мост()
    const первый = stepPipeline(ПУСТОЕ_СОСТОЯНИЕ, вход([бокс(0)], 1000), м)
    const пропуск = stepPipeline(первый.состояние, вход([], 1000 + LOST_MAX_MS), м)
    const возврат = stepPipeline(пропуск.состояние, вход([бокс(0)], 1000 + LOST_MAX_MS + 10), м)
    expect(возврат.состояние.tracks[0]!.id).toBe(первый.состояние.tracks[0]!.id)
    expect(возврат.состояние.tracks[0]!.observations).toBeGreaterThan(1)
  })

  test('переродившийся трек начинает с нуля и снова проходит через определяю', () => {
    const м = мост()
    const первый = stepPipeline(ПУСТОЕ_СОСТОЯНИЕ, вход([бокс(0)], 1000), м)
    const смерть = stepPipeline(первый.состояние, вход([], 1000 + LOST_MAX_MS + 1), м)
    expect(смерть.состояние.tracks).toHaveLength(0)
    const заново = stepPipeline(смерть.состояние, вход([бокс(0)], 2000), м)
    expect(заново.состояние.tracks[0]!.id).not.toBe(первый.состояние.tracks[0]!.id)
    expect(заново.подписи[0]!.decision.kind).toBe('определяю')
  })

  test('битый эмбеддинг не увеличивает наблюдений', () => {
    const м = мост({
      embed(): EmbedMatchResult[] {
        return [{ embedding: Float32Array.from([NaN, 0, 0]), matches: [{ classIdx: 1, sim: 0.9 }] }]
      },
    })
    const итог = stepPipeline(ПУСТОЕ_СОСТОЯНИЕ, вход([бокс(0)], 1000), м)
    expect(итог.состояние.tracks[0]!.observations).toBe(0)
    expect(итог.состояние.tracks[0]!.ema).toBeNull()
  })

  test('короткий ответ моста роняет кадр целиком и не трогает ни один трек', () => {
    const начальное = прогнать(3, мост())
    const наблюденийДо = начальное.tracks[0]!.observations
    const emaДо = начальное.tracks[0]!.ema
    const битый = мост({
      embed(): EmbedMatchResult[] {
        return []
      },
    })
    expect(() => stepPipeline(начальное, вход([бокс(0)], 10_000), битый)).toThrow(
      BRIDGE_CONTRACT_ERROR,
    )
    expect(начальное.tracks[0]!.observations).toBe(наблюденийДо)
    expect(начальное.tracks[0]!.ema).toBe(emaДо)
  })

  test('события классификации несут оба топ-1 и ключ решения', () => {
    const состояние = прогнать(4, мост())
    const итог = stepPipeline(состояние, вход([бокс(0)], 4 * 200), мост())
    expect(итог.события).toHaveLength(1)
    expect(итог.события[0]!.top1Мгновенный).toBe(77)
    expect(итог.события[0]!.top1Ema).toBe(11)
    expect(итог.события[0]!.decisionKey).toBe('модель:11')
  })

  test('кадр без классификации событий не даёт', () => {
    const м = мост()
    const первый = stepPipeline(ПУСТОЕ_СОСТОЯНИЕ, вход([бокс(0)], 1000), м)
    const второй = stepPipeline(первый.состояние, вход([бокс(0)], 1010), м)
    expect(второй.события).toHaveLength(0)
  })

  test('EMA_ALPHA применяется: второй эмбеддинг сдвигает вектор, но не заменяет его', () => {
    const м = мост({
      embed(): EmbedMatchResult[] {
        return [{ embedding: Float32Array.from([0, 1]), matches: [{ classIdx: 1, sim: 0.9 }] }]
      },
    })
    const первый = stepPipeline(
      ПУСТОЕ_СОСТОЯНИЕ,
      вход([бокс(0)], 1000),
      мост({
        embed(): EmbedMatchResult[] {
          return [{ embedding: Float32Array.from([1, 0]), matches: [{ classIdx: 1, sim: 0.9 }] }]
        },
      }),
    )
    const второй = stepPipeline(первый.состояние, вход([бокс(0)], 1000 + CLASSIFY_INTERVAL_MS), м)
    const вектор = второй.состояние.tracks[0]!.ema!
    expect(вектор[0]).toBeGreaterThan(0)
    expect(вектор[1]).toBeGreaterThan(0)
    expect(вектор[1]).toBeLessThan(1)
    expect(EMA_ALPHA).toBe(0.35)
  })

  test('трек несёт не больше MAX_HYPOTHESES гипотез и в порядке моста', () => {
    const м = мост({
      matchEma: (): Match[] => [
        { classIdx: 1, sim: 0.9 },
        { classIdx: 2, sim: 0.8 },
        { classIdx: 3, sim: 0.7 },
        { classIdx: 4, sim: 0.6 },
        { classIdx: 5, sim: 0.5 },
      ],
    })
    const итог = stepPipeline(ПУСТОЕ_СОСТОЯНИЕ, вход([бокс(0)], 1000), м)
    const топы = итог.состояние.tracks[0]!.tops
    expect(MAX_HYPOTHESES).toBe(3)
    expect(топы).toHaveLength(MAX_HYPOTHESES)
    expect(топы.map((матч) => матч.classIdx)).toEqual([1, 2, 3])
    expect(итог.подписи[0]!.tops).toEqual(топы)
  })

  test('решение считают две первые гипотезы: третья на него не влияет', () => {
    const м = мост({
      matchEma: (): Match[] => [
        { classIdx: 11, sim: 0.9 },
        { classIdx: 12, sim: 0.1 },
        // Ближе к топ-1, чем позволяет marginMin: попади она в топ-2, «модель» стала бы
        // «неизвестно». Тест на то, что решатель берёт первые две гипотезы, а не любые две.
        { classIdx: 13, sim: 0.89 },
      ],
    })
    const состояние = прогнать(4, м)
    const итог = stepPipeline(состояние, вход([бокс(0)], 800), м)
    const решение = итог.подписи[0]!.decision
    expect(решение.kind).toBe('модель')
    expect(решение.kind === 'модель' && решение.classIdx).toBe(11)
    expect(итог.подписи[0]!.tops).toHaveLength(3)
  })

  test('одна гипотеза от моста — один элемент в топах, решение без топ-2', () => {
    const м = мост({ matchEma: (): Match[] => [{ classIdx: 7, sim: 0.95 }] })
    const состояние = прогнать(4, м)
    const итог = stepPipeline(состояние, вход([бокс(0)], 800), м)
    expect(итог.состояние.tracks[0]!.tops).toHaveLength(1)
    expect(итог.подписи[0]!.decision.kind).toBe('модель')
  })
})

/**
 * Стража против дефекта, стоившего приёмки M-1.6 на устройстве: `SharedValue` из
 * `react-native-worklets-core` 1.5.0 не переносит ключи объекта с не-ASCII символами —
 * ни из JS в воркет, ни обратно. Проверено на SM-S918B: воркет записал
 * `{ проба: 1, probe: 1 }`, на следующем кадре прочитал `проба=undefined`,
 * `probe=number`. Состояние конвейера живёт именно в `SharedValue`, поэтому его поля
 * обязаны быть ASCII — иначе каждый кадр начинается с пустого состояния, а
 * `stepPipeline` падает на первом же чтении.
 *
 * Тест моделирует границу дословно: между кадрами состояние проходит через фильтр,
 * выбрасывающий не-ASCII ключи. Конвейер обязан этого не заметить.
 */
const толькоASCIIКлючи = (значение: unknown): unknown => {
  if (Array.isArray(значение)) return значение.map(толькоASCIIКлючи)
  if (значение === null || typeof значение !== 'object') return значение
  if (ArrayBuffer.isView(значение)) return значение
  const итог: Record<string, unknown> = {}
  for (const [ключ, поле] of Object.entries(значение as Record<string, unknown>)) {
    let ascii = true
    for (let позиция = 0; позиция < ключ.length; позиция++) {
      const код = ключ.charCodeAt(позиция)
      if (код < 0x20 || код > 0x7e) ascii = false
    }
    if (ascii) итог[ключ] = толькоASCIIКлючи(поле)
  }
  return итог
}

describe('состояние переживает SharedValue', () => {
  test('методика: фильтр действительно выбрасывает кириллический ключ', () => {
    expect(толькоASCIIКлючи({ ключ: 1, key: 2 })).toEqual({ key: 2 })
  })

  test('трек живёт через границу: конвейер не падает и продолжает тот же трек', () => {
    const м = мост()
    let состояние = ПУСТОЕ_СОСТОЯНИЕ
    for (let кадр = 0; кадр < 4; кадр++) {
      const шаг = stepPipeline(состояние, вход([бокс(100)], кадр * 200), м)
      состояние = толькоASCIIКлючи(шаг.состояние) as СостояниеКонвейера
    }
    const треки = Object.values(состояние)[0] as unknown[]
    expect(Array.isArray(треки)).toBe(true)
    expect(треки).toHaveLength(1)
  })

  test('наблюдения копятся, а не обнуляются каждым кадром', () => {
    const м = мост()
    let состояние = ПУСТОЕ_СОСТОЯНИЕ
    let последний = stepPipeline(состояние, вход([бокс(100)], 0), м)
    for (let кадр = 1; кадр < 5; кадр++) {
      состояние = толькоASCIIКлючи(последний.состояние) as СостояниеКонвейера
      последний = stepPipeline(состояние, вход([бокс(100)], кадр * 200), м)
    }
    // Пять кадров с шагом 200 мс — пять классификаций: троттлинг §6.2 пропускает все.
    expect(последний.подписи[0]!.decision.kind).not.toBe('определяю')
  })
})

/**
 * Та же граница, что и у состояния, но в другую сторону: подписи и события уезжают на
 * JS-сторону через `Worklets.createRunOnJS`, и там теряются ровно те же не-ASCII ключи.
 * Проверено на SM-S918B: объект `{ проба: 1, probe: 1, вложенный: { поле: 1 } }` приехал
 * как `проба=undefined`, `probe=number`, `вложенный=undefined` — поддерево под
 * кириллическим ключом пропало целиком. Строковые значения при этом целы: `'определяю'`
 * доезжает девятью символами, поэтому кириллица в значениях `вид` остаётся.
 */
describe('подписи и события переживают runOnJS', () => {
  const прогнатьОдин = () => stepPipeline(ПУСТОЕ_СОСТОЯНИЕ, вход([бокс(100)], 0), мост())

  test('у подписи переживают рамка, решение и гипотезы', () => {
    const итог = прогнатьОдин()
    const подпись = (толькоASCIIКлючи(итог.подписи) as typeof итог.подписи)[0]!
    expect(подпись.id).toBe(1)
    expect(Object.keys(подпись).sort()).toEqual(Object.keys(итог.подписи[0]!).sort())
  })

  test('у решения переживает вид', () => {
    const итог = прогнатьОдин()
    const подпись = (толькоASCIIКлючи(итог.подписи) as typeof итог.подписи)[0]!
    expect(Object.keys(подпись.decision)).toContain('kind')
  })

  test('у события переживает ключ решения', () => {
    const итог = stepPipeline(ПУСТОЕ_СОСТОЯНИЕ, вход([бокс(100)], 0), мост())
    const событие = (толькоASCIIКлючи(итог.события) as typeof итог.события)[0]!
    expect(Object.keys(событие)).toContain('decisionKey')
  })
})
