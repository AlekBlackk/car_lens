import { transformFileSync } from '@babel/core'
import { describe, expect, test } from 'vitest'

/**
 * Стража против молчаливого класса ошибок, стоившего приёмки M-1.6 на устройстве:
 * `react-native-worklets-core` 1.5.0 не переносит в worklet-рантайм ключи `__closure`
 * с не-ASCII символами. Захват приезжает `undefined`, исключения не бросается, и вместо
 * внятной поломки видно «плагин carvisionDetect не зарегистрирован» — при том, что
 * logcat рапортует `FrameProcessorPluginRegistry: "carvisionDetect" found!`.
 *
 * Тест гоняет реальный babel-конвейер приложения по файлам с директивой `'worklet'` и
 * требует, чтобы каждое захваченное имя было ASCII. Локальные переменные внутри тела
 * воркета под правило не попадают — они перенос переживают и остаются кириллическими.
 */
const ФАЙЛЫ_С_ВОРКЛЕТАМИ = [
  'App.tsx',
  'src/vision/plugins.ts',
  'src/vision/throttle.ts',
  'src/vision/latency.ts',
  'src/vision/thresholds.ts',
  'src/vision/labels.ts',
  'src/vision/ema.ts',
  'src/vision/tracker.ts',
  'src/vision/decide.ts',
  'src/vision/pipeline.ts',
]

/** Печатный ASCII: коды 0x20..0x7E. Регулярка тут не используется намеренно — экранирование
 * диапазона само по себе источник ошибок, а перебор кодов читается однозначно. */
function толькоASCII(имя: string): boolean {
  for (const символ of кодыСимволов(имя)) {
    if (символ < 0x20 || символ > 0x7e) {
      return false
    }
  }
  return true
}

function кодыСимволов(имя: string): number[] {
  const коды: number[] = []
  for (let позиция = 0; позиция < имя.length; позиция++) {
    коды.push(имя.charCodeAt(позиция))
  }
  return коды
}

function транспилировать(файл: string): string {
  const результат = transformFileSync(файл, {
    filename: файл,
    presets: ['module:@react-native/babel-preset'],
    plugins: [['react-native-worklets-core/plugin']],
    babelrc: false,
    configFile: false,
  })
  return результат?.code ?? ''
}

function ключиЗамыканий(файл: string): string[] {
  const код = транспилировать(файл)
  const ключи: string[] = []
  for (const кусок of код.matchAll(/_closure\s*=\s*\{([^}]*)\}/g)) {
    for (const пара of кусок[1]!.split(',')) {
      const имя = пара.split(':')[0]?.trim()
      if (имя != null && имя.length > 0) {
        ключи.push(имя)
      }
    }
  }
  return ключи
}

/**
 * Имена, которые сам плагин worklets-core объявил как init-data вложенного воркета:
 * `var _worklet_<число>_init_data = {code: "...", location: ..., ...}`. Внешний воркет
 * захватывает их, когда внутри него объявлен ещё один воркет — например мост конвейера в
 * `App.tsx`, который обязан жить внутри frame processor'а, потому что только там есть
 * `frame`. Это объектные литералы со строками, а не функции, и worklet-рантайм переносит
 * их штатно.
 *
 * Набор собирается из объявлений в том же файле, а не по маске имени: иначе под правило
 * пролезла бы любая переменная с подходящим названием, и стража ниже ослабла бы до
 * проверки орфографии.
 */
function инитДанныеВоркетов(файл: string): Set<string> {
  const имена = new Set<string>()
  for (const кусок of транспилировать(файл).matchAll(/var\s+(_worklet_\d+_init_data)\s*=\s*\{code:/g)) {
    имена.add(кусок[1]!)
  }
  return имена
}

describe('замыкания воркетов', () => {
  test.each(ФАЙЛЫ_С_ВОРКЛЕТАМИ)('%s: все захваченные имена ASCII', (файл) => {
    expect(ключиЗамыканий(файл).filter((имя) => !толькоASCII(имя))).toEqual([])
  })

  test('babel-конвейер вообще создаёт замыкания — иначе проверка выше пуста', () => {
    expect(ключиЗамыканий('App.tsx').length).toBeGreaterThan(0)
  })

  test.each(ФАЙЛЫ_С_ВОРКЛЕТАМИ)('%s: в замыкании нет babel-хелперов', (файл) => {
    // Хелперы транспайлера (`_toConsumableArray` от spread, `_createForOfIteratorHelper`
    // от for-of) — обычные функции: worklet-рантайм отказывается их переносить и роняет
    // кадр. Все наши имена пишутся без ведущего подчёркивания, поэтому признак точный.
    // Исключение — init-data вложенных воркетов, которую генерирует сам плагин
    // worklets-core (см. `инитДанныеВоркетов`).
    const свои = инитДанныеВоркетов(файл)
    const подозрительные = ключиЗамыканий(файл).filter((имя) => имя.startsWith('_') && !свои.has(имя))
    expect(подозрительные).toEqual([])
  })

  test('исключение для init-data не пропускает babel-хелперы', () => {
    // Стража самой поблажки: в `App.tsx` вложенные воркеты есть, значит и init-data есть,
    // а хелперов транспайлера среди захватов быть не должно ни одного.
    const свои = инитДанныеВоркетов('App.tsx')
    expect(свои.size).toBeGreaterThan(0)
    for (const имя of свои) {
      expect(имя).toMatch(/^_worklet_\d+_init_data$/)
    }
  })

  test('кириллический ключ ловится — проверка не тавтология', () => {
    expect(толькоASCII('detectPlugin')).toBe(true)
    expect(толькоASCII('плагинДетекции')).toBe(false)
  })
})
