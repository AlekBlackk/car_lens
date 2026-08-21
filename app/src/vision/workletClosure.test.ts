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

function ключиЗамыканий(файл: string): string[] {
  const результат = transformFileSync(файл, {
    filename: файл,
    presets: ['module:@react-native/babel-preset'],
    plugins: [['react-native-worklets-core/plugin']],
    babelrc: false,
    configFile: false,
  })
  const код = результат?.code ?? ''
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

describe('замыкания воркетов', () => {
  test.each(ФАЙЛЫ_С_ВОРКЛЕТАМИ)('%s: все захваченные имена ASCII', (файл) => {
    expect(ключиЗамыканий(файл).filter((имя) => !толькоASCII(имя))).toEqual([])
  })

  test('babel-конвейер вообще создаёт замыкания — иначе проверка выше пуста', () => {
    expect(ключиЗамыканий('App.tsx').length).toBeGreaterThan(0)
  })

  test('кириллический ключ ловится — проверка не тавтология', () => {
    expect(толькоASCII('detectPlugin')).toBe(true)
    expect(толькоASCII('плагинДетекции')).toBe(false)
  })
})
