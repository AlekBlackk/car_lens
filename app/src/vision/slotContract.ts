import type { Frame } from 'react-native-vision-camera'
import { embedAndMatch, type Box } from './plugins'

/**
 * Самопроверка контракта слотов `SharedArray` (§4.8) прямо в конвейере на устройстве.
 *
 * Почему не инструментальный тест. Проверка задумывалась как `:app:androidTest`
 * (`SharedArrayContractTest`), который конструирует `VisionCameraProxy` руками и зовёт
 * `CarVisionEmbedPlugin.callback` напрямую. На новой архитектуре RN это невозможно:
 * приложение живёт в bridgeless-режиме (`ReactContext.isBridgeless == true`), а
 * `VisionCameraProxy(ReactApplicationContext)` рассчитан на старую — на устройстве его
 * конструктор падает нативно внутри `librnworklets.so`
 * (`SIGSEGV` в `JsiWorkletContext::initialize` → `JsiConsoleDecorator::initialize`,
 * проверено на SM-S918B, в том числе при вызове с JS-потока). Настоящий прокси создаёт
 * сама VisionCamera внутри JS-рантайма, и достать его из Kotlin нельзя.
 *
 * Что проверяется здесь — то же, ради чего писался тот тест, но на настоящем кадре и
 * настоящем `SharedArray`, пришедшем через мост:
 *
 * 1. Два бокса одного вызова лежат в независимой памяти. Один слот на оба означал бы,
 *    что вторая машина в кадре получает подпись первой.
 * 2. Снятая копия переживает следующий вызов плагина. Слот переиспользуется нативной
 *    стороной, и если бы `plugins.ts` отдавал наружу вид на его память вместо копии,
 *    значения менялись бы задним числом.
 *
 * Чего проверить с этой стороны нельзя: что слот именно **переиспользован**, а не
 * выделен заново — сам объект `SharedArray` в JS не виден, наружу приходит только его
 * буфер. Это ограничение, а не недосмотр.
 *
 * Боксы берутся синтетические, а не из детектора: контракт памяти не зависит от того,
 * есть ли в кадре машина, и проверка не должна ждать удачной сцены.
 */
export type РезультатКонтракта = { ок: boolean; отчёт: string }

/*
 * Имена в этом модуле латиницей не по стилю, а по необходимости: всё, что воркет
 * захватывает из области видимости модуля, становится ключом `__closure`, а ключи с
 * не-ASCII символами `react-native-worklets-core` 1.5.0 молча теряет. Стража —
 * `workletClosure.test.ts`, она это и поймала.
 */

const FIRST_BOX: Box = { x: 10, y: 10, w: 100, h: 100, conf: 1 }
const SECOND_BOX: Box = { x: 160, y: 120, w: 120, h: 120, conf: 1 }

function sameValues(первый: Float32Array, второй: readonly number[]): boolean {
  'worklet'
  if (первый.length !== второй.length) {
    return false
  }
  for (let индекс = 0; индекс < первый.length; индекс++) {
    if (первый[индекс] !== второй[индекс]) {
      return false
    }
  }
  return true
}

function snapshot(вектор: Float32Array): number[] {
  'worklet'
  const значения: number[] = []
  for (let индекс = 0; индекс < вектор.length; индекс++) {
    значения.push(вектор[индекс]!)
  }
  return значения
}

export function checkSlotContract(
  frame: Frame,
  pad: number,
  deviceDegrees: number,
  previewDegrees: number,
): РезультатКонтракта {
  'worklet'
  const первый = embedAndMatch(frame, [FIRST_BOX, SECOND_BOX], pad, 1, deviceDegrees, previewDegrees)
  if (первый.results.length !== 2) {
    return { ок: false, отчёт: `на два бокса вернулось ${первый.results.length} результатов` }
  }
  const векторА = первый.results[0]!.embedding
  // Признак берётся по ссылкам на буферы слотов, а не сравнением эмбеддингов: на
  // однородной сцене (камера в стол, тёмный кадр) два разных кропа дают численно
  // одинаковый вектор совершенно законно, и проверка по значениям врала бы «слоты не
  // независимы» — проверено на устройстве, именно так она и соврала дважды подряд.
  const независимы = первый.slotsDistinct

  // Снимок снимается до следующего вызова — именно его переживание и есть контракт.
  const доВторого = snapshot(векторА)
  embedAndMatch(frame, [SECOND_BOX], pad, 1, deviceDegrees, previewDegrees)
  const пережил = sameValues(векторА, доВторого)

  return {
    ок: независимы && пережил,
    отчёт:
      `слоты независимы: ${независимы ? 'да' : 'НЕТ'}, ` +
      `копия пережила следующий вызов: ${пережил ? 'да' : 'НЕТ'}, ` +
      `длина эмбеддинга ${векторА.length}`,
  }
}
