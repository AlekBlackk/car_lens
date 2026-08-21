import type { Box } from './plugins'

/** §6.2 спеки: не чаще 8 Гц на трек. Трекера ещё нет, поэтому — на кадр. */
export const CLASSIFY_INTERVAL_MS = 125

/** §6.2 спеки: не больше двух кропов за кадр. Столько же слотов в мосту. */
export const MAX_CROPS = 2

/** Какие боксы классифицировать сейчас. Чистая функция — тестируется без worklet. */
export function shouldClassify(now: number, lastAt: number, boxes: Box[]): Box[] {
  'worklet'
  if (boxes.length === 0) return []
  if (now - lastAt < CLASSIFY_INTERVAL_MS) return []
  // `boxes.slice()`, а не `[...boxes]`: spread babel разворачивает в хелпер
  // `_toConsumableArray`, обычную JS-функцию, которая попадает в замыкание воркета и
  // роняет кадр — «Regular javascript function '_toConsumableArray' cannot be shared»
  // (проверено на SM-S918B). Копия нужна: `sort` правит массив на месте, а `boxes`
  // принадлежит вызывающему.
  return boxes.slice().sort((первый, второй) => второй.conf - первый.conf).slice(0, MAX_CROPS)
}
