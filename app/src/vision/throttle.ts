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
  return [...boxes].sort((первый, второй) => второй.conf - первый.conf).slice(0, MAX_CROPS)
}
