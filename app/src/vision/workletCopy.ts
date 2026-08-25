import type { Решение } from './decide'
import type { ПодписьТрека } from './pipeline'
import type { Box, Match } from './plugins'

/**
 * Снимок подписей, пересёкших границу воркета.
 *
 * Восьмое правило границы `react-native-worklets-core` 1.5.0, добытое пробой на SM-S918B:
 * аргументы `createRunOnJS` действительны ровно на время синхронного вызова. Внутри
 * колбэка `подпись.box.w` читается настоящим числом, но та же ссылка, положенная в
 * состояние React, доезжает до рендера огрызком — `Object.keys` показал единственный ключ
 * `decision`, `box` стал `undefined`, и `mapBoxToView` падал с «Cannot read property 'x' of
 * undefined». Ни ошибки, ни предупреждения на границе при этом нет.
 *
 * Отсюда правило: всё, что переживает вызов колбэка — состояние экрана, ref «Деталей», —
 * обязано быть копией, снятой синхронно. Потребители, читающие поля тут же и складывающие
 * примитивы (`StabilityCollector`, `записатьРаспознанное`), в копии не нуждаются.
 *
 * Копия структурная и написана полями вручную, а не `JSON.parse(JSON.stringify())`:
 * сериализация прокси границы — лишний способ нарваться на её же особенности, и цена её
 * на горячем пути выше при том же результате.
 */
export function копияПодписей(подписи: ПодписьТрека[]): ПодписьТрека[] {
  const снимок: ПодписьТрека[] = new Array(подписи.length)
  for (let i = 0; i < подписи.length; i++) {
    const подпись = подписи[i]!
    снимок[i] = {
      id: подпись.id,
      box: копияБокса(подпись.box),
      decision: копияРешения(подпись.decision),
      tops: копияТопов(подпись.tops),
    }
  }
  return снимок
}

const копияБокса = (box: Box): Box => ({ x: box.x, y: box.y, w: box.w, h: box.h, conf: box.conf })

const копияТопов = (топы: Match[]): Match[] => {
  const снимок: Match[] = new Array(топы.length)
  for (let i = 0; i < топы.length; i++) {
    const топ = топы[i]!
    снимок[i] = { classIdx: топ.classIdx, sim: топ.sim }
  }
  return снимок
}

/**
 * Разбор по видам, а не спред: спред снял бы поля с прокси через `ownKeys`, а именно их
 * прокси и теряет. Перечисление видов заодно ловит новый вид решения ошибкой типа.
 */
const копияРешения = (решение: Решение): Решение => {
  switch (решение.kind) {
    case 'модель':
      return {
        kind: 'модель',
        classIdx: решение.classIdx,
        confidence: решение.confidence,
        calibrated: решение.calibrated,
      }
    case 'бренд':
      return { kind: 'бренд', classIdx: решение.classIdx }
    default:
      return { kind: решение.kind }
  }
}
