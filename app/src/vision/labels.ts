/**
 * Человекочитаемые подписи классов из `catalog_meta.json` (§13.2).
 *
 * Мост отдаёт только `classIdx` — §13.3 разрешает через границу лишь числа, — а имя
 * класса нужно интерфейсу. Файл тот же, что лежит рядом с `catalog_index.bin` и уезжает
 * в assets APK; сюда он приходит импортом Metro, потому что читается один раз при старте
 * и заводить ради этого четвёртый плагин незачем.
 */
import сыройКаталог from '../../android/models/catalog_meta.json'
import { AssetError } from './thresholds'

export type Класс = {
  display: string
  brand: string
  platformGroup: string | null
}

const строка = (значение: unknown, поле: string): string => {
  if (typeof значение !== 'string' || значение.length === 0) {
    throw new AssetError(`${поле}: ожидалась непустая строка, пришло ${JSON.stringify(значение)}`)
  }
  return значение
}

export function parseCatalog(документ: unknown): Класс[] {
  const поля = документ as Record<string, unknown>
  const сырые = поля.classes
  if (!Array.isArray(сырые) || сырые.length === 0) {
    throw new AssetError('classes: каталог пуст или не является массивом')
  }
  const классы: Класс[] = new Array(сырые.length)
  for (let номер = 0; номер < сырые.length; номер++) {
    const запись = сырые[номер] as Record<string, unknown>
    const группа = запись.platform_group
    классы[номер] = {
      display: строка(запись.display, `classes[${номер}].display`),
      brand: строка(запись.brand, `classes[${номер}].brand`),
      platformGroup: typeof группа === 'string' && группа.length > 0 ? группа : null,
    }
  }
  return классы
}

/**
 * Имя ASCII, потому что константу захватывает воркет `classLabel`: имя захваченной
 * переменной становится ключом `__closure`, а кириллический ключ до worklet-рантайма не
 * доезжает и приезжает `undefined` (см. `workletClosure.test.ts`). Ограничение из Global
 * Constraints распространяется не только на экспорты — на всё, что попадает в замыкание.
 */
const CATALOG: Класс[] = parseCatalog(сыройКаталог)

/** `null` вместо падения: индекс приходит с нативной стороны, и рассинхрон каталога не должен ронять кадр. */
export function classLabel(classIdx: number): Класс | null {
  'worklet'
  if (!Number.isInteger(classIdx) || classIdx < 0 || classIdx >= CATALOG.length) {
    return null
  }
  return CATALOG[classIdx]!
}

/** Совпадают ли бренды двух классов — вход ветви «показать только бренд» из §6.4. */
export function sameBrand(первый: number, второй: number): boolean {
  'worklet'
  const один = classLabel(первый)
  const другой = classLabel(второй)
  if (один == null || другой == null) return false
  return один.brand === другой.brand
}
