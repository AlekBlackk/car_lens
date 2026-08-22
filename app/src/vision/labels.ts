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

export type ИнфоКаталога = {
  formatVersion: number
  generatedAt: string
  weightsSha: string
  классов: number
}

const целое = (значение: unknown, поле: string): number => {
  if (typeof значение !== 'number' || !Number.isInteger(значение)) {
    throw new AssetError(`${поле}: ожидалось целое, пришло ${JSON.stringify(значение)}`)
  }
  return значение
}

export function parseCatalogInfo(документ: unknown): ИнфоКаталога {
  const поля = документ as Record<string, unknown>
  const классы = поля.classes
  // Пустой массив здесь допустим, в отличие от `parseCatalog`: версия каталога читается и у
  // пустого. Модуль это переживает лишь потому, что `CATALOG` объявлен выше и на пустом
  // каталоге падает первым — переставлять два объявления нельзя.
  if (!Array.isArray(классы)) {
    throw new AssetError('classes: каталог не является массивом')
  }
  return {
    formatVersion: целое(поля.format_version, 'format_version'),
    generatedAt: строка(поля.generated_at, 'generated_at'),
    weightsSha: строка(поля.model_weights_sha256, 'model_weights_sha256'),
    классов: классы.length,
  }
}

const CATALOG_INFO: ИнфоКаталога = parseCatalogInfo(сыройКаталог)

/** Общий объект, а не копия: правка его полей отравила бы digest всем последующим вызовам. */
export function catalogInfo(): ИнфоКаталога {
  return CATALOG_INFO
}

/**
 * Метка версии каталога для записей истории и очереди отзывов. Три поля, а не одно:
 * `format_version` меняется при смене формата, sha весов — при переобучении, число
 * классов — при расширении каталога (§10 спеки, M6: 150 → 600 → 1500). Любое из трёх
 * меняет смысл `classIdx`, и любого одного было бы мало.
 */
export function catalogDigest(): string {
  return `${CATALOG_INFO.formatVersion}:${CATALOG_INFO.weightsSha.slice(0, 12)}:${CATALOG_INFO.классов}`
}

/**
 * Весь каталог для списка выбора в «это неверно». Без `'worklet'`: нужен только JS-стороне.
 * Отдаётся живой `CATALOG`, из которого воркет `classLabel` читает подписи: `readonly`
 * запрещает мутацию только на уровне типов, поэтому не сортировать и не переставлять на
 * месте — сортировку списка делать в копию.
 */
export function allClasses(): readonly Класс[] {
  return CATALOG
}
