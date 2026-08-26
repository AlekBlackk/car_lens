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
  classId: string
  display: string
  brand: string | null
  /**
   * `member_id` участников класса — `<слаг_марки>__<слаг_модели>` из §13.2. Нужны списку
   * выбора: у разнобрендовой группы марки нет, и найти её по марке участника больше нечем.
   */
  members: readonly string[]
}

const строка = (значение: unknown, поле: string): string => {
  if (typeof значение !== 'string' || значение.length === 0) {
    throw new AssetError(`${поле}: ожидалась непустая строка, пришло ${JSON.stringify(значение)}`)
  }
  return значение
}

/**
 * `brand` разнобрендовой группы — `null` по §13.2: «Geely Coolray / Belgee X50» одной маркой
 * не назвать. Поле при этом обязано быть в записи: пропуск означает мету другого формата.
 */
const марка = (значение: unknown, поле: string): string | null =>
  значение === null ? null : строка(значение, поле)

const участники = (значение: unknown, поле: string): readonly string[] => {
  if (!Array.isArray(значение) || значение.length === 0) {
    throw new AssetError(`${поле}: ожидался непустой массив участников класса`)
  }
  return значение.map((участник, номер) => строка(участник, `${поле}[${номер}]`))
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
    классы[номер] = {
      classId: строка(запись.class_id, `classes[${номер}].class_id`),
      display: строка(запись.display, `classes[${номер}].display`),
      brand: марка(запись.brand, `classes[${номер}].brand`),
      members: участники(запись.members, `classes[${номер}].members`),
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

/**
 * Совпадают ли марки двух классов. Правило вынесено из `sameBrand` отдельной функцией,
 * потому что проверить его на настоящем каталоге нечем: классов без марки в нём пока нет,
 * а тест на индексах ловил бы только совпадающие марки.
 */
export function brandsMatch(первый: Класс | null, второй: Класс | null): boolean {
  'worklet'
  if (первый == null || второй == null) return false
  // Два класса без марки — две разные разнобрендовые группы, а не «одна марка»: иначе §6.4
  // показал бы brand-only там, где бренда нет вовсе.
  if (первый.brand == null || второй.brand == null) return false
  return первый.brand === второй.brand
}

/** Совпадают ли бренды двух классов — вход ветви «показать только бренд» из §6.4. */
export function sameBrand(первый: number, второй: number): boolean {
  'worklet'
  return brandsMatch(classLabel(первый), classLabel(второй))
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
