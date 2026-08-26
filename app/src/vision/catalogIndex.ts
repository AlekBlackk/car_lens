/**
 * Читалка бинарного индекса эталонов `catalog_index.bin` (§13.2).
 *
 * На устройстве индекс читает нативная сторона (`CatalogIndex.kt`, M-1.5) — пиксели и
 * эмбеддинги границу JS не пересекают (§13.3). Эта реализация существует как
 * исполняемый эталон формата: она гоняется в CI на каждом коммите и служит основой
 * оффлайн-инструментов.
 *
 * Хеш-функция передаётся параметром, а не импортируется из `node:crypto`: модуль лежит
 * в дереве приложения, а в React Native такого модуля нет.
 */

const MAGIC = 'CLIX'
const FORMAT_VERSION = 1
const HEADER_SIZE = 64
const DTYPE_FLOAT16 = 0
const DTYPE_FLOAT32 = 1

export class IndexFormatError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'IndexFormatError'
  }
}

export interface IndexHeader {
  formatVersion: number
  embeddingDim: number
  numClasses: number
  centroidsPerClass: number
  dtype: number
  normalized: boolean
  metaSha256: string
}

/**
 * Запись описывает класс, а не модель (§13.2): состав группы лежит в `members`, а `brand`
 * пуст у разнобрендовой группы. Поля `idx` нет — индексом служит позиция в `classes`.
 */
export interface CatalogClass {
  class_id: string
  members: string[]
  brand: string | null
  display: string
}

export interface CatalogIndex {
  header: IndexHeader
  classes: CatalogClass[]
  centroid(classIdx: number, k: number): Float32Array
  match(embedding: Float32Array): { classIdx: number; sim: number }
}

const decodeHeader = (bin: Uint8Array): IndexHeader => {
  if (bin.length < HEADER_SIZE) {
    throw new IndexFormatError(`файл короче заголовка (${bin.length} < ${HEADER_SIZE})`)
  }
  const view = new DataView(bin.buffer, bin.byteOffset, bin.byteLength)
  const magic = String.fromCharCode(bin[0]!, bin[1]!, bin[2]!, bin[3]!)
  if (magic !== MAGIC) {
    throw new IndexFormatError(`magic "${magic}" вместо "${MAGIC}"`)
  }
  const formatVersion = view.getUint16(4, true)
  if (formatVersion !== FORMAT_VERSION) {
    throw new IndexFormatError(`format_version ${formatVersion}, поддерживается ${FORMAT_VERSION}`)
  }
  const dtype = view.getUint8(16)
  if (dtype === DTYPE_FLOAT16) {
    throw new IndexFormatError('float16 не поддерживается: §13.2 предписывает float32 в v1')
  }
  if (dtype !== DTYPE_FLOAT32) {
    throw new IndexFormatError(`неизвестный dtype ${dtype}`)
  }
  if (view.getUint16(18, true) !== 0 || bin.slice(52, 64).some((byte) => byte !== 0)) {
    throw new IndexFormatError('reserved не нулевой — файл писала другая версия формата')
  }
  const metaSha256 = Array.from(bin.slice(20, 52), (byte) =>
    byte.toString(16).padStart(2, '0'),
  ).join('')
  return {
    formatVersion,
    embeddingDim: view.getUint16(6, true),
    numClasses: view.getUint32(8, true),
    centroidsPerClass: view.getUint32(12, true),
    dtype,
    normalized: view.getUint8(17) === 1,
    metaSha256,
  }
}

export function loadIndex(
  bin: Uint8Array,
  meta: Uint8Array,
  sha256: (data: Uint8Array) => string,
): CatalogIndex {
  const header = decodeHeader(bin)
  const values = header.numClasses * header.centroidsPerClass * header.embeddingDim
  const expected = HEADER_SIZE + values * 4
  if (bin.length !== expected) {
    throw new IndexFormatError(`длина ${bin.length} против ${expected} по заголовку`)
  }
  const actual = sha256(meta)
  if (actual !== header.metaSha256) {
    throw new IndexFormatError(`sha256 меты ${actual} против ${header.metaSha256} в заголовке`)
  }
  // Записи меты здесь не проверяются, и это осознанно: sha256 из заголовка связывает мету с
  // бинарником, а соответствие §13.2 сторожат две настоящие читалки — `labels.ts` (роняет
  // старт бандла с указанием поля) и `CatalogIndex.kt` (`IndexFormatException`). Третьему
  // разборщику формата места нет: он разошёлся бы с ними. Цена приведения типа — мета
  // старого формата даёт здесь `class_id === undefined` в типе `string`.
  const document = JSON.parse(new TextDecoder().decode(meta)) as { classes?: CatalogClass[] }
  const classes = document.classes ?? []
  if (classes.length !== header.numClasses) {
    throw new IndexFormatError(
      `classes меты: ${classes.length} против ${header.numClasses} в заголовке`,
    )
  }
  // Заголовок 64 байта кратен четырём, поэтому тело читается одной вьюхой без копии.
  // Копия делается только если сам буфер пришёл со смещением не кратным четырём:
  // Float32Array такого не принимает, а падать RangeError из недр рантайма незачем.
  const start = bin.byteOffset + HEADER_SIZE
  const body =
    start % 4 === 0
      ? new Float32Array(bin.buffer, start, values)
      : new Float32Array(bin.slice(HEADER_SIZE).buffer, 0, values)

  const centroid = (classIdx: number, k: number): Float32Array => {
    if (!Number.isInteger(classIdx) || classIdx < 0 || classIdx >= header.numClasses) {
      throw new IndexFormatError(`класс ${classIdx} вне 0..${header.numClasses - 1}`)
    }
    if (!Number.isInteger(k) || k < 0 || k >= header.centroidsPerClass) {
      throw new IndexFormatError(`центроид ${k} вне 0..${header.centroidsPerClass - 1}`)
    }
    const offset = (classIdx * header.centroidsPerClass + k) * header.embeddingDim
    return body.subarray(offset, offset + header.embeddingDim)
  }

  const match = (embedding: Float32Array): { classIdx: number; sim: number } => {
    if (embedding.length !== header.embeddingDim) {
      throw new IndexFormatError(
        `размерность ${embedding.length} против ${header.embeddingDim} в индексе`,
      )
    }
    let bestClass = -1
    let bestSim = -Infinity
    for (let c = 0; c < header.numClasses; c += 1) {
      for (let k = 0; k < header.centroidsPerClass; k += 1) {
        const vector = centroid(c, k)
        let dot = 0
        for (let d = 0; d < header.embeddingDim; d += 1) {
          dot += vector[d]! * embedding[d]!
        }
        if (dot > bestSim) {
          bestSim = dot
          bestClass = c
        }
      }
    }
    return { classIdx: bestClass, sim: bestSim }
  }

  return { header, classes, centroid, match }
}
