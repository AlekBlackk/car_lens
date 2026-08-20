import { VisionCameraProxy, type Frame } from 'react-native-vision-camera'

export type Box = { x: number; y: number; w: number; h: number; conf: number }

export type DetectResult = {
  boxes: Box[]
  frameWidth: number
  frameHeight: number
  nativeMs: number
}

type RawDetectResult = {
  nativeNs: number
  frameWidth: number
  frameHeight: number
  count: number
  boxes: ArrayBuffer
}

const плагинДетекции = VisionCameraProxy.initFrameProcessorPlugin('carvisionDetect', {})

/** Длины проверяются жёстко: рассинхрон Kotlin и TS иначе выглядит как плохая модель. */
export function detect(frame: Frame, maxDet: number): DetectResult {
  'worklet'
  if (плагинДетекции == null) {
    throw new Error('плагин carvisionDetect не зарегистрирован')
  }
  const сырое = плагинДетекции.call(frame, { maxDet }) as unknown as RawDetectResult
  const числа = new Float32Array(сырое.boxes)
  if (числа.length !== сырое.count * 5) {
    throw new Error(`боксы: ${числа.length} чисел против ${сырое.count * 5} по count`)
  }
  const боксы: Box[] = []
  for (let номер = 0; номер < сырое.count; номер++) {
    const сдвиг = номер * 5
    боксы.push({
      x: числа[сдвиг]!,
      y: числа[сдвиг + 1]!,
      w: числа[сдвиг + 2]!,
      h: числа[сдвиг + 3]!,
      conf: числа[сдвиг + 4]!,
    })
  }
  return {
    boxes: боксы,
    frameWidth: сырое.frameWidth,
    frameHeight: сырое.frameHeight,
    nativeMs: сырое.nativeNs / 1e6,
  }
}

export type Match = { classIdx: number; sim: number }

export type EmbedMatchResult = { embedding: Float32Array; matches: Match[] }

/**
 * `results` — по одному элементу на бокс (см. `EmbedMatchResult`). `nativeMs` — время
 * нативной стороны на весь вызов `embedAndMatch` целиком, а не на бокс: сырой мост отдаёт
 * его один раз (`RawEmbedResult.nativeNs`), Task 10 (замер цены моста) читает именно это
 * поле.
 */
export type EmbedMatchCallResult = { results: EmbedMatchResult[]; nativeMs: number }

/**
 * Сырой ответ моста. `emb0`/`emb1` — слоты, по одному на бокс (не больше двух, см.
 * `MAX_CROPS` в `throttle.ts` и `МАКСИМУМ_КРОПОВ` в `CarVisionEmbedPlugin.kt`), поэтому
 * ключи адресуются динамически, а не перечислены полями.
 *
 * `matchCount` — фактическое число совпадений на бокс, а не запрошенный `topK` (находка
 * I1 финального ревью M-1.6): `CatalogIndex.knn` на Kotlin-стороне отдаёт
 * `min(topK, numClasses)`, и на каталоге меньше `topK` буфер `matches` укладывается по
 * этому меньшему числу. Считать по `topK`, как раньше, значило бы читать за пределами
 * того, что нативная сторона реально записала — TS-проверка длины ниже это ловит.
 */
type RawEmbedResult = {
  nativeNs: number
  count: number
  matchCount: number
  matches: ArrayBuffer
  [ключСлота: string]: unknown
}

const плагинЭмбеддинга = VisionCameraProxy.initFrameProcessorPlugin('carvisionEmbedAndMatch', {})

/**
 * Эмбеддинг и top-K для уже отобранных троттлингом боксов (см. `shouldClassify` в
 * `throttle.ts`). Длина `matches` проверяется жёстко по тому же принципу, что в `detect`,
 * но по `matchCount`, а не по запрошенному `topK` — см. комментарий у `RawEmbedResult`.
 */
export function embedAndMatch(frame: Frame, boxes: Box[], pad: number, topK: number): EmbedMatchCallResult {
  'worklet'
  if (плагинЭмбеддинга == null) {
    throw new Error('плагин carvisionEmbedAndMatch не зарегистрирован')
  }
  const плоские: number[] = []
  boxes.forEach((бокс) => {
    плоские.push(бокс.x, бокс.y, бокс.w, бокс.h, бокс.conf)
  })
  const сырое = плагинЭмбеддинга.call(frame, { boxes: плоские, pad, topK }) as unknown as RawEmbedResult
  const matchCount = сырое.matchCount
  // `count` меньше запрошенного `topK` законен (в индексе может быть меньше классов),
  // больше — нет: это означало бы, что нативная сторона считает `topK` не тем, что
  // прислал JS. Та же проверка, что уже есть в `matchEmbedding` ниже.
  if (matchCount > topK) {
    throw new Error(`совпадений на бокс ${matchCount} больше запрошенных topK=${topK}`)
  }
  const числаСовпадений = new Float32Array(сырое.matches)
  if (числаСовпадений.length !== сырое.count * matchCount * 2) {
    throw new Error(
      `совпадения: ${числаСовпадений.length} чисел против ${сырое.count * matchCount * 2} по count*matchCount*2`,
    )
  }
  const результаты: EmbedMatchResult[] = []
  for (let номер = 0; номер < сырое.count; номер++) {
    const слот = сырое[`emb${номер}`] as ArrayBuffer
    // Копия обязательна: слот переиспользуется нативной стороной на следующий вызов
    // плагина (см. комментарий в `CarVisionEmbedPlugin.kt` про контракт времени жизни
    // слотов) — обёртка без копии протухла бы уже к следующему кадру, если результат
    // уедет в состояние трека.
    const эмбеддинг = new Float32Array(слот).slice()
    const matches: Match[] = []
    for (let к = 0; к < matchCount; к++) {
      const сдвиг = (номер * matchCount + к) * 2
      matches.push({
        classIdx: числаСовпадений[сдвиг]!,
        sim: числаСовпадений[сдвиг + 1]!,
      })
    }
    результаты.push({ embedding: эмбеддинг, matches })
  }
  return { results: результаты, nativeMs: сырое.nativeNs / 1e6 }
}

type RawMatchResult = {
  nativeNs: number
  count: number
  matches: ArrayBuffer
}

const плагинСовпадений = VisionCameraProxy.initFrameProcessorPlugin('carvisionMatchEmbedding', {})

/** `nativeMs` — время нативной стороны на весь вызов `matchEmbedding` (Task 10). */
export type MatchCallResult = { matches: Match[]; nativeMs: number }

/**
 * top-K для вектора, который JS уже держит у себя.
 *
 * Тот же поиск, что внутри `embedAndMatch`, но по чужому эмбеддингу: в следующих вехах это
 * будет EMA трека, здесь — сверка, что третий путь границы согласуется с первым.
 *
 * Наружу уходит `embedding.buffer`, а не сам `Float32Array`: в JSI `isArrayBuffer` истинно
 * только для буфера, и окно над ним ушло бы в ветку «обычный объект», приехав на Kotlin
 * как `Map<String, Any>` с числовыми ключами. Копия делается лишь тогда, когда окно не
 * покрывает буфер целиком — иначе на нативную сторону уехали бы лишние числа.
 */
export function matchEmbedding(frame: Frame, embedding: Float32Array, topK: number): MatchCallResult {
  'worklet'
  if (плагинСовпадений == null) {
    throw new Error('плагин carvisionMatchEmbedding не зарегистрирован')
  }
  const целиком = embedding.byteOffset === 0 && embedding.buffer.byteLength === embedding.length * 4
  // `.buffer` типизирован как `ArrayBufferLike` из-за `SharedArrayBuffer`, которого в
  // Hermes нет: сузить до `ArrayBuffer` безопасно, другого варианта тут не бывает.
  const буфер = (целиком ? embedding.buffer : embedding.slice().buffer) as ArrayBuffer
  const сырое = плагинСовпадений.call(frame, { embedding: буфер, topK }) as unknown as RawMatchResult
  const числа = new Float32Array(сырое.matches)
  if (числа.length !== сырое.count * 2) {
    throw new Error(`совпадения: ${числа.length} чисел против ${сырое.count * 2} по count*2`)
  }
  // `count` меньше `topK` законен (в индексе может быть меньше классов), больше — нет:
  // это означало бы, что нативная сторона считает `topK` не тем, что прислал JS.
  if (сырое.count > topK) {
    throw new Error(`совпадений ${сырое.count} больше запрошенных topK=${topK}`)
  }
  const результат: Match[] = []
  for (let к = 0; к < сырое.count; к++) {
    результат.push({ classIdx: числа[к * 2]!, sim: числа[к * 2 + 1]! })
  }
  return { matches: результат, nativeMs: сырое.nativeNs / 1e6 }
}
