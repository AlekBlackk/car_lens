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

/*
 * Латиница в именах трёх хэндлов — требование `react-native-worklets-core` 1.5.0, а не
 * стиль: имя модульной переменной становится ключом `__closure` воркета, и ключи с
 * не-ASCII символами не доезжают до worklet-рантайма — на устройстве захват приходит
 * `undefined`. Симптом был именно такой: плагины регистрировались
 * (`FrameProcessorPluginRegistry: "carvisionDetect" found!` в logcat), а воркет падал на
 * `плагин carvisionDetect не зарегистрирован`.
 */
const detectPlugin = VisionCameraProxy.initFrameProcessorPlugin('carvisionDetect', {})

/** Длины проверяются жёстко: рассинхрон Kotlin и TS иначе выглядит как плохая модель. */
export function detect(
  frame: Frame,
  maxDet: number,
  deviceDegrees: number,
  previewDegrees: number,
): DetectResult {
  'worklet'
  if (detectPlugin == null) {
    throw new Error('плагин carvisionDetect не зарегистрирован')
  }
  // `deviceDegrees` — физический угол телефона. Без него кадр выпрямляется только
  // относительно вывода, и детектор находит машину ровно в одном положении устройства:
  // `frame.orientation` поворот телефона не отражает (проверено на SM-S918B).
  const сырое = detectPlugin.call(
    frame, { maxDet, deviceDegrees, previewDegrees },
  ) as unknown as RawDetectResult
  if (сырое.boxes == null) {
    throw new Error(`${BRIDGE_SLOT_ERROR}: detect не отдал boxes при count=${сырое.count}`)
  }
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

/**
 * Префикс ошибки контракта слотов. Не класс: подкласс `Error`, развёрнутый babel,
 * worklet-рантайм конструктором звать отказывается (см. `EMBEDDING_DIM_ERROR` в `ema.ts`).
 * Имя латиницей — константа захватывается воркетом.
 */
export const BRIDGE_SLOT_ERROR = 'слот эмбеддинга'

export type Match = { classIdx: number; sim: number }

export type EmbedMatchResult = { embedding: Float32Array; matches: Match[] }

/**
 * `results` — по одному элементу на бокс (см. `EmbedMatchResult`). `nativeMs` — время
 * нативной стороны на весь вызов `embedAndMatch` целиком, а не на бокс: сырой мост отдаёт
 * его один раз (`RawEmbedResult.nativeNs`), Task 10 (замер цены моста) читает именно это
 * поле.
 */
export type EmbedMatchCallResult = {
  results: EmbedMatchResult[]
  nativeMs: number
  /**
   * Разные ли буферы у слотов этого вызова. Признак прямой — сравнение `ArrayBuffer` по
   * ссылке, — и потому не зависит от содержимого кадра: сравнивать вместо этого сами
   * эмбеддинги нельзя, на однородной сцене два разных кропа дают численно одинаковый
   * вектор законно. Читается самопроверкой контракта §4.8 (`slotContract.ts`).
   */
  slotsDistinct: boolean
}

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

const embedPlugin = VisionCameraProxy.initFrameProcessorPlugin('carvisionEmbedAndMatch', {})

/**
 * Эмбеддинг и top-K для уже отобранных троттлингом боксов (см. `pickForClassify` в
 * `throttle.ts`). Длина `matches` проверяется жёстко по тому же принципу, что в `detect`,
 * но по `matchCount`, а не по запрошенному `topK` — см. комментарий у `RawEmbedResult`.
 */
export function embedAndMatch(
  frame: Frame,
  boxes: Box[],
  pad: number,
  topK: number,
  deviceDegrees: number,
  previewDegrees: number,
): EmbedMatchCallResult {
  'worklet'
  if (embedPlugin == null) {
    throw new Error('плагин carvisionEmbedAndMatch не зарегистрирован')
  }
  const плоские: number[] = []
  boxes.forEach((бокс) => {
    плоские.push(бокс.x, бокс.y, бокс.w, бокс.h, бокс.conf)
  })
  // Тот же угол устройства, что и у `detect`: кроп обязан жить в той же системе
  // координат, что и боксы, иначе классификатор режет мимо машины и делает это тихо.
  const сырое = embedPlugin.call(
    frame, { boxes: плоские, pad, topK, deviceDegrees, previewDegrees },
  ) as unknown as RawEmbedResult
  const matchCount = сырое.matchCount
  // `count` меньше запрошенного `topK` законен (в индексе может быть меньше классов),
  // больше — нет: это означало бы, что нативная сторона считает `topK` не тем, что
  // прислал JS. Та же проверка, что уже есть в `matchEmbedding` ниже.
  if (matchCount > topK) {
    throw new Error(`совпадений на бокс ${matchCount} больше запрошенных topK=${topK}`)
  }
  if (сырое.matches == null) {
    throw new Error(
      `${BRIDGE_SLOT_ERROR}: embedAndMatch не отдал matches при count=${сырое.count}`,
    )
  }
  const числаСовпадений = new Float32Array(сырое.matches)
  if (числаСовпадений.length !== сырое.count * matchCount * 2) {
    throw new Error(
      `совпадения: ${числаСовпадений.length} чисел против ${сырое.count * matchCount * 2} по count*matchCount*2`,
    )
  }
  const результаты: EmbedMatchResult[] = []
  const слоты: ArrayBuffer[] = []
  for (let номер = 0; номер < сырое.count; номер++) {
    const слот = сырое[`emb${номер}`] as ArrayBuffer | undefined
    // Слот берётся по имени, а не из массива, и отсутствующее имя иначе доехало бы до
    // `new Float32Array(undefined)` — на устройстве это «Cannot convert undefined value to
    // object», сообщение, по которому не догадаться ни про мост, ни про номер слота.
    if (слот == null) {
      throw new Error(
        `${BRIDGE_SLOT_ERROR}: emb${номер} не пришёл при count=${сырое.count}`,
      )
    }
    слоты.push(слот)
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
  let slotsDistinct = true
  for (let первый = 0; первый < слоты.length; первый++) {
    for (let второй = первый + 1; второй < слоты.length; второй++) {
      if (слоты[первый] === слоты[второй]) {
        slotsDistinct = false
      }
    }
  }
  return { results: результаты, nativeMs: сырое.nativeNs / 1e6, slotsDistinct }
}

type RawMatchResult = {
  nativeNs: number
  count: number
  matches: ArrayBuffer
}

const matchPlugin = VisionCameraProxy.initFrameProcessorPlugin('carvisionMatchEmbedding', {})

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
  if (matchPlugin == null) {
    throw new Error('плагин carvisionMatchEmbedding не зарегистрирован')
  }
  const целиком = embedding.byteOffset === 0 && embedding.buffer.byteLength === embedding.length * 4
  // `.buffer` типизирован как `ArrayBufferLike` из-за `SharedArrayBuffer`, которого в
  // Hermes нет: сузить до `ArrayBuffer` безопасно, другого варианта тут не бывает.
  const буфер = (целиком ? embedding.buffer : embedding.slice().buffer) as ArrayBuffer
  const сырое = matchPlugin.call(frame, { embedding: буфер, topK }) as unknown as RawMatchResult
  if (сырое.matches == null) {
    throw new Error(`${BRIDGE_SLOT_ERROR}: matchEmbedding не отдал matches`)
  }
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
