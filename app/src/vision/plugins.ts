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
 * Сырой ответ моста. `emb0`/`emb1` — слоты, по одному на бокс (не больше двух, см.
 * `MAX_CROPS` в `throttle.ts` и `МАКСИМУМ_КРОПОВ` в `CarVisionEmbedPlugin.kt`), поэтому
 * ключи адресуются динамически, а не перечислены полями.
 */
type RawEmbedResult = {
  nativeNs: number
  count: number
  matches: ArrayBuffer
  [ключСлота: string]: unknown
}

const плагинЭмбеддинга = VisionCameraProxy.initFrameProcessorPlugin('carvisionEmbedAndMatch', {})

/**
 * Эмбеддинг и top-K для уже отобранных троттлингом боксов (см. `shouldClassify` в
 * `throttle.ts`). Длина `matches` проверяется жёстко по тому же принципу, что в `detect`.
 */
export function embedAndMatch(frame: Frame, boxes: Box[], pad: number, topK: number): EmbedMatchResult[] {
  'worklet'
  if (плагинЭмбеддинга == null) {
    throw new Error('плагин carvisionEmbedAndMatch не зарегистрирован')
  }
  const плоские: number[] = []
  boxes.forEach((бокс) => {
    плоские.push(бокс.x, бокс.y, бокс.w, бокс.h, бокс.conf)
  })
  const сырое = плагинЭмбеддинга.call(frame, { boxes: плоские, pad, topK }) as unknown as RawEmbedResult
  const числаСовпадений = new Float32Array(сырое.matches)
  if (числаСовпадений.length !== сырое.count * topK * 2) {
    throw new Error(
      `совпадения: ${числаСовпадений.length} чисел против ${сырое.count * topK * 2} по count*topK*2`,
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
    for (let к = 0; к < topK; к++) {
      const сдвиг = (номер * topK + к) * 2
      matches.push({
        classIdx: числаСовпадений[сдвиг]!,
        sim: числаСовпадений[сдвиг + 1]!,
      })
    }
    результаты.push({ embedding: эмбеддинг, matches })
  }
  return результаты
}
