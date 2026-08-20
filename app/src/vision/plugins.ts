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
