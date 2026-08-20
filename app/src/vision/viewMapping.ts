import type { Box } from './plugins'

export type ViewRect = { left: number; top: number; width: number; height: number }

/**
 * Бокс кадра в координаты `View`.
 *
 * Превью показывает кадр в режиме FILL_CENTER: масштаб берётся по большей стороне, а не
 * по каждой оси порознь — масштабирование по осям растянуло бы рамку относительно
 * картинки. Та же арифметика, что в `OverlayView.kt` модуля `:demo`, проверенная на
 * устройстве; здесь она вторая реализация той же формулы намеренно — общего рантайма у
 * Kotlin и TS нет, поэтому обе стороны сверяются глазами на одном кадре при приёмке.
 */
export function mapBoxToView(
  box: Box,
  frameWidth: number,
  frameHeight: number,
  viewWidth: number,
  viewHeight: number,
): ViewRect {
  if (frameWidth <= 0 || frameHeight <= 0) {
    throw new Error(`пустой кадр ${frameWidth}x${frameHeight}`)
  }
  const масштаб = Math.max(viewWidth / frameWidth, viewHeight / frameHeight)
  const сдвигX = (viewWidth - frameWidth * масштаб) / 2
  const сдвигY = (viewHeight - frameHeight * масштаб) / 2
  return {
    left: box.x * масштаб + сдвигX,
    top: box.y * масштаб + сдвигY,
    width: box.w * масштаб,
    height: box.h * масштаб,
  }
}
