package com.carlens.vision

import android.content.Context
import android.os.SystemClock
import com.mrousavy.camera.core.types.Orientation
import com.mrousavy.camera.frameprocessors.Frame
import com.mrousavy.camera.frameprocessors.FrameProcessorPlugin
import com.mrousavy.camera.frameprocessors.SharedArray
import com.mrousavy.camera.frameprocessors.VisionCameraProxy
import java.nio.ByteOrder

/**
 * Угол кадра в градусах поворота по часовой стрелке.
 *
 * В `com.mrousavy.camera.core.types.Orientation` (VisionCamera 4.6.4) такого метода нет —
 * есть только `toSurfaceRotation()` (возвращает `Surface.ROTATION_0/90/180/270`, то есть
 * enum-константы Android, а не градусы) и `reversed()`. Проверено по исходнику
 * `android/src/main/java/com/mrousavy/camera/core/types/Orientation.kt`.
 *
 * Маппинг ниже — инверсия `Orientation.Companion.fromRotationDegrees`: там `0°→PORTRAIT`,
 * `45-135°→LANDSCAPE_LEFT`, `135-225°→PORTRAIT_UPSIDE_DOWN`, `225-315°→LANDSCAPE_RIGHT`,
 * то есть обратно — `PORTRAIT↔0°`, `LANDSCAPE_LEFT↔90°`, `PORTRAIT_UPSIDE_DOWN↔180°`,
 * `LANDSCAPE_RIGHT↔270°`.
 */
internal fun Orientation.toDegrees(): Int =
    when (this) {
        Orientation.PORTRAIT -> 0
        Orientation.LANDSCAPE_LEFT -> 90
        Orientation.PORTRAIT_UPSIDE_DOWN -> 180
        Orientation.LANDSCAPE_RIGHT -> 270
    }

/**
 * Детекция: рамки в координатах экрана.
 *
 * Угол берётся из `frame.orientation` — по документации VisionCamera 4.6.4 это
 * «ориентация кадра относительно желаемой выходной ориентации», то есть уже готовая
 * разница. Считать её из угла экрана, присланного из JS, значило бы пересчитывать
 * посчитанное, да ещё и с задержкой JS-потока.
 */
class CarVisionDetectPlugin(
    private val proxy: VisionCameraProxy,
    private val context: Context,
) : FrameProcessorPlugin() {

    override fun callback(frame: Frame, params: Map<String, Any>?): Any {
        val начало = SystemClock.elapsedRealtimeNanos()
        require(!frame.isMirrored) { "фронтальная камера не поддерживается" }
        val maxDet = (params?.get("maxDet") as? Number)?.toInt() ?: 4
        val уголКадра = frame.orientation.toDegrees()

        return CarVisionHolder.подЗамком(context) { зрение ->
            // Второго поворота здесь нет намеренно. `frame.orientation` — уже разница до
            // желаемой выходной ориентации, поэтому `detect` с этим углом выпрямляет кадр
            // ровно в неё, и боксы выходят сразу в координатах вывода. Довернуть их ещё
            // раз (через `betweenUpright`/`rotateBox`) означало бы применить поворот
            // дважды — тот самый баг, что чинили в `:demo` коммитом 64c2edd.
            val боксыЭкрана = зрение.detect(frame.image, уголКадра)
            val кадрЭкрана = Rotation.upright(frame.image.width, frame.image.height, уголКадра)
            val отобранные = боксыЭкрана.sortedByDescending { it.conf }.take(maxDet)

            val буфер = SharedArray(proxy, отобранные.size * 5 * 4)
            val числа = буфер.byteBuffer.order(ByteOrder.nativeOrder()).asFloatBuffer()
            отобранные.forEach { бокс ->
                числа.put(бокс.x); числа.put(бокс.y)
                числа.put(бокс.w); числа.put(бокс.h)
                числа.put(бокс.conf)
            }
            mapOf(
                "nativeNs" to (SystemClock.elapsedRealtimeNanos() - начало).toDouble(),
                "frameWidth" to кадрЭкрана.width,
                "frameHeight" to кадрЭкрана.height,
                "count" to отобранные.size,
                "boxes" to буфер,
            )
        }
    }
}
