package com.carlens.vision

import android.content.Context
import android.os.SystemClock
import com.mrousavy.camera.core.types.Orientation
import com.mrousavy.camera.frameprocessors.Frame
import com.mrousavy.camera.frameprocessors.FrameProcessorPlugin
import com.mrousavy.camera.frameprocessors.VisionCameraProxy

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
 * Угол, которым препроцессинг выпрямляет кадр, — по гравитации, а не по выводу.
 *
 * Складывается из двух частей, и обе выяснены на устройстве (SM-S918B), потому что до
 * телефона они неразличимы.
 *
 * Первая — знак. `frame.orientation` описывает, **как повёрнут кадр** относительно
 * желаемого вывода, а `FrameConvert`/`Rotation` ждут угол, **на который кадр нужно
 * повернуть**, чтобы его выпрямить: знаки противоположны. Телефон вертикально,
 * `frame.orientation` = `LANDSCAPE_RIGHT` (270°): с углом 270° кадр приезжает лежащим на
 * боку, максимум сырых скорингов графа 0.004-0.058 при пороге 0.35, ни одного кандидата;
 * с углом 90° та же машина находится с уверенностью 0.93.
 *
 * Вторая — угол устройства. `frame.orientation` физический поворот телефона **не
 * отражает**: и в портрете, и в горизонте он остаётся `LANDSCAPE_RIGHT`, поэтому одного
 * его хватает ровно на одну строку таблицы приёмки — ту, где телефон вертикально. Угол
 * устройства приходит из JS: `Camera.onOutputOrientationChanged` при
 * `outputOrientation="device"` слушает акселерометр и меняется даже при выключенном
 * системном автоповороте (проверено: `portrait` → `landscape-left` → `landscape-right`).
 */
internal fun Orientation.уголВыпрямления(уголУстройства: Int): Int =
    (((360 - toDegrees()) + уголУстройства) % 360 + 360) % 360

/**
 * Угол из параметров вызова: 0, если JS его не прислал.
 *
 * Углов два и путать их нельзя. `deviceDegrees` — физический поворот телефона, им
 * выпрямляется кадр для моделей. `previewDegrees` — поворот, в котором показывает
 * превью, в его системе рисуются рамки. При выключенном автоповороте они расходятся:
 * телефон в горизонте, а превью осталось портретным.
 */
internal fun уголИзПараметров(params: Map<String, Any>?, ключ: String): Int {
    val сырой = (params?.get(ключ) as? Number)?.toInt() ?: 0
    require(сырой % 90 == 0) { "угол $ключ = $сырой не кратен 90" }
    return ((сырой % 360) + 360) % 360
}

/**
 * Детекция: считает по гравитации, отдаёт в координатах экрана.
 *
 * Две системы координат, и обе нужны. Детектор обязан видеть машину «на колёсах», иначе
 * он её не находит вовсе — значит кадр выпрямляется по физическому углу телефона. Рисует
 * рамки превью, а оно живёт в ориентации вывода, которая при заблокированном автоповороте
 * с физическим углом не совпадает — значит наружу боксы уходят довёрнутыми в неё.
 * `Rotation.rotateBox` написан ровно под этот разрыв ещё в M-1.5.
 */
class CarVisionDetectPlugin(
    private val proxy: VisionCameraProxy,
    private val context: Context,
) : FrameProcessorPlugin() {

    private val массивы = sharedArrayPool(proxy)

    override fun callback(frame: Frame, params: Map<String, Any>?): Any {
        val начало = SystemClock.elapsedRealtimeNanos()
        require(!frame.isMirrored) { "фронтальная камера не поддерживается" }
        val maxDet = (params?.get("maxDet") as? Number)?.toInt() ?: 4
        val уголГравитации = frame.orientation.уголВыпрямления(уголИзПараметров(params, "deviceDegrees"))
        val уголЭкрана = frame.orientation.уголВыпрямления(уголИзПараметров(params, "previewDegrees"))

        return CarVisionHolder.подЗамком(context) { зрение ->
            val боксыГравитации = зрение.detect(frame.image, уголГравитации)
            val кадрГравитации = Rotation.upright(frame.image.width, frame.image.height, уголГравитации)
            val кадрЭкрана = Rotation.upright(frame.image.width, frame.image.height, уголЭкрана)
            val доЭкрана = Rotation.betweenUpright(уголГравитации, уголЭкрана)
            val боксыЭкрана = боксыГравитации.map { Rotation.rotateBox(it, кадрГравитации, доЭкрана) }
            val отобранные = боксыЭкрана.sortedByDescending { it.conf }.take(maxDet)

            val (буфер, числа) = floatSharedArray(массивы, отобранные.size * 5)
            отобранные.forEach { бокс ->
                числа.put(бокс.x); числа.put(бокс.y)
                числа.put(бокс.w); числа.put(бокс.h)
                числа.put(бокс.conf)
            }
            ответМоста(
                SystemClock.elapsedRealtimeNanos() - начало,
                отобранные.size,
                mapOf(
                    "frameWidth" to кадрЭкрана.width,
                    "frameHeight" to кадрЭкрана.height,
                    "boxes" to буфер,
                ),
            )
        }
    }
}
