package com.carlens.vision

import kotlin.math.max
import kotlin.math.min

data class Box(val x: Float, val y: Float, val w: Float, val h: Float, val conf: Float)

/** Размер выпрямленного кадра: ширина и высота после применения `rotationDegrees`. */
data class UprightSize(val width: Int, val height: Int)

/**
 * Отображение выпрямленного кадра в квадратный тензор детектора с сохранением
 * пропорций и серыми полями (§6.2 дизайна).
 */
data class Letterbox(val scale: Float, val padX: Float, val padY: Float, val target: Int) {

    fun uprightToTensor(box: Box): Box =
        Box(box.x * scale + padX, box.y * scale + padY, box.w * scale, box.h * scale, box.conf)

    fun tensorToUpright(box: Box): Box =
        Box((box.x - padX) / scale, (box.y - padY) / scale, box.w / scale, box.h / scale, box.conf)

    companion object {
        fun forFrame(upright: UprightSize, target: Int): Letterbox {
            // Нулевая сторона дала бы бесконечный `scale`, а дальше NaN в координатах:
            // отказ здесь, потому что рамка из NaN выглядит правдоподобной, а не битой.
            require(upright.width > 0 && upright.height > 0) {
                "пустой кадр ${upright.width}x${upright.height}"
            }
            require(target > 0) { "сторона тензора $target" }
            val масштаб = min(target.toFloat() / upright.width, target.toFloat() / upright.height)
            return Letterbox(
                scale = масштаб,
                padX = (target - upright.width * масштаб) / 2f,
                padY = (target - upright.height * масштаб) / 2f,
                target = target,
            )
        }
    }
}

object Rotation {

    fun upright(width: Int, height: Int, degrees: Int): UprightSize =
        when (нормализовать(degrees)) {
            90, 270 -> UprightSize(height, width)
            else -> UprightSize(width, height)
        }

    /**
     * Точка выпрямленного кадра в координатах буфера сенсора.
     *
     * Поворот применяется ровно здесь и больше нигде: наружу плагин отдаёт уже
     * выпрямленную геометрию, потому что трекер на M-1.6 будет хранить боксы между
     * кадрами, и поворот, живущий снаружи, каждый потребитель применит по-своему.
     */
    fun toSensor(xUpright: Float, yUpright: Float, upright: UprightSize, degrees: Int): FloatArray =
        when (нормализовать(degrees)) {
            0 -> floatArrayOf(xUpright, yUpright)
            90 -> floatArrayOf(yUpright, upright.width - 1f - xUpright)
            180 -> floatArrayOf(upright.width - 1f - xUpright, upright.height - 1f - yUpright)
            else -> floatArrayOf(upright.height - 1f - yUpright, xUpright)
        }

    private fun нормализовать(degrees: Int): Int {
        val угол = ((degrees % 360) + 360) % 360
        require(угол % 90 == 0) { "поворот $degrees не кратен 90" }
        return угол
    }
}

object Boxes {

    /** Расширение на `pad` с каждой стороны с обрезкой по кадру — как `pipeline/crop.py:84`. */
    fun pad(box: Box, frameW: Int, frameH: Int, pad: Float): Box {
        val дельтаX = box.w * pad
        val дельтаY = box.h * pad
        val x0 = max(0f, box.x - дельтаX)
        val y0 = max(0f, box.y - дельтаY)
        val x1 = min(frameW.toFloat(), box.x + box.w + дельтаX)
        val y1 = min(frameH.toFloat(), box.y + box.h + дельтаY)
        return Box(x0, y0, x1 - x0, y1 - y0, box.conf)
    }

    fun iou(a: Box, b: Box): Float {
        val x0 = max(a.x, b.x)
        val y0 = max(a.y, b.y)
        val x1 = min(a.x + a.w, b.x + b.w)
        val y1 = min(a.y + a.h, b.y + b.h)
        val пересечение = max(0f, x1 - x0) * max(0f, y1 - y0)
        val объединение = a.w * a.h + b.w * b.h - пересечение
        return if (объединение <= 0f) 0f else пересечение / объединение
    }

    fun nms(boxes: List<Box>, iouThreshold: Float, maxDet: Int): List<Box> {
        // Пустой вход — законный случай (кадр без машин), а вот `maxDet = 0` или порог
        // вне [0, 1] означают битый контракт детектора: молча вернуть пусто нельзя.
        require(maxDet >= 1) { "maxDet $maxDet: ответ не может быть пустым по построению" }
        require(iouThreshold in 0f..1f) { "порог IoU $iouThreshold вне 0..1" }
        val порядок = boxes.sortedByDescending { it.conf }
        val оставленные = ArrayList<Box>(minOf(maxDet, порядок.size))
        for (кандидат in порядок) {
            if (оставленные.size >= maxDet) break
            if (оставленные.none { iou(it, кандидат) > iouThreshold }) оставленные.add(кандидат)
        }
        return оставленные
    }

    /**
     * Развернуть сырой бокс графа в пиксели тензора.
     *
     * Формат и единицы приходят из контракта детектора (§7.1 дизайна), а не
     * предполагаются: `[1, 84, N]` выглядит одинаково для долей и пикселей.
     */
    fun fromRaw(
        cx: Float,
        cy: Float,
        a: Float,
        b: Float,
        format: String,
        unitsScale: Float,
        conf: Float,
    ): Box = when (format) {
        "xywh" -> Box(
            x = (cx - a / 2f) * unitsScale,
            y = (cy - b / 2f) * unitsScale,
            w = a * unitsScale,
            h = b * unitsScale,
            conf = conf,
        )
        "xyxy" -> Box(
            x = cx * unitsScale,
            y = cy * unitsScale,
            w = (a - cx) * unitsScale,
            h = (b - cy) * unitsScale,
            conf = conf,
        )
        else -> throw IllegalArgumentException("неизвестный box_format \"$format\"")
    }
}
