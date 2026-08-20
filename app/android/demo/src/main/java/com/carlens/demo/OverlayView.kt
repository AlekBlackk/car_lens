package com.carlens.demo

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import com.carlens.vision.Box
import com.carlens.vision.DelegateMode
import kotlin.math.min

/** Времена стадий одного кадра, миллисекунды. */
data class Времена(val детектор: Float, val классификатор: Float)

/**
 * Рамки, подписи и строка времён поверх превью.
 *
 * Координаты приходят в пикселях выпрямленного кадра — того самого, который видел
 * плагин, — и здесь только масштабируются к размеру `View`. Пересчитывать геометрию
 * второй раз нельзя: поворот уже применён внутри `:vision`, и второе применение даёт
 * правдоподобную рамку не на машине (§5.1 дизайна).
 *
 * На диск не пишется ничего, в лог — только числа и имена классов (§11 дизайна).
 */
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0,
) : View(context, attrs, defStyleAttr) {

    private val рамка = Paint().apply {
        color = Color.parseColor("#FF4CD964")
        style = Paint.Style.STROKE
        strokeWidth = 6f
        isAntiAlias = true
    }
    private val фонПодписи = Paint().apply {
        color = Color.parseColor("#B0000000")
        style = Paint.Style.FILL
    }
    private val текст = Paint().apply {
        color = Color.WHITE
        textSize = 42f
        isAntiAlias = true
    }
    private var боксы: List<Box> = emptyList()
    private var подписи: List<String> = emptyList()
    private var кадрШирина = 0
    private var кадрВысота = 0
    /**
     * Куда отдавать строку состояния.
     *
     * `Canvas` здесь рисует только рамки и подписи — то, что привязано к координатам
     * кадра. Строка времён к кадру не привязана, и обычному тексту место в `TextView`:
     * перенос длинной строки, фон и отступы там уже есть, а вручную их пришлось бы
     * считать в `onDraw`.
     */
    var приСостоянии: (String) -> Unit = {}

    // p50 за последние 30 кадров: мгновенное значение прыгает так, что прочитать его с
    // экрана нельзя, а среднее прячет редкие длинные кадры.
    private val окноДетектора = ArrayDeque<Float>()
    private val окноКлассификатора = ArrayDeque<Float>()

    fun показать(
        боксы: List<Box>,
        подписи: List<String>,
        кадрШирина: Int,
        кадрВысота: Int,
        времена: Времена,
        делегат: DelegateMode,
    ) {
        this.боксы = боксы
        this.подписи = подписи
        this.кадрШирина = кадрШирина
        this.кадрВысота = кадрВысота
        добавить(окноДетектора, времена.детектор)
        if (времена.классификатор > 0f) добавить(окноКлассификатора, времена.классификатор)
        val классификатор =
            if (окноКлассификатора.isEmpty()) "—" else "%.0f".format(перцентиль(окноКлассификатора))
        приСостоянии(
            "детектор ${"%.0f".format(перцентиль(окноДетектора))} мс · " +
                "классификатор $классификатор мс · p50/30 кадров · $делегат"
        )
        invalidate()
    }

    private fun добавить(окно: ArrayDeque<Float>, значение: Float) {
        окно.addLast(значение)
        while (окно.size > ОКНО) окно.removeFirst()
    }

    private fun перцентиль(окно: ArrayDeque<Float>): Float {
        val отсортированные = окно.sorted()
        return отсортированные[отсортированные.size / 2]
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (кадрШирина <= 0 || кадрВысота <= 0) return
        // Превью держит пропорции кадра и обрезает лишнее (`PreviewView` в режиме
        // FILL_CENTER), поэтому масштаб берётся по большей стороне, а не по каждой оси:
        // масштабирование по осям порознь растянуло бы рамку относительно картинки.
        val масштаб = maxOf(width.toFloat() / кадрШирина, height.toFloat() / кадрВысота)
        val сдвигX = (width - кадрШирина * масштаб) / 2f
        val сдвигY = (height - кадрВысота * масштаб) / 2f
        боксы.forEachIndexed { номер, бокс ->
            val x = бокс.x * масштаб + сдвигX
            val y = бокс.y * масштаб + сдвигY
            val ширина = бокс.w * масштаб
            val высота = бокс.h * масштаб
            canvas.drawRect(x, y, x + ширина, y + высота, рамка)
            val подпись = подписи.getOrNull(номер) ?: return@forEachIndexed
            val ширинаТекста = текст.measureText(подпись)
            val верхПодписи = min(y - 56f, (height - 56f))
            canvas.drawRect(
                x, верхПодписи, x + ширинаТекста + 24f, верхПодписи + 56f, фонПодписи,
            )
            canvas.drawText(подпись, x + 12f, верхПодписи + 42f, текст)
        }
    }

    private companion object {
        const val ОКНО = 30
    }
}
