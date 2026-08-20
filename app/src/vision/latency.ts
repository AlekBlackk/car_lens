/**
 * Цена перехода границы native↔JS сверх времени, которое уже посчитано на нативной
 * стороне (`nativeMs` в каждом результате мостовых вызовов): `ценаМоста =
 * (t_после − t_до) − nativeMs`, где `t_до`/`t_после` берутся вокруг вызова плагина на
 * JS/worklet-стороне. Task 10 M-1.6.
 */

/**
 * Перцентиль ближайшим рангом — тот же метод, что `перцентиль()` в
 * `LatencyTest.kt` (M-1.5, `app/android/vision/src/androidTest/java/com/carlens/vision/LatencyTest.kt`):
 * без совпадения метода числа двух вех несравнимы.
 */
export function percentile(значения: number[], процент: number): number {
  if (значения.length === 0) {
    throw new Error('перцентиль пустой выборки')
  }
  const отсортированные = [...значения].sort((первое, второе) => первое - второе)
  const ранг = Math.max(1, Math.ceil((процент / 100) * отсортированные.length))
  return отсортированные[ранг - 1]!
}

/** Три вызова границы native↔JS, чью цену меряет накопитель. */
export type СтадияМоста = 'detect' | 'embedAndMatch' | 'matchEmbedding'

/** Сколько замеров стадии накопить перед печатью p50/p95 в лог. */
const РАЗМЕР_ВЫБОРКИ = 200

/**
 * Копит `ценаМоста` по трём стадиям границы и печатает p50/p95 в лог ровно один раз на
 * стадию, набрав `РАЗМЕР_ВЫБОРКИ` замеров. Живёт на JS-стороне (не в worklet-контексте):
 * `t_до`/`t_после` вокруг вызова плагина меряются внутри worklet'а, а само число —
 * простой `number` — долетает сюда через `Worklets.createRunOnJS`, как и остальные
 * результаты кадра в `App.tsx`. Класс, а не набор SharedValue, потому что копится массив
 * произвольной длины, а не примитив, переживающий клонирование worklet-замыкания.
 */
export class НакопительЦеныМоста {
  private readonly замеры: Record<СтадияМоста, number[]> = {
    detect: [],
    embedAndMatch: [],
    matchEmbedding: [],
  }

  private readonly напечатано: Record<СтадияМоста, boolean> = {
    detect: false,
    embedAndMatch: false,
    matchEmbedding: false,
  }

  /** `tДо`/`tПосле` — `Date.now()` непосредственно вокруг вызова плагина в worklet'е. */
  добавить(стадия: СтадияМоста, tДо: number, tПосле: number, nativeMs: number): void {
    const список = this.замеры[стадия]
    список.push(tПосле - tДо - nativeMs)
    if (список.length >= РАЗМЕР_ВЫБОРКИ && !this.напечатано[стадия]) {
      this.напечатано[стадия] = true
      // Единственный канал снятия чисел с устройства (Step 5 брифа Task 10):
      // `adb logcat -d -s ReactNativeJS:I | grep ЦЕНА_МОСТА`.
      // eslint-disable-next-line no-console
      console.log(
        `ЦЕНА_МОСТА ${стадия} p50=${percentile(список, 50).toFixed(3)} p95=${percentile(список, 95).toFixed(3)}`,
      )
    }
  }
}
