/**
 * Цена перехода границы native↔JS сверх времени, которое уже посчитано на нативной
 * стороне (`nativeMs` в каждом результате мостовых вызовов): `ценаМоста =
 * (t_после − t_до) − nativeMs`, где `t_до`/`t_после` берутся вокруг вызова плагина на
 * JS/worklet-стороне. Task 10 M-1.6.
 */

/**
 * `performance` в worklet-рантайме объявлен здесь, а не взят из `globalThis`, и это
 * требование устройства, а не вкусовщина. `react-native-worklets-core` 1.5.0 на каждое
 * обращение к объекту `globalThis` изнутри воркета заводит глобальную JNI-ссылку и не
 * освобождает её: на SM-S918B при 28-30 кадрах/с таблица ссылок (max=51200) переполнялась
 * примерно за секунду и процесс падал с
 * `JNI ERROR (app bug): global reference table overflow`. Бисекция на устройстве: пустой
 * воркет — жив; `detect()` каждый кадр — жив; `Date.now()` голым идентификатором — жив;
 * `performance.now()` голым идентификатором — жив; любое обращение вида
 * `globalThis.performance` или `globalThis['Date']` — падение. Голый идентификатор
 * резолвится без обращения к объекту глобали и течи не даёт.
 */
declare const performance: { now: () => number } | undefined

/**
 * Часы для замера цены моста. `Date.now()` квантуется до 1 мс в Hermes, а сама цена
 * перехода границы ожидается в десятках-сотнях **микросекунд** — на такой шкале `Date.now()`
 * не измеряет цену, а генерирует шум (см. находку I3 финального ревью M-1.6): разность двух
 * целых миллисекунд то и дело отрицательна, и p50/p95 по такой выборке ничего не значат.
 *
 * `react-native-worklets-core` 1.5.0 сам кладёт `performance.now()` в глобальный объект
 * каждого worklet-рантайма — `WKTJsiPerformanceDecorator.h` в его исходниках, обёртка над
 * `std::chrono::steady_clock` с наносекундным внутренним разрешением, отданная в JS как
 * дробные миллисекунды. `Date.now()` — фолбэк на случай окружения, где `performance` не
 * задекорирован (например юнит-тест вне worklet-рантайма).
 *
 * Директива `'worklet'` обязательна: функция захватывается замыканием frame processor'а, а
 * `react-native-worklets-core` 1.5.0 отказывается переносить в worklet-рантайм обычную
 * JS-функцию («Regular javascript function 'bridgeClock' cannot be shared», проверено на
 * SM-S918B). Юнит-тесты вне worklet-рантайма директиву игнорируют.
 */
export function bridgeClock(): number {
  'worklet'
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now()
}

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

  /** `tДо`/`tПосле` — `bridgeClock()` непосредственно вокруг вызова плагина в worklet'е. */
  добавить(стадия: СтадияМоста, tДо: number, tПосле: number, nativeMs: number): void {
    // Ранний выход после печати: без него массив стадии рос бы неограниченно на длинной
    // сессии — `напечатано` раньше блокировал только повторную печать, не рост массива
    // (отложенный minor Task 10, закрыт заодно с I3 финального ревью M-1.6).
    if (this.напечатано[стадия]) {
      return
    }
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
