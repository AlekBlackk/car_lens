import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { LayoutChangeEvent, StyleSheet, Text, View } from 'react-native'
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
  type Orientation,
} from 'react-native-vision-camera'
import { useSharedValue, Worklets } from 'react-native-worklets-core'
import { НакопительЦеныМоста, bridgeClock, type СтадияМоста } from './src/vision/latency'
import { detect, embedAndMatch, matchEmbedding, type Box } from './src/vision/plugins'
import { shouldClassify } from './src/vision/throttle'
import { mapBoxToView } from './src/vision/viewMapping'

const MAX_DET = 4

/**
 * Физический угол телефона в градусах. Живёт здесь, а не в нативной части, потому что
 * источник угла — сам `Camera`: при `outputOrientation="device"` он слушает акселерометр
 * и рапортует поворот даже при выключенном системном автоповороте. `frame.orientation`
 * на нативной стороне поворот телефона не отражает вовсе — проверено на устройстве, он
 * остаётся `LANDSCAPE_RIGHT` и в портрете, и в горизонте.
 */
const УГЛЫ_УСТРОЙСТВА: Record<Orientation, number> = {
  portrait: 0,
  'landscape-left': 90,
  'portrait-upside-down': 180,
  'landscape-right': 270,
}
const CLASSIFY_PAD = 0.08
const CLASSIFY_TOP_K = 5

type Детекция = { боксы: Box[]; frameWidth: number; frameHeight: number }

/**
 * Топ-1 совпадение последнего классифицированного бокса. Чисто отладочный показ:
 * §13.3 дизайна прямо говорит, что итоговая подпись считается по EMA, которой в этой
 * вехе ещё нет — здесь просто «последнее, что вернул мост».
 *
 * `сверкаClassIdx` и `сверкаDelta` — тот же топ-1, но полученный вторым путём, плагином
 * `carvisionMatchEmbedding` на скопированном эмбеддинге. Существуют ради приёмки задачи 8:
 * два пути к одному и тому же kNN обязаны согласоваться, и увидеть это надо на устройстве,
 * а не в логе.
 */
type Классификация = {
  classIdx: number
  sim: number
  сверкаClassIdx: number
  сверкаDelta: number
} | null

export default function App(): React.JSX.Element {
  const { hasPermission, requestPermission } = useCameraPermission()
  // `outputOrientation` не указан явно: значение по умолчанию `'device'` уже следует
  // физической ориентации телефона и не влияет на Frame Processors (заметки задачи 1,
  // §5) — конвейер сам вычисляет угол кадра из `frame.orientation`.
  const device = useCameraDevice('back')
  const [детекция, установитьДетекцию] = useState<Детекция>({ боксы: [], frameWidth: 0, frameHeight: 0 })
  const [классификация, установитьКлассификацию] = useState<Классификация>(null)
  const [размерПревью, установитьРазмерПревью] = useState({ width: 0, height: 0 })
  // Ошибка моста: worklet исполняется вне React-дерева, ErrorBoundary там не работает,
  // поэтому исключение из detect/embedAndMatch/matchEmbedding долетает до JS-потока
  // только через runOnJS и оседает здесь баннером.
  const [ошибкаМоста, установитьОшибкуМоста] = useState<string | null>(null)
  // Момент последней классификации живёт на стороне worklet-контекста между вызовами
  // frame processor'а: обычный `useRef`/модульная переменная не пережили бы клонирование
  // замыкания worklet'а, а `useSharedValue` из react-native-worklets-core — переживает.
  const lastClassifyAt = useSharedValue(0)
  // Угол телефона переживает клонирование worklet-замыкания только как SharedValue:
  // обычное состояние React в воркет не доезжает.
  const deviceDegrees = useSharedValue(0)
  // Угол, в котором показывает превью. При выключенном автоповороте он остаётся
  // портретным, пока телефон крутят, а при включённом — следует за экраном. Боксы
  // рисуются поверх превью, поэтому наружу они обязаны выходить именно в этой системе.
  const previewDegrees = useSharedValue(0)

  // Накопитель цены моста (Task 10 M-1.6): живёт на JS-стороне, а не в worklet-контексте —
  // копит массивы произвольной длины по трём стадиям, что `useSharedValue` не умеет.
  // `useMemo`, а не `useRef`: инстанс нужен один раз и не должен создаваться заново на
  // каждый рендер, как и остальные мемоизированные мосты в этом компоненте.
  const накопительЦеныМоста = useMemo(() => new НакопительЦеныМоста(), [])
  const markBridgeCost = useMemo(
    () =>
      Worklets.createRunOnJS((стадия: СтадияМоста, tДо: number, tПосле: number, nativeMs: number) =>
        накопительЦеныМоста.добавить(стадия, tДо, tПосле, nativeMs),
      ),
    [накопительЦеныМоста],
  )

  useEffect(() => {
    if (!hasPermission) {
      requestPermission().catch(() => {
        // Отказ пользователя обрабатывается экраном ниже (hasPermission остаётся false).
      })
    }
  }, [hasPermission, requestPermission])

  // Мемоизировано: `useFrameProcessor` пересоздаёт worklet-контекст заново при каждой
  // смене зависимости, а немемоизированная функция меняла бы ссылку на каждый рендер.
  const publishDetection = useMemo(
    () =>
      Worklets.createRunOnJS((боксы: Box[], frameWidth: number, frameHeight: number) =>
        установитьДетекцию({ боксы, frameWidth, frameHeight }),
      ),
    [],
  )

  const publishClassification = useMemo(
    () =>
      Worklets.createRunOnJS(
        (classIdx: number, sim: number, сверкаClassIdx: number, сверкаDelta: number) =>
          установитьКлассификацию({ classIdx, sim, сверкаClassIdx, сверкаDelta }),
      ),
    [],
  )

  // Приёмник ошибок моста: вызывается из worklet'а через `Worklets.createRunOnJS`.
  // Рейт-лимит по значению, а не по времени — устойчивая ошибка (например, незагруженные
  // модели) иначе переотрисовывала бы баннер на каждом кадре без всякого смысла.
  const reportBridgeError = useCallback((текст: string) => {
    установитьОшибкуМоста((прежняя) => (прежняя === текст ? прежняя : текст))
  }, [])

  const reportErrorFromWorklet = useMemo(() => Worklets.createRunOnJS(reportBridgeError), [reportBridgeError])

  // Всё, что этот worklet захватывает из внешней области видимости, названо латиницей
  // намеренно: `react-native-worklets-core` 1.5.0 теряет ключи `__closure` с не-ASCII
  // символами, и на устройстве такой захват приезжает `undefined`. Проверено пробой
  // `typeof` внутри воркета на SM-S918B: `detect` и `shouldClassify` доехали функциями,
  // а `часыМоста`/`сообщитьОбОшибке` — `undefined`, из-за чего падал первый же вызов, а
  // за ним и `catch`, которому нечем было сообщить об ошибке. Локальные переменные
  // внутри тела воркета кириллицу переживают — ломается только захват.
  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet'
      try {
        // Замер цены моста (Task 10 M-1.6): `t_до`/`t_после` вокруг каждого из трёх
        // вызовов границы native↔JS через `bridgeClock()` — `performance.now()` из
        // `react-native-worklets-core` (декоратор кладёт его в рантайм воркета сам,
        // см. комментарий в `latency.ts`), а не `Date.now()`: цена перехода границы
        // ожидается в микросекундах, и миллисекундное разрешение `Date.now()` там —
        // не измерение, а шум (находка I3 финального ревью M-1.6). Троттлинг ниже
        // по-прежнему на `Date.now()` — там счёт на десятки миллисекунд, и цена
        // квантования не имеет значения.
        const tДетектДо = bridgeClock()
        const результат = detect(frame, MAX_DET, deviceDegrees.value, previewDegrees.value)
        markBridgeCost('detect', tДетектДо, bridgeClock(), результат.nativeMs)
        publishDetection(результат.boxes, результат.frameWidth, результат.frameHeight)

        // Троттлинг — на JS-стороне (§6.2 спеки): не чаще ~8 Гц и не больше двух кропов
        // за кадр, иначе классификатор молотил бы на каждом кадре впустую.
        const сейчас = Date.now()
        const дляКлассификации = shouldClassify(сейчас, lastClassifyAt.value, результат.boxes)
        if (дляКлассификации.length > 0) {
          lastClassifyAt.value = сейчас
          const tEmbedДо = bridgeClock()
          const { results: результатыКлассификации, nativeMs: embedNativeMs } = embedAndMatch(
            frame,
            дляКлассификации,
            CLASSIFY_PAD,
            CLASSIFY_TOP_K,
            deviceDegrees.value,
            previewDegrees.value,
          )
          markBridgeCost('embedAndMatch', tEmbedДо, bridgeClock(), embedNativeMs)
          const первый = результатыКлассификации[0]
          const топ1 = первый?.matches[0]
          if (первый != null && топ1 != null) {
            // Третий путь границы на том же векторе. EMA в этой вехе нет (§13.3), поэтому
            // вызов существует ради сверки: `matchEmbedding` обязан вернуть ровно то, что
            // `embedAndMatch` уже посчитал внутри себя.
            const tMatchДо = bridgeClock()
            const { matches: сверкаMatches, nativeMs: matchNativeMs } = matchEmbedding(
              frame,
              первый.embedding,
              CLASSIFY_TOP_K,
            )
            markBridgeCost('matchEmbedding', tMatchДо, bridgeClock(), matchNativeMs)
            const сверка = сверкаMatches[0]
            publishClassification(
              топ1.classIdx,
              топ1.sim,
              сверка == null ? -1 : сверка.classIdx,
              сверка == null ? Number.NaN : Math.abs(сверка.sim - топ1.sim),
            )
          }
        }
      } catch (отказ) {
        // Ранний выход на этом кадре: дорисовывать рамки по неполным данным опаснее,
        // чем пропустить кадр целиком.
        reportErrorFromWorklet(String(отказ))
        return
      }
    },
    [publishDetection, publishClassification, markBridgeCost, lastClassifyAt, reportErrorFromWorklet, deviceDegrees, previewDegrees],
  )

  const наРазметке = (событие: LayoutChangeEvent): void => {
    const { width, height } = событие.nativeEvent.layout
    установитьРазмерПревью({ width, height })
  }

  if (!hasPermission) {
    return (
      <View style={styles.экран}>
        <Text style={styles.текст}>Нужен доступ к камере</Text>
      </View>
    )
  }

  if (device == null) {
    return (
      <View style={styles.экран}>
        <Text style={styles.текст}>Задняя камера не найдена</Text>
      </View>
    )
  }

  const рамки =
    размерПревью.width > 0 && размерПревью.height > 0 && детекция.frameWidth > 0 && детекция.frameHeight > 0
      ? детекция.боксы.map((бокс) =>
          mapBoxToView(бокс, детекция.frameWidth, детекция.frameHeight, размерПревью.width, размерПревью.height),
        )
      : []

  return (
    <View style={styles.экран} onLayout={наРазметке}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        frameProcessor={frameProcessor}
        outputOrientation="device"
        onOutputOrientationChanged={(ориентация) => {
          deviceDegrees.value = УГЛЫ_УСТРОЙСТВА[ориентация]
        }}
        onPreviewOrientationChanged={(ориентация) => {
          previewDegrees.value = УГЛЫ_УСТРОЙСТВА[ориентация]
        }}
      />
      {ошибкаМоста != null && (
        <View style={styles.баннерОшибки} pointerEvents="none">
          <Text style={styles.текстОшибки}>{ошибкаМоста}</Text>
        </View>
      )}
      {рамки.map((рамка, индекс) => (
        <View
          key={индекс}
          pointerEvents="none"
          style={[
            styles.рамка,
            { left: рамка.left, top: рамка.top, width: рамка.width, height: рамка.height },
          ]}
        />
      ))}
      <View style={styles.оверлей}>
        <Text style={styles.текст}>Боксы: {детекция.боксы.length}</Text>
        {классификация != null && (
          <>
            <Text style={styles.текст}>
              Класс {классификация.classIdx} · sim {классификация.sim.toFixed(2)}
            </Text>
            <Text style={styles.текст}>
              сверка: класс {классификация.сверкаClassIdx} · Δ{' '}
              {классификация.сверкаDelta.toExponential(1)}
            </Text>
          </>
        )}
      </View>
    </View>
  )
}

const styles = StyleSheet.create({
  экран: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: '#000' },
  оверлей: { position: 'absolute', bottom: 32, alignItems: 'center' },
  текст: { color: '#fff', fontSize: 18 },
  рамка: { position: 'absolute', borderWidth: 2, borderColor: '#00ff00' },
  баннерОшибки: {
    position: 'absolute',
    top: 32,
    left: 16,
    right: 16,
    backgroundColor: 'rgba(178, 24, 24, 0.9)',
    borderRadius: 8,
    padding: 12,
  },
  текстОшибки: { color: '#fff', fontSize: 15, textAlign: 'center' },
})
