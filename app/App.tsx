import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { LayoutChangeEvent, StyleSheet, Text, View } from 'react-native'
import { Camera, useCameraDevice, useCameraPermission, useFrameProcessor } from 'react-native-vision-camera'
import { useSharedValue, Worklets } from 'react-native-worklets-core'
import { detect, embedAndMatch, matchEmbedding, type Box } from './src/vision/plugins'
import { shouldClassify } from './src/vision/throttle'
import { mapBoxToView } from './src/vision/viewMapping'

const MAX_DET = 4
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
  const последняяКлассификация = useSharedValue(0)

  useEffect(() => {
    if (!hasPermission) {
      requestPermission().catch(() => {
        // Отказ пользователя обрабатывается экраном ниже (hasPermission остаётся false).
      })
    }
  }, [hasPermission, requestPermission])

  // Мемоизировано: `useFrameProcessor` пересоздаёт worklet-контекст заново при каждой
  // смене зависимости, а немемоизированная функция меняла бы ссылку на каждый рендер.
  const обновитьДетекцию = useMemo(
    () =>
      Worklets.createRunOnJS((боксы: Box[], frameWidth: number, frameHeight: number) =>
        установитьДетекцию({ боксы, frameWidth, frameHeight }),
      ),
    [],
  )

  const обновитьКлассификацию = useMemo(
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

  const сообщитьОбОшибке = useMemo(() => Worklets.createRunOnJS(reportBridgeError), [reportBridgeError])

  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet'
      try {
        const результат = detect(frame, MAX_DET)
        обновитьДетекцию(результат.boxes, результат.frameWidth, результат.frameHeight)

        // Троттлинг — на JS-стороне (§6.2 спеки): не чаще ~8 Гц и не больше двух кропов
        // за кадр, иначе классификатор молотил бы на каждом кадре впустую.
        const сейчас = Date.now()
        const дляКлассификации = shouldClassify(сейчас, последняяКлассификация.value, результат.boxes)
        if (дляКлассификации.length > 0) {
          последняяКлассификация.value = сейчас
          const результатыКлассификации = embedAndMatch(frame, дляКлассификации, CLASSIFY_PAD, CLASSIFY_TOP_K)
          const первый = результатыКлассификации[0]
          const топ1 = первый?.matches[0]
          if (первый != null && топ1 != null) {
            // Третий путь границы на том же векторе. EMA в этой вехе нет (§13.3), поэтому
            // вызов существует ради сверки: `matchEmbedding` обязан вернуть ровно то, что
            // `embedAndMatch` уже посчитал внутри себя.
            const сверка = matchEmbedding(frame, первый.embedding, CLASSIFY_TOP_K)[0]
            обновитьКлассификацию(
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
        сообщитьОбОшибке(String(отказ))
        return
      }
    },
    [обновитьДетекцию, обновитьКлассификацию, последняяКлассификация, сообщитьОбОшибке],
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
      <Camera style={StyleSheet.absoluteFill} device={device} isActive={true} frameProcessor={frameProcessor} />
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
