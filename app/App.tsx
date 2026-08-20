import React, { useEffect, useMemo, useState } from 'react'
import { LayoutChangeEvent, StyleSheet, Text, View } from 'react-native'
import { Camera, useCameraDevice, useCameraPermission, useFrameProcessor } from 'react-native-vision-camera'
import { Worklets } from 'react-native-worklets-core'
import { detect, type Box } from './src/vision/plugins'
import { mapBoxToView } from './src/vision/viewMapping'

const MAX_DET = 4

type Детекция = { боксы: Box[]; frameWidth: number; frameHeight: number }

export default function App(): React.JSX.Element {
  const { hasPermission, requestPermission } = useCameraPermission()
  // `outputOrientation` не указан явно: значение по умолчанию `'device'` уже следует
  // физической ориентации телефона и не влияет на Frame Processors (заметки задачи 1,
  // §5) — конвейер сам вычисляет угол кадра из `frame.orientation`.
  const device = useCameraDevice('back')
  const [детекция, установитьДетекцию] = useState<Детекция>({ боксы: [], frameWidth: 0, frameHeight: 0 })
  const [размерПревью, установитьРазмерПревью] = useState({ width: 0, height: 0 })

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

  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet'
      const результат = detect(frame, MAX_DET)
      обновитьДетекцию(результат.boxes, результат.frameWidth, результат.frameHeight)
    },
    [обновитьДетекцию],
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
      </View>
    </View>
  )
}

const styles = StyleSheet.create({
  экран: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: '#000' },
  оверлей: { position: 'absolute', bottom: 32, alignItems: 'center' },
  текст: { color: '#fff', fontSize: 18 },
  рамка: { position: 'absolute', borderWidth: 2, borderColor: '#00ff00' },
})
