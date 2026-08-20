import React, { useEffect, useMemo } from 'react'
import { StyleSheet, Text, View } from 'react-native'
import { Camera, useCameraDevice, useCameraPermission, useFrameProcessor } from 'react-native-vision-camera'
import { Worklets } from 'react-native-worklets-core'
import { detect } from './src/vision/plugins'

const MAX_DET = 4

export default function App(): React.JSX.Element {
  const { hasPermission, requestPermission } = useCameraPermission()
  // `outputOrientation` не указан явно: значение по умолчанию `'device'` уже следует
  // физической ориентации телефона и не влияет на Frame Processors (заметки задачи 1,
  // §5) — конвейер сам вычисляет угол кадра из `frame.orientation`.
  const device = useCameraDevice('back')
  const [числоБоксов, установитьЧислоБоксов] = React.useState(0)

  useEffect(() => {
    if (!hasPermission) {
      requestPermission().catch(() => {
        // Отказ пользователя обрабатывается экраном ниже (hasPermission остаётся false).
      })
    }
  }, [hasPermission, requestPermission])

  // Мемоизировано: `useFrameProcessor` пересоздаёт worklet-контекст заново при каждой
  // смене зависимости, а немемоизированная функция меняла бы ссылку на каждый рендер.
  const обновитьЧислоБоксов = useMemo(
    () => Worklets.createRunOnJS((count: number) => установитьЧислоБоксов(count)),
    [],
  )

  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet'
      const результат = detect(frame, MAX_DET)
      обновитьЧислоБоксов(результат.boxes.length)
    },
    [обновитьЧислоБоксов],
  )

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

  return (
    <View style={styles.экран}>
      <Camera style={StyleSheet.absoluteFill} device={device} isActive={true} frameProcessor={frameProcessor} />
      <View style={styles.оверлей}>
        <Text style={styles.текст}>Боксы: {числоБоксов}</Text>
      </View>
    </View>
  )
}

const styles = StyleSheet.create({
  экран: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: '#000' },
  оверлей: { position: 'absolute', bottom: 32, alignItems: 'center' },
  текст: { color: '#fff', fontSize: 18 },
})
