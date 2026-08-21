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
import { type Решение } from './src/vision/decide'
import { classLabel } from './src/vision/labels'
import { НакопительЦеныМоста, bridgeClock, type СтадияМоста } from './src/vision/latency'
import {
  stepPipeline,
  ПУСТОЕ_СОСТОЯНИЕ,
  type МостКонвейера,
  type ПодписьТрека,
  type СобытиеКлассификации,
  type СостояниеКонвейера,
} from './src/vision/pipeline'
import { detect, embedAndMatch, matchEmbedding, type Box } from './src/vision/plugins'
import { checkSlotContract } from './src/vision/slotContract'
import { StabilityCollector } from './src/vision/stability'
import { mapBoxToView } from './src/vision/viewMapping'

const MAX_DET = 4

/**
 * Множитель `frame.timestamp` → миллисекунды. Единица выяснена по исходникам
 * VisionCamera 4.6.4, а не угадана: `FrameHostObject.cpp` отдаёт значение
 * `Frame.getTimestamp()` без преобразования, приводя его к `double`, а `Frame.java`
 * возвращает `imageProxy.getImageInfo().getTimestamp()` — то есть штамп CameraX, который
 * документирован в наносекундах. Отсюда 1e-6.
 *
 * Этими часами конвейер мерит и троттлинг, и смерть трека, поэтому `Date.now()` тут не
 * используется: прыжок стенных часов (NTP, смена зоны) разом убил бы половину треков
 * (см. комментарий в `throttle.ts`).
 *
 * Имя латиницей: константу захватывает воркет, а ключи `__closure` с не-ASCII символами
 * до worklet-рантайма не доезжают (см. `workletClosure.test.ts`).
 */
const TIME_SCALE_MS = 1e-6

/**
 * Физический угол телефона в градусах. Живёт здесь, а не в нативной части, потому что
 * источник угла — сам `Camera`: при `outputOrientation="device"` он слушает акселерометр
 * и рапортует поворот даже при выключенном системном автоповороте. `frame.orientation`
 * на нативной стороне поворот телефона не отражает вовсе — проверено на устройстве, он
 * остаётся `LANDSCAPE_RIGHT` и в портрете, и в горизонте.
 */
/** Отступ подписей от края экрана. */
const ОТСТУП_ПОДПИСЕЙ = 32

const УГЛЫ_УСТРОЙСТВА: Record<Orientation, number> = {
  portrait: 0,
  'landscape-left': 90,
  'portrait-upside-down': 180,
  'landscape-right': 270,
}
const CLASSIFY_PAD = 0.08
const CLASSIFY_TOP_K = 5

/**
 * Всё, что экрану нужно для отрисовки одного кадра: подписи треков плюс разрешение кадра,
 * в котором посчитаны их боксы. Размер лежит рядом с подписями, а не отдельным состоянием,
 * намеренно: два состояния обновлялись бы двумя рендерами, и в промежутке боксы одного
 * разрешения рисовались бы по системе координат другого.
 */
type КадрРазметки = { подписи: ПодписьТрека[]; frameWidth: number; frameHeight: number }

/**
 * Текст подписи по решению §6.4. Отметка «не откалибровано» стоит, пока таблица калибровки
 * не размечена (`tuned: false`, §13.5): без неё плейсхолдер незаметно доживает до релиза и
 * выдаётся пользователю за измеренный процент.
 */
const текстПодписи = (решение: Решение): string => {
  if (решение.вид === 'определяю') return 'Определяю…'
  if (решение.вид === 'ближе') return 'Подойдите ближе'
  if (решение.вид === 'неизвестно') return 'Не узнаю'
  if (решение.вид === 'бренд') return classLabel(решение.classIdx)?.brand ?? 'Не узнаю'
  const имя = classLabel(решение.classIdx)?.display ?? `класс ${решение.classIdx}`
  const процент = Math.round(решение.уверенность * 100)
  return решение.откалибровано ? `${имя} · ${процент}%` : `${имя} · ${процент}% (не откалибровано)`
}

export default function App(): React.JSX.Element {
  const { hasPermission, requestPermission } = useCameraPermission()
  // `outputOrientation` не указан явно: значение по умолчанию `'device'` уже следует
  // физической ориентации телефона и не влияет на Frame Processors (заметки задачи 1,
  // §5) — конвейер сам вычисляет угол кадра из `frame.orientation`.
  const device = useCameraDevice('back')
  const [разметка, установитьРазметку] = useState<КадрРазметки>({
    подписи: [],
    frameWidth: 0,
    frameHeight: 0,
  })
  const [размерПревью, установитьРазмерПревью] = useState({ width: 0, height: 0 })
  // Ошибка моста: worklet исполняется вне React-дерева, ErrorBoundary там не работает,
  // поэтому исключение из detect/embedAndMatch/matchEmbedding долетает до JS-потока
  // только через runOnJS и оседает здесь баннером.
  const [ошибкаМоста, установитьОшибкуМоста] = useState<string | null>(null)
  // Состояние конвейера (треки, их EMA и топ-1) переживает кадры только как SharedValue:
  // обычный `useRef` или модульная переменная не пережили бы клонирование замыкания
  // worklet'а, а `useSharedValue` из react-native-worklets-core — переживает.
  const pipelineState = useSharedValue<СостояниеКонвейера>(ПУСТОЕ_СОСТОЯНИЕ)
  // Угол телефона переживает клонирование worklet-замыкания только как SharedValue:
  // обычное состояние React в воркет не доезжает.
  const deviceDegrees = useSharedValue(0)
  // Самопроверка контракта слотов гоняется один раз за запуск: она делает два лишних
  // вызова моста, и в горячем пути ей делать нечего.
  const slotContractChecked = useSharedValue(false)
  // Угол, в котором показывает превью. При выключенном автоповороте он остаётся
  // портретным, пока телефон крутят, а при включённом — следует за экраном. Боксы
  // рисуются поверх превью, поэтому наружу они обязаны выходить именно в этой системе.
  const previewDegrees = useSharedValue(0)
  // Тот же угол устройства, но состоянием React, а не SharedValue: им доворачиваются
  // подписи. Окно приложения залочено в портрет, поэтому при горизонтальном хвате текст
  // без доворота лежал бы на боку — единственное, что теряется от фиксации окна.
  const [уголПодписей, установитьУголПодписей] = useState(0)
  // Высота блока подписей нужна, чтобы прижать его к краю ровно, а не на глаз: сдвиг
  // считается от центра экрана, и без реальной высоты блок либо не дойдёт до края, либо
  // уедет за него.
  const [высотаПодписей, установитьВысотуПодписей] = useState(0)

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

  // Накопитель устойчивости подписи (Task 10 M-1.6). Живёт рядом с накопителем цены моста
  // и по той же причине: копит словари произвольного размера, чего SharedValue не умеет.
  // События приходят из конвейера кадра — по одному на каждую состоявшуюся классификацию.
  const накопительСтабильности = useMemo(() => new StabilityCollector(), [])
  const markStability = useMemo(
    () =>
      Worklets.createRunOnJS((события: СобытиеКлассификации[]) =>
        накопительСтабильности.добавить(события),
      ),
    [накопительСтабильности],
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
  const publishOverlay = useMemo(
    () =>
      Worklets.createRunOnJS((подписи: ПодписьТрека[], frameWidth: number, frameHeight: number) =>
        установитьРазметку({ подписи, frameWidth, frameHeight }),
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
  // `typeof` внутри воркета на SM-S918B: латиничные захваты вроде `detect` доехали
  // функциями, а `часыМоста`/`сообщитьОбОшибке` — `undefined`, из-за чего падал первый же
  // вызов, а за ним и `catch`, которому нечем было сообщить об ошибке. Локальные
  // переменные внутри тела воркета кириллицу переживают — ломается только захват.
  const reportSlotContract = useMemo(
    () =>
      Worklets.createRunOnJS((ок: boolean, отчёт: string) => {
        // eslint-disable-next-line no-console
        console.log(`КОНТРАКТ_СЛОТОВ ${ок ? 'OK' : 'ПРОВАЛ'} ${отчёт}`)
      }),
    [],
  )

  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet'
      try {
        // Замер цены моста (Task 10 M-1.6): `t_до`/`t_после` вокруг каждого из трёх
        // вызовов границы native↔JS через `bridgeClock()` — `performance.now()` из
        // `react-native-worklets-core` (декоратор кладёт его в рантайм воркета сам,
        // см. комментарий в `latency.ts`), а не `Date.now()`: цена перехода границы
        // ожидается в микросекундах, и миллисекундное разрешение `Date.now()` там —
        // не измерение, а шум (находка I3 финального ревью M-1.6).
        if (!slotContractChecked.value) {
          slotContractChecked.value = true
          const итог = checkSlotContract(frame, CLASSIFY_PAD, deviceDegrees.value, previewDegrees.value)
          reportSlotContract(итог.ок, итог.отчёт)
        }

        const tДетектДо = bridgeClock()
        const результат = detect(frame, MAX_DET, deviceDegrees.value, previewDegrees.value)
        markBridgeCost('detect', tДетектДо, bridgeClock(), результат.nativeMs)

        // Часы конвейера — timestamp кадра, а не стенные часы: троттлинг и смерть трека
        // обязаны мерить одной монотонной линейкой (см. `TIME_SCALE_MS` и `throttle.ts`).
        const nowMs = frame.timestamp * TIME_SCALE_MS

        // Мост конвейера собирается здесь, потому что только здесь есть `frame`: сам
        // `stepPipeline` о `Frame` не знает ничего и потому гоняется в Node на фейке.
        // Замер цены моста остаётся на этой стороне по той же причине — в чистом модуле
        // ему делать нечего.
        const мост: МостКонвейера = {
          embed(боксы: Box[]) {
            'worklet'
            const tEmbedДо = bridgeClock()
            const ответ = embedAndMatch(
              frame, боксы, CLASSIFY_PAD, CLASSIFY_TOP_K, deviceDegrees.value, previewDegrees.value,
            )
            markBridgeCost('embedAndMatch', tEmbedДо, bridgeClock(), ответ.nativeMs)
            return ответ.results
          },
          matchEma(вектор: Float32Array) {
            'worklet'
            const tMatchДо = bridgeClock()
            // `frame` первым аргументом — фактическая сигнатура `matchEmbedding`: плагин
            // зовётся на кадре, хотя ищет по чужому вектору.
            const ответ = matchEmbedding(frame, вектор, CLASSIFY_TOP_K)
            markBridgeCost('matchEmbedding', tMatchДо, bridgeClock(), ответ.nativeMs)
            return ответ.matches
          },
        }

        const шаг = stepPipeline(
          pipelineState.value,
          {
            боксы: результат.boxes,
            nowMs,
            ширинаКадра: результат.frameWidth,
            высотаКадра: результат.frameHeight,
          },
          мост,
        )
        pipelineState.value = шаг.состояние
        // Пустой массив через границу не гоняется: вызов `createRunOnJS` сам по себе не
        // бесплатен, а кадров без классификации по §6.2 большинство.
        if (шаг.события.length > 0) markStability(шаг.события)
        publishOverlay(шаг.подписи, результат.frameWidth, результат.frameHeight)
      } catch (отказ) {
        // Ранний выход на этом кадре: дорисовывать рамки по неполным данным опаснее,
        // чем пропустить кадр целиком. Сюда же приходит `BridgeContractError` из
        // конвейера — состояние в этом случае не записано, поэтому следующий кадр
        // считается от прежних треков, а не от половинчатых.
        reportErrorFromWorklet(String(отказ))
        return
      }
    },
    [publishOverlay, markBridgeCost, markStability, pipelineState, reportErrorFromWorklet, deviceDegrees, previewDegrees, slotContractChecked, reportSlotContract],
  )

  // Пользовательская вертикаль при горизонтальном хвате идёт вдоль ширины экрана, при
  // вертикальном — вдоль высоты. Сдвиг ведёт блок из центра экрана к нижнему краю.
  const вертикальПользователя =
    уголПодписей === 90 || уголПодписей === 270 ? размерПревью.width : размерПревью.height
  const сдвигПодписей = Math.max(
    0,
    вертикальПользователя / 2 - ОТСТУП_ПОДПИСЕЙ - высотаПодписей / 2,
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

  // Отказ от фронтальной камеры живёт здесь, а не в нативном плагине, потому что только
  // здесь известно положение линзы. В плагине для этого есть единственный признак —
  // `frame.isMirrored`, и он отвечает на другой вопрос: отражён ли буфер кадра
  // (отрицательный X в `sensorToBufferTransformMatrix`). CameraX буфер анализа не
  // отражает даже на селфи-камере, поэтому проверено на SM-S918B: с фронтальной камерой
  // флаг остаётся `false`, `require` в плагине молчит и кадры идут в конвейер как ни в
  // чём не бывало. `require` там оставлен страховкой от действительно зеркального
  // буфера, а внятный отказ пользователю выдаётся отсюда.
  if (device.position !== 'back') {
    return (
      <View style={styles.экран}>
        <Text style={styles.текст}>Фронтальная камера не поддерживается</Text>
      </View>
    )
  }

  // Порядок строк — по id трека, а не по порядку в состоянии конвейера: там треки
  // переставляются при смерти и рождении, и строки прыгали бы местами на живом кадре.
  const подписи = разметка.подписи.slice().sort((первая, вторая) => первая.id - вторая.id)

  // Стража на размер кадра обязательна: `mapBoxToView` на пустом кадре бросает, а до
  // первого результата конвейера разметка ровно такая.
  const рамки =
    размерПревью.width > 0 && размерПревью.height > 0 && разметка.frameWidth > 0 && разметка.frameHeight > 0
      ? подписи.map((подпись) => ({
          id: подпись.id,
          рамка: mapBoxToView(
            подпись.бокс,
            разметка.frameWidth,
            разметка.frameHeight,
            размерПревью.width,
            размерПревью.height,
          ),
        }))
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
          установитьУголПодписей(УГЛЫ_УСТРОЙСТВА[ориентация])
        }}
        onPreviewOrientationChanged={(ориентация) => {
          previewDegrees.value = УГЛЫ_УСТРОЙСТВА[ориентация]
        }}
      />
      {ошибкаМоста != null && (
        <View
          style={[styles.баннерОшибки, { transform: [{ rotate: `${-уголПодписей}deg` }] }]}
          pointerEvents="none"
        >
          <Text style={styles.текстОшибки}>{ошибкаМоста}</Text>
        </View>
      )}
      {рамки.map((запись) => (
        <View
          key={запись.id}
          pointerEvents="none"
          style={[
            styles.рамка,
            {
              left: запись.рамка.left,
              top: запись.рамка.top,
              width: запись.рамка.width,
              height: запись.рамка.height,
            },
          ]}
        />
      ))}
      {/*
        Подписи стоят у нижнего края с точки зрения того, кто держит телефон. Прижать их
        стилями контейнера нельзя: `transform` крутит блок вокруг его центра уже после
        раскладки, и прижатый к краю блок после поворота уезжает к середине экрана —
        ровно это и было видно на устройстве. Поэтому блок центрируется, а потом
        сдвигается `translateY` — он применяется после `rotate`, то есть вдоль
        пользовательской вертикали, а не экранной.
      */}
      <View style={styles.оверлей} pointerEvents="none">
        <View
          onLayout={(событие) => установитьВысотуПодписей(событие.nativeEvent.layout.height)}
          style={{
            alignItems: 'center',
            transform: [{ rotate: `${-уголПодписей}deg` }, { translateY: сдвигПодписей }],
          }}
        >
        {/*
          Подпись живёт здесь, а не у рамки: блок доворачивается на угол устройства, и
          только в нём текст стоит вертикально при любом хвате. Строка на трек, текст
          считается по EMA трека решением §6.4 — не по последнему ответу моста.
        */}
        {подписи.length === 0 ? (
          <Text style={styles.текст}>Определяю…</Text>
        ) : (
          подписи.map((подпись) => (
            <Text key={подпись.id} style={styles.текст}>
              {текстПодписи(подпись.решение)}
            </Text>
          ))
        )}
        </View>
      </View>
    </View>
  )
}

const styles = StyleSheet.create({
  экран: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: '#000' },
  оверлей: { ...StyleSheet.absoluteFillObject, alignItems: 'center', justifyContent: 'center' },
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
