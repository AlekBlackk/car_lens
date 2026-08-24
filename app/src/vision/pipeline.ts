/**
 * Порядок стадий §6.2: трекинг → троттлинг → embed → EMA → matchEma → decide.
 *
 * Функция не зовёт мост сама — мост приходит аргументом. Отсюда её главное свойство: она
 * детерминирована относительно `мост`, и весь конвейер гоняется в Node на фейке, а не
 * только на телефоне. Про `Frame` модуль не знает ничего.
 *
 * `matches` от мгновенного эмбеддинга §13.3 разрешает использовать только для отладочного
 * показа; в решение они не идут — иначе EMA оказывается декоративной. Наружу они выходят
 * событиями классификации, которые читает счётчик стабильности.
 */
import { decide, decisionKey, type Решение } from './decide'
import { EMA_ALPHA, emaStep } from './ema'
import { sameBrand } from './labels'
import type { Box, EmbedMatchResult, Match } from './plugins'
import { CALIBRATION, THRESHOLDS } from './thresholds'
import { CLASSIFY_INTERVAL_MS, pickForClassify } from './throttle'
import { updateTracks, type ТрекГеометрия } from './tracker'

/**
 * Имена полей — ASCII, и это не стиль, а требование границы: состояние конвейера живёт в
 * `SharedValue`, а `react-native-worklets-core` 1.5.0 не переносит ключи с не-ASCII
 * символами ни в воркет-рантайм, ни обратно. Ключ просто исчезает, молча. Проверено на
 * SM-S918B; стережёт `pipeline.test.ts` («состояние переживает SharedValue»).
 */
export type Трек = {
  id: number
  box: Box
  observations: number
  seenAt: number
  classifiedAt: number
  ema: number[] | null
  tops: Match[]
}

/** Поля ASCII по той же причине, что у `Трек`: объект целиком лежит в `SharedValue`. */
export type СостояниеКонвейера = { tracks: Трек[]; nextId: number }

/**
 * Подпись трека для отрисовки. `топы` — тот же массив, что лежит в треке, той же ссылкой:
 * при классификации он переприсваивается целиком, на месте его никто не правит, поэтому
 * копия здесь означала бы новый массив каждым кадром без единого читателя.
 */
export type ПодписьТрека = { id: number; box: Box; decision: Решение; tops: Match[] }

/**
 * То, что читает счётчик стабильности: оба топ-1 одного и того же наблюдения плюс ключ
 * решения. `ключРешения` здесь — сырой, от `decide()`: пользовательский порог лежит в
 * настройках, а настройки в воркет не доезжают. Ключ видимого решения ставит
 * `ключиВидимыхРешений` на JS-стороне, и в накопитель едет уже он.
 */
export type СобытиеКлассификации = {
  id: number
  top1Мгновенный: number
  top1Ema: number
  decisionKey: string
}

export type МостКонвейера = {
  embed(боксы: Box[]): EmbedMatchResult[]
  matchEma(вектор: Float32Array): Match[]
}

export class BridgeContractError extends Error {
  constructor(сообщение: string) {
    super(сообщение)
    this.name = 'BridgeContractError'
  }
}

export const ПУСТОЕ_СОСТОЯНИЕ: СостояниеКонвейера = { tracks: [], nextId: 1 }

/**
 * Сколько гипотез трек несёт наружу. Три — требование §6.6: столько показывает экран
 * «Детали». Мост по-прежнему просят про пять (`CLASSIFY_TOP_K`), потому что это его цена
 * поиска по индексу, а не цена переноса; лишние две кочевали бы в React-состояние каждым
 * кадром без единого потребителя.
 *
 * Решение §6.4 считается по первым двум гипотезам и на эту константу не смотрит: опустить
 * её до двух — потерять третью строку «Деталей», поднять до пяти — ничего не изменить в
 * подписи.
 *
 * Имя латиницей по той же причине, что у `toFloat32`: константа захватывается замыканием
 * `stepPipeline`, а ключи с не-ASCII символами до worklet-рантайма не доезжают. Кириллица
 * здесь дала бы на телефоне `slice(0, undefined)` — то есть все пять гипотез моста и
 * молчащие тесты. Стража — `workletClosure.test.ts`.
 */
export const MAX_HYPOTHESES = 3

/*
 * Латиница в имени — то же требование `__closure`, что у хэндлов плагинов в `plugins.ts`:
 * модульный хелпер захватывается замыканием `stepPipeline`, а ключи с не-ASCII символами
 * до worklet-рантайма не доезжают и приходят `undefined`. Стража — `workletClosure.test.ts`.
 */
const toFloat32 = (числа: number[]): Float32Array => {
  'worklet'
  const итог = new Float32Array(числа.length)
  for (let номер = 0; номер < числа.length; номер++) итог[номер] = числа[номер]!
  return итог
}

export function stepPipeline(
  состояние: СостояниеКонвейера,
  вход: { боксы: Box[]; nowMs: number; ширинаКадра: number; высотаКадра: number },
  мост: МостКонвейера,
): { состояние: СостояниеКонвейера; подписи: ПодписьТрека[]; события: СобытиеКлассификации[] } {
  'worklet'
  const nowMs = вход.nowMs

  const геометрия: ТрекГеометрия[] = new Array(состояние.tracks.length)
  for (let номер = 0; номер < состояние.tracks.length; номер++) {
    const трек = состояние.tracks[номер]!
    геометрия[номер] = { id: трек.id, бокс: трек.box, виденВ: трек.seenAt }
  }
  const итогТрекера = updateTracks(геометрия, вход.боксы, nowMs, состояние.nextId)

  // Новый массив треков: входное состояние не мутируется, поэтому «шаг не испортил вход»
  // проверяется тестом в две строки.
  const треки: Трек[] = []
  for (let номер = 0; номер < состояние.tracks.length; номер++) {
    const прежний = состояние.tracks[номер]!
    let мёртв = false
    for (let к = 0; к < итогТрекера.умершие.length; к++) {
      if (итогТрекера.умершие[к] === прежний.id) {
        мёртв = true
        break
      }
    }
    if (мёртв) continue
    let бокс = прежний.box
    let виденВ = прежний.seenAt
    for (let к = 0; к < итогТрекера.сопоставленные.length; к++) {
      if (итогТрекера.сопоставленные[к]!.id === прежний.id) {
        бокс = итогТрекера.сопоставленные[к]!.бокс
        виденВ = nowMs
        break
      }
    }
    треки.push({
      id: прежний.id,
      box: бокс,
      seenAt: виденВ,
      observations: прежний.observations,
      classifiedAt: прежний.classifiedAt,
      ema: прежний.ema,
      tops: прежний.tops,
    })
  }
  for (let к = 0; к < итогТрекера.рождённые.length; к++) {
    const рождённый = итогТрекера.рождённые[к]!
    треки.push({
      id: рождённый.id,
      box: рождённый.бокс,
      seenAt: nowMs,
      observations: 0,
      // Родившийся трек классифицируется в том же кадре: интервал считается уже истёкшим.
      // Так, а не `0` и не `-Infinity`, потому что `nowMs` — время кадра, и оно может
      // начинаться с любого значения.
      classifiedAt: nowMs - CLASSIFY_INTERVAL_MS,
      ema: null,
      tops: [],
    })
  }

  const кандидаты = pickForClassify(треки, nowMs, THRESHOLDS.minShortSidePx)
  const события: СобытиеКлассификации[] = []

  if (кандидаты.length > 0) {
    const боксы: Box[] = new Array(кандидаты.length)
    for (let к = 0; к < кандидаты.length; к++) боксы[к] = кандидаты[к]!.box
    const результаты = мост.embed(боксы)
    // Проверка до единой записи в состояние: частичное применение оставило бы часть
    // треков верными, часть испорченными, и различить их потом уже нечем.
    if (результаты.length !== кандидаты.length) {
      throw new BridgeContractError(
        `embed вернул ${результаты.length} результатов на ${кандидаты.length} боксов`,
      )
    }
    for (let к = 0; к < кандидаты.length; к++) {
      const id = кандидаты[к]!.id
      let трек: Трек | null = null
      for (let номер = 0; номер < треки.length; номер++) {
        if (треки[номер]!.id === id) {
          трек = треки[номер]!
          break
        }
      }
      if (трек == null) continue
      const шаг = emaStep(трек.ema, результаты[к]!.embedding, EMA_ALPHA)
      if (!шаг.принят) continue
      трек.ema = шаг.вектор
      трек.observations = трек.observations + 1
      трек.classifiedAt = nowMs
      трек.tops = мост.matchEma(toFloat32(шаг.вектор!)).slice(0, MAX_HYPOTHESES)
      const мгновенные = результаты[к]!.matches
      события.push({
        id,
        top1Мгновенный: мгновенные.length > 0 ? мгновенные[0]!.classIdx : -1,
        top1Ema: трек.tops.length > 0 ? трек.tops[0]!.classIdx : -1,
        decisionKey: '',
      })
    }
  }

  const подписи: ПодписьТрека[] = new Array(треки.length)
  for (let номер = 0; номер < треки.length; номер++) {
    const трек = треки[номер]!
    const топ1 = трек.tops.length > 0 ? трек.tops[0]! : null
    const топ2 = трек.tops.length > 1 ? трек.tops[1]! : null
    const решение = decide(
      {
        наблюдений: трек.observations,
        короткаяСторона: Math.min(трек.box.w, трек.box.h),
        топ1,
        топ2,
        брендыСовпали: топ1 != null && топ2 != null ? sameBrand(топ1.classIdx, топ2.classIdx) : false,
      },
      THRESHOLDS,
      CALIBRATION,
    )
    подписи[номер] = { id: трек.id, box: трек.box, decision: решение, tops: трек.tops }
    for (let к = 0; к < события.length; к++) {
      if (события[к]!.id === трек.id) события[к]!.decisionKey = decisionKey(решение)
    }
  }

  return { состояние: { tracks: треки, nextId: итогТрекера.следующийId }, подписи, события }
}
