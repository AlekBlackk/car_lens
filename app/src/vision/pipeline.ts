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

export type Трек = {
  id: number
  бокс: Box
  наблюдений: number
  виденВ: number
  классифицированВ: number
  ema: number[] | null
  топ1: Match | null
  топ2: Match | null
}

export type СостояниеКонвейера = { треки: Трек[]; следующийId: number }

export type ПодписьТрека = { id: number; бокс: Box; решение: Решение }

/** То, что читает счётчик стабильности: оба топ-1 одного и того же наблюдения плюс ключ решения. */
export type СобытиеКлассификации = {
  id: number
  top1Мгновенный: number
  top1Ema: number
  ключРешения: string
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

export const ПУСТОЕ_СОСТОЯНИЕ: СостояниеКонвейера = { треки: [], следующийId: 1 }

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

  const геометрия: ТрекГеометрия[] = new Array(состояние.треки.length)
  for (let номер = 0; номер < состояние.треки.length; номер++) {
    const трек = состояние.треки[номер]!
    геометрия[номер] = { id: трек.id, бокс: трек.бокс, виденВ: трек.виденВ }
  }
  const итогТрекера = updateTracks(геометрия, вход.боксы, nowMs, состояние.следующийId)

  // Новый массив треков: входное состояние не мутируется, поэтому «шаг не испортил вход»
  // проверяется тестом в две строки.
  const треки: Трек[] = []
  for (let номер = 0; номер < состояние.треки.length; номер++) {
    const прежний = состояние.треки[номер]!
    let мёртв = false
    for (let к = 0; к < итогТрекера.умершие.length; к++) {
      if (итогТрекера.умершие[к] === прежний.id) {
        мёртв = true
        break
      }
    }
    if (мёртв) continue
    let бокс = прежний.бокс
    let виденВ = прежний.виденВ
    for (let к = 0; к < итогТрекера.сопоставленные.length; к++) {
      if (итогТрекера.сопоставленные[к]!.id === прежний.id) {
        бокс = итогТрекера.сопоставленные[к]!.бокс
        виденВ = nowMs
        break
      }
    }
    треки.push({
      id: прежний.id,
      бокс,
      виденВ,
      наблюдений: прежний.наблюдений,
      классифицированВ: прежний.классифицированВ,
      ema: прежний.ema,
      топ1: прежний.топ1,
      топ2: прежний.топ2,
    })
  }
  for (let к = 0; к < итогТрекера.рождённые.length; к++) {
    const рождённый = итогТрекера.рождённые[к]!
    треки.push({
      id: рождённый.id,
      бокс: рождённый.бокс,
      виденВ: nowMs,
      наблюдений: 0,
      // Родившийся трек классифицируется в том же кадре: интервал считается уже истёкшим.
      // Так, а не `0` и не `-Infinity`, потому что `nowMs` — время кадра, и оно может
      // начинаться с любого значения.
      классифицированВ: nowMs - CLASSIFY_INTERVAL_MS,
      ema: null,
      топ1: null,
      топ2: null,
    })
  }

  const кандидаты = pickForClassify(треки, nowMs, THRESHOLDS.minShortSidePx)
  const события: СобытиеКлассификации[] = []

  if (кандидаты.length > 0) {
    const боксы: Box[] = new Array(кандидаты.length)
    for (let к = 0; к < кандидаты.length; к++) боксы[к] = кандидаты[к]!.бокс
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
      трек.наблюдений = трек.наблюдений + 1
      трек.классифицированВ = nowMs
      const поEma = мост.matchEma(toFloat32(шаг.вектор!))
      трек.топ1 = поEma.length > 0 ? поEma[0]! : null
      трек.топ2 = поEma.length > 1 ? поEma[1]! : null
      const мгновенные = результаты[к]!.matches
      события.push({
        id,
        top1Мгновенный: мгновенные.length > 0 ? мгновенные[0]!.classIdx : -1,
        top1Ema: трек.топ1 != null ? трек.топ1.classIdx : -1,
        ключРешения: '',
      })
    }
  }

  const подписи: ПодписьТрека[] = new Array(треки.length)
  for (let номер = 0; номер < треки.length; номер++) {
    const трек = треки[номер]!
    const топ1 = трек.топ1
    const топ2 = трек.топ2
    const решение = decide(
      {
        наблюдений: трек.наблюдений,
        короткаяСторона: Math.min(трек.бокс.w, трек.бокс.h),
        топ1,
        топ2,
        брендыСовпали: топ1 != null && топ2 != null ? sameBrand(топ1.classIdx, топ2.classIdx) : false,
      },
      THRESHOLDS,
      CALIBRATION,
    )
    подписи[номер] = { id: трек.id, бокс: трек.бокс, решение }
    for (let к = 0; к < события.length; к++) {
      if (события[к]!.id === трек.id) события[к]!.ключРешения = decisionKey(решение)
    }
  }

  return { состояние: { треки, следующийId: итогТрекера.следующийId }, подписи, события }
}
