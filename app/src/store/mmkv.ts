import { MMKV } from 'react-native-mmkv'
import { NativeModules } from 'react-native'
import type { Хранилище } from './storage'

type ОтветМоста = { ключ: string; сброшен: boolean }

/**
 * Единственное место в проекте, знающее про `react-native-mmkv`. Ключ приходит из
 * `AndroidKeyStore` через нативный модуль `CarlensStorage` (см. `StorageKey.kt`), а не
 * генерируется здесь: JS-случайность пришлось бы где-то хранить, и хранить её было бы
 * негде, кроме того же диска.
 */
export async function создатьХранилище(): Promise<{ хранилище: Хранилище; сброшен: boolean }> {
  const мост = NativeModules.CarlensStorage as { initStorage(): Promise<ОтветМоста> } | undefined
  if (мост == null) {
    throw new Error('нативный модуль CarlensStorage не зарегистрирован')
  }
  const ответ = await мост.initStorage()
  const mmkv = new MMKV({ id: 'carlens', encryptionKey: ответ.ключ })
  const хранилище: Хранилище = {
    прочитать: (ключ) => mmkv.getString(ключ) ?? null,
    записать: (ключ, значение) => mmkv.set(ключ, значение),
    удалить: (ключ) => mmkv.delete(ключ),
  }
  return { хранилище, сброшен: ответ.сброшен }
}
