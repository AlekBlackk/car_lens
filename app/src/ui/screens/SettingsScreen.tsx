/**
 * Экран «Настройки» §6.6: порог подписи, два переключателя, версии приложения и каталога,
 * вход в список моделей каталога.
 *
 * Компонент ничего не пишет на диск. Наружу уходит целиком новый объект настроек, а запись,
 * как и чистка очереди отзывов при выключении сбора, живёт в корне приложения — там же, где
 * хранилище. Так у настроек один писатель, а не два.
 */
import React from 'react'
import { NativeModules, Pressable, ScrollView, StyleSheet, Switch, Text, View } from 'react-native'
import { версияПриложения, type КонстантыВерсии } from '../../appInfo'
import { МАКСИМУМ_ПОРОГА, type Настройки } from '../../store/settings'
import { catalogInfo } from '../../vision/labels'

export type СвойстваНастроек = {
  настройки: Настройки
  наИзменение: (настройки: Настройки) => void
  наКаталог: () => void
  наЗакрытие: () => void
}

/**
 * Ползунка в react-native нет, а `@react-native-community/slider` — новая зависимость ради
 * одного экрана. Шаг 0.1 от нуля до потолка настроек; потолок не единица, потому что порог
 * 1.0 отключил бы подписи полностью (см. KDoc `МАКСИМУМ_ПОРОГА`).
 */
const ШАГИ = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, МАКСИМУМ_ПОРОГА]

/** Сравнение долей после умножения на 100: `0.1 * 3` не равно `0.3` в двоичной плавающей. */
const тотЖеШаг = (первый: number, второй: number): boolean =>
  Math.round(первый * 100) === Math.round(второй * 100)

export default function SettingsScreen(свойства: СвойстваНастроек): React.JSX.Element {
  const { настройки, наИзменение, наКаталог, наЗакрытие } = свойства
  const инфо = catalogInfo()

  return (
    <ScrollView style={styles.экран} contentContainerStyle={styles.содержимое}>
      <Text style={styles.заголовок}>Настройки</Text>

      <Text style={styles.раздел}>Порог подписи</Text>
      <View style={styles.рядШагов}>
        {ШАГИ.map((шаг) => {
          const выбран = тотЖеШаг(шаг, настройки.порогУверенности)
          return (
            <Pressable
              key={шаг}
              style={[styles.шаг, выбран && styles.шагВыбран]}
              onPress={() => наИзменение({ ...настройки, порогУверенности: шаг })}
            >
              <Text style={styles.текстШага}>{Math.round(шаг * 100)}</Text>
            </Pressable>
          )
        })}
      </View>
      <Text style={styles.пояснение}>
        0 — показывать всё, что прошло пороги отказа. Выше — подпись скрывается, пока
        уверенность не дотянет до выбранного процента.
      </Text>

      <View style={styles.строкаТумблера}>
        <Text style={styles.текст}>Показывать топ-3 в «Деталях»</Text>
        <Switch
          value={настройки.показыватьТоп3}
          onValueChange={(включено) => наИзменение({ ...настройки, показыватьТоп3: включено })}
        />
      </View>

      <View style={styles.строкаТумблера}>
        <Text style={styles.текст}>Собирать жалобы на подписи</Text>
        <Switch
          value={настройки.сборОбратнойСвязи}
          onValueChange={(включено) => наИзменение({ ...настройки, сборОбратнойСвязи: включено })}
        />
      </View>
      <Text style={styles.пояснение}>
        §8: сбор выключен по умолчанию и включается только вручную. Данные не покидают
        телефон; при выключении уже собранная очередь стирается.
      </Text>

      <Text style={styles.раздел}>Приложение</Text>
      <Text style={styles.пояснение}>
        версия {версияПриложения(NativeModules.CarlensAppInfo as КонстантыВерсии | undefined)}
      </Text>

      <Text style={styles.раздел}>Каталог</Text>
      {/* Показ, а не настройка: отдельного поля «версия каталога» в ассете нет. */}
      <Text style={styles.пояснение}>
        format_version {инфо.formatVersion} · {инфо.классов} класса · веса{' '}
        {инфо.weightsSha.slice(0, 12)} · собран {инфо.generatedAt}
      </Text>
      <Pressable onPress={наКаталог}>
        <Text style={styles.ссылка}>Каталог машин ({инфо.классов}) →</Text>
      </Pressable>

      <Pressable onPress={наЗакрытие}>
        <Text style={styles.кнопка}>Назад</Text>
      </Pressable>
    </ScrollView>
  )
}

const styles = StyleSheet.create({
  экран: { flex: 1, backgroundColor: '#000' },
  содержимое: { paddingHorizontal: 20, paddingVertical: 24 },
  заголовок: { color: '#fff', fontSize: 22, marginBottom: 16 },
  раздел: { color: '#fff', fontSize: 17, marginTop: 8, marginBottom: 12 },
  рядШагов: { flexDirection: 'row', flexWrap: 'wrap' },
  шаг: {
    minWidth: 44,
    alignItems: 'center',
    paddingVertical: 8,
    paddingHorizontal: 10,
    marginRight: 8,
    marginBottom: 8,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#3a3a3a',
  },
  шагВыбран: { backgroundColor: '#1f4b7a', borderColor: '#4da3ff' },
  текстШага: { color: '#fff', fontSize: 16 },
  строкаТумблера: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: 8,
  },
  текст: { color: '#fff', fontSize: 17, flexShrink: 1, paddingRight: 12 },
  пояснение: { color: '#9a9a9a', fontSize: 15, marginTop: 8, marginBottom: 8, lineHeight: 21 },
  ссылка: { color: '#4da3ff', fontSize: 17, paddingVertical: 8 },
  кнопка: { color: '#4da3ff', fontSize: 18, marginTop: 24 },
})
