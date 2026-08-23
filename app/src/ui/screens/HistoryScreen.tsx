/**
 * Экран «История» §6.6: последние 50 распознаваний, только текст, без изображений.
 *
 * Строки не нажимаются, и это не упущение: запись не хранит ни id живого трека, ни гипотез —
 * открывать по ней нечего. «Детали» достижимы только с замороженного кадра.
 */
import React from 'react'
import { FlatList, Pressable, StyleSheet, Text, View } from 'react-native'
import { своя, type ЗаписьИстории } from '../../store/history'
import { classLabel } from '../../vision/labels'

export type СвойстваИстории = {
  история: ЗаписьИстории[]
  сброшена: boolean
  наЗакрытие: () => void
}

/** Время записи — стенные часы (`Date.now()`), поэтому показывается местное время телефона. */
const времяСтрокой = (время: number): string => {
  const дата = new Date(время)
  const двузначно = (число: number): string => String(число).padStart(2, '0')
  return `${двузначно(дата.getDate())}.${двузначно(дата.getMonth() + 1)} ${двузначно(дата.getHours())}:${двузначно(дата.getMinutes())}`
}

/**
 * Запись чужого каталога в имя не разворачивается: тот же `classIdx` там означает другой
 * автомобиль. Стирать её не надо — время и факт распознавания остаются правдой.
 */
const имяЗаписи = (запись: ЗаписьИстории): string =>
  своя(запись)
    ? (classLabel(запись.classIdx)?.display ?? `класс ${запись.classIdx}`)
    : `класс ${запись.classIdx} · другая версия каталога`

const процентомИлиТире = (запись: ЗаписьИстории): string => {
  // У ветви «бренд» уверенности нет вовсе (§6.4), а не «ноль процентов».
  if (запись.уверенность === null) return 'бренд'
  const процент = Math.round(запись.уверенность * 100)
  return запись.откалибровано ? `${процент}%` : `${процент}% (не откалибровано)`
}

export default function HistoryScreen(свойства: СвойстваИстории): React.JSX.Element {
  const { история, сброшена, наЗакрытие } = свойства

  return (
    <View style={styles.экран}>
      <Text style={styles.заголовок}>История</Text>

      {сброшена && (
        // Единственный способ узнать про утрату ключа Keystore. Молчать о ней нельзя:
        // пользователь иначе решит, что история пропала сама.
        <Text style={styles.тревога}>Данные не открылись, история очищена.</Text>
      )}

      {история.length === 0 ? (
        <Text style={styles.пояснение}>Пока пусто. Наведите камеру на автомобиль.</Text>
      ) : (
        <FlatList
          data={история}
          keyExtractor={(запись) => `${запись.время}-${запись.classIdx}`}
          renderItem={({ item }) => (
            <View style={styles.строка}>
              <Text style={styles.имя}>{имяЗаписи(item)}</Text>
              <Text style={styles.мелкое}>
                {времяСтрокой(item.время)} · {процентомИлиТире(item)}
              </Text>
            </View>
          )}
        />
      )}

      <Pressable onPress={наЗакрытие}>
        <Text style={styles.кнопка}>Назад</Text>
      </Pressable>
    </View>
  )
}

const styles = StyleSheet.create({
  экран: { flex: 1, backgroundColor: '#000', paddingHorizontal: 20, paddingVertical: 24 },
  заголовок: { color: '#fff', fontSize: 22, marginBottom: 16 },
  строка: { paddingVertical: 10, borderBottomColor: '#242424', borderBottomWidth: 1 },
  имя: { color: '#fff', fontSize: 17 },
  мелкое: { color: '#9a9a9a', fontSize: 14, marginTop: 2 },
  пояснение: { color: '#9a9a9a', fontSize: 15, flex: 1 },
  тревога: { color: '#ffb020', fontSize: 15, marginBottom: 16, lineHeight: 21 },
  кнопка: { color: '#4da3ff', fontSize: 18, marginTop: 16 },
})
