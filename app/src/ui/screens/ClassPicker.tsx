/**
 * Выбор правильной модели для жалобы «это неверно» (§8).
 *
 * Список каталога целиком — 171 класс на M-1, и до 1500 по дорожной карте §10, — поэтому
 * фильтр по подстроке здесь не удобство, а условие пользуемости: пролистать столько строк
 * пальцем нельзя.
 *
 * Наружу уходит `classIdx`, то есть индекс в каталоге, а не имя: именно им оперируют
 * `ЗаписьОтзыва` и мост. Индекс берётся из позиции в `allClasses()` — это тот же массив,
 * по которому `classLabel` разворачивает подписи, поэтому отбор идёт по копии пар
 * «класс + индекс», а не сортировкой на месте.
 */
import React, { useMemo, useState } from 'react'
import { FlatList, Pressable, StyleSheet, Text, TextInput, View } from 'react-native'
import { allClasses } from '../../vision/labels'

export type СвойстваВыбораКласса = {
  наВыбор: (classIdx: number) => void
  наОтмену: () => void
}

export default function ClassPicker(свойства: СвойстваВыбораКласса): React.JSX.Element {
  const { наВыбор, наОтмену } = свойства
  const [запрос, установитьЗапрос] = useState('')

  const отобранные = useMemo(() => {
    const строка = запрос.trim().toLowerCase()
    const все = allClasses()
    const пары = все.map((класс, индекс) => ({ класс, индекс }))
    if (строка.length === 0) return пары
    // Поиск идёт и по марке: пользователь, не знающий модель, ищет «toyota», а не «camry».
    // У разнобрендовой группы марки нет вовсе (§13.2), и найти её по марке участника можно
    // только через `members`: `member_id` вида «belgee__x50» несёт и марку, и модель.
    return пары.filter(
      ({ класс }) =>
        класс.display.toLowerCase().includes(строка) ||
        (класс.brand?.toLowerCase().includes(строка) ?? false) ||
        класс.members.some((участник) => участник.toLowerCase().includes(строка)),
    )
  }, [запрос])

  return (
    <View style={styles.экран}>
      <Text style={styles.заголовок}>Что на самом деле?</Text>
      <TextInput
        style={styles.поиск}
        value={запрос}
        onChangeText={установитьЗапрос}
        placeholder="Поиск по модели или бренду"
        placeholderTextColor="#8a8a8a"
        autoCorrect={false}
        autoCapitalize="none"
      />
      {отобранные.length === 0 ? (
        <Text style={styles.пусто}>Ничего не найдено</Text>
      ) : (
        <FlatList
          data={отобранные}
          keyExtractor={(запись) => String(запись.индекс)}
          renderItem={({ item }) => (
            <Pressable style={styles.строка} onPress={() => наВыбор(item.индекс)}>
              <Text style={styles.текст}>{item.класс.display}</Text>
              {/* У группы марки нет, а `display` уже называет обе: прочерк, как в деталях. */}
              <Text style={styles.бренд}>{item.класс.brand ?? '—'}</Text>
            </Pressable>
          )}
        />
      )}
      <Pressable onPress={наОтмену}>
        <Text style={styles.кнопка}>Отмена</Text>
      </Pressable>
    </View>
  )
}

const styles = StyleSheet.create({
  экран: { flex: 1, backgroundColor: '#000', paddingHorizontal: 20, paddingVertical: 24 },
  заголовок: { color: '#fff', fontSize: 20, marginBottom: 16 },
  поиск: {
    color: '#fff',
    fontSize: 17,
    borderColor: '#3a3a3a',
    borderWidth: 1,
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
    marginBottom: 16,
  },
  строка: { paddingVertical: 12, borderBottomColor: '#242424', borderBottomWidth: 1 },
  текст: { color: '#fff', fontSize: 17 },
  бренд: { color: '#9a9a9a', fontSize: 14, marginTop: 2 },
  пусто: { color: '#9a9a9a', fontSize: 16, flex: 1 },
  кнопка: { color: '#4da3ff', fontSize: 18, marginTop: 16 },
})
