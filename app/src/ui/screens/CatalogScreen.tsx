/**
 * Экран «Каталог» §6.6: какие модели приложение умеет распознавать. Только просмотр —
 * по маркам, с поиском; ничего не пишет и не хранит.
 *
 * Показывается каталог классов, а не моделей: пять групп (§13.2) объединяют внешне
 * неотличимые машины разных марок, и приложение подписывает их одним названием.
 */
import React, { useMemo, useState } from 'react'
import { Pressable, SectionList, StyleSheet, Text, TextInput, View } from 'react-native'
import { allClasses } from '../../vision/labels'
import { РАЗДЕЛ_БЕЗ_МАРКИ, разделыКаталога } from '../catalogList'

export type СвойстваКаталога = {
  наЗакрытие: () => void
}

export default function CatalogScreen(свойства: СвойстваКаталога): React.JSX.Element {
  const { наЗакрытие } = свойства
  const [запрос, установитьЗапрос] = useState('')

  const классов = allClasses().length
  const разделы = useMemo(() => разделыКаталога(allClasses(), запрос), [запрос])

  return (
    <View style={styles.экран}>
      <Text style={styles.заголовок}>Каталог</Text>
      <TextInput
        style={styles.поиск}
        value={запрос}
        onChangeText={установитьЗапрос}
        placeholder="Поиск по модели или бренду"
        placeholderTextColor="#8a8a8a"
        autoCorrect={false}
        autoCapitalize="none"
      />
      {разделы.length === 0 ? (
        <Text style={styles.пусто}>Ничего не найдено</Text>
      ) : (
        <SectionList
          sections={разделы}
          keyExtractor={(класс) => класс.classId}
          stickySectionHeadersEnabled
          ListHeaderComponent={<Text style={styles.пояснение}>Всего в каталоге: {классов}</Text>}
          renderSectionHeader={({ section }) => (
            <View style={styles.шапкаРаздела}>
              <Text style={styles.названиеРаздела}>
                {section.название} · {section.data.length}
              </Text>
              {section.название === РАЗДЕЛ_БЕЗ_МАРКИ && (
                <Text style={styles.мелкое}>
                  Внешне неотличимые модели разных марок — приложение называет их вместе.
                </Text>
              )}
            </View>
          )}
          renderItem={({ item }) => (
            <View style={styles.строка}>
              <Text style={styles.текст}>{item.display}</Text>
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
  поиск: {
    color: '#fff',
    fontSize: 17,
    borderColor: '#3a3a3a',
    borderWidth: 1,
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
    marginBottom: 12,
  },
  пояснение: { color: '#9a9a9a', fontSize: 15, marginBottom: 8 },
  // Фон у закреплённой шапки обязателен: без него строки списка просвечивают сквозь неё.
  шапкаРаздела: { backgroundColor: '#000', paddingTop: 12, paddingBottom: 6 },
  названиеРаздела: { color: '#4da3ff', fontSize: 16 },
  мелкое: { color: '#9a9a9a', fontSize: 14, marginTop: 2, lineHeight: 20 },
  строка: { paddingVertical: 10, borderBottomColor: '#242424', borderBottomWidth: 1 },
  текст: { color: '#fff', fontSize: 17 },
  пусто: { color: '#9a9a9a', fontSize: 16, flex: 1 },
  кнопка: { color: '#4da3ff', fontSize: 18, marginTop: 16 },
})
