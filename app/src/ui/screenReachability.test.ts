import { parse } from '@babel/parser'
import traverse from '@babel/traverse'
import { readFileSync, readdirSync } from 'node:fs'
import { basename, join } from 'node:path'
import { describe, expect, it } from 'vitest'

/**
 * Стража против дыры, найденной на приёмке §10 M-1.6: экран «История» существовал целиком —
 * компонент, хранилище, тесты, ветка в `App.tsx`, — но открыть его было нечем. Ни одного
 * `push` с этим видом во всём дереве, то есть §6.6 обещал четыре экрана, а пользователю их
 * было доступно три.
 *
 * Проверка статическая по той же причине, что в `rootLayout.test.ts`: рендерить в этом
 * проекте нечем, тесты гоняются в Node без react-test-renderer. Спрашиваем исходники о
 * факте — какие виды экранов упоминаются в аргументах `push`, — а не о намерении.
 *
 * `камера` из требования исключена намеренно: это дно стека, оно ставится
 * `НАЧАЛЬНАЯ_НАВИГАЦИЯ`, а не переходом, и `push` для него не нужен.
 */
const ДНО = 'камера'

const разобрать = (файл: string) =>
  parse(readFileSync(файл, 'utf8'), { sourceType: 'module', plugins: ['typescript', 'jsx'] })

/** Виды из объявления `type Экран` — источник истины, а не список, переписанный в тест. */
function видыЭкранов(): string[] {
  const дерево = разобрать('src/ui/navigation.ts')
  const виды: string[] = []
  traverse(дерево, {
    TSTypeAliasDeclaration(путь) {
      if (путь.node.id.name !== 'Экран') return
      const тип = путь.node.typeAnnotation
      const члены = тип.type === 'TSUnionType' ? тип.types : [тип]
      for (const член of члены) {
        if (член.type !== 'TSTypeLiteral') continue
        for (const поле of член.members) {
          if (поле.type !== 'TSPropertySignature') continue
          if (поле.key.type !== 'Identifier' || поле.key.name !== 'вид') continue
          const значение = поле.typeAnnotation?.typeAnnotation
          if (значение?.type === 'TSLiteralType' && значение.literal.type === 'StringLiteral') {
            виды.push(значение.literal.value)
          }
        }
      }
    },
  })
  return виды
}

/** Все `.ts`/`.tsx` дерева приложения плюс корневой `App.tsx`. */
function исходники(каталог: string): string[] {
  const найденные: string[] = []
  for (const запись of readdirSync(каталог, { withFileTypes: true })) {
    const путь = join(каталог, запись.name)
    if (запись.isDirectory()) найденные.push(...исходники(путь))
    else if (/\.tsx?$/.test(запись.name) && !запись.name.includes('.test.')) найденные.push(путь)
  }
  return найденные
}

/** Виды, встреченные литералом `{ вид: '...' }` внутри вызова `push(...)`. */
function видыПереходов(): Set<string> {
  const найденные = new Set<string>()
  for (const файл of [...исходники('src'), 'App.tsx']) {
    traverse(разобрать(файл), {
      CallExpression(путь) {
        const вызываемое = путь.node.callee
        if (вызываемое.type !== 'Identifier' || вызываемое.name !== 'push') return
        for (const аргумент of путь.node.arguments) {
          if (аргумент.type !== 'ObjectExpression') continue
          for (const свойство of аргумент.properties) {
            if (свойство.type !== 'ObjectProperty') continue
            if (свойство.key.type !== 'Identifier' || свойство.key.name !== 'вид') continue
            if (свойство.value.type === 'StringLiteral') найденные.add(свойство.value.value)
          }
        }
      },
    })
  }
  return найденные
}

/** Читает ли узел поле объекта `настройки` — `настройки.сборОбратнойСвязи` и подобные. */
const читаетПолеНастроек = (узел: { type: string; object?: unknown }): boolean => {
  if (узел.type !== 'MemberExpression') return false
  const объект = узел.object as { type: string; name?: string }
  return объект.type === 'Identifier' && объект.name === 'настройки'
}

/**
 * Колбэки экранов, все вызовы которых заперты значением настройки.
 *
 * Дыра, найденная на приёмке §10 2026-08-25: `push({ вид: 'настройки' })` в `App.tsx` есть
 * и безусловен, поэтому первая стража его пропускала. Заперт был вызов колбэка —
 * `настройки.сборОбратнойСвязи ? установитьСписок(true) : наНастройки()` в `DetailsScreen`.
 * Включив сбор жалоб, пользователь терял Настройки насовсем: ни порог сменить, ни сбор
 * выключить, выход только через стирание данных.
 *
 * Считаем колбэк запертым, если каждый его вызов в экранах стоит внутри тернарника, чьё
 * условие читает поле объекта `настройки`. Хоть один вызов вне такого тернарника — колбэк
 * свободен, и заперт он больше не считается.
 *
 * Ключ — пара «экран.колбэк», а не одно имя: `наНастройки` заперт в `DetailsScreen` и
 * свободен на `CameraScreen`, и без экрана в ключе починка выглядела бы как всё тот же
 * дефект.
 */
function запертыеКолбэки(): Set<string> {
  const заперты = new Map<string, boolean>()
  for (const файл of исходники(join('src', 'ui', 'screens'))) {
    const экран = basename(файл, '.tsx')
    traverse(разобрать(файл), {
      CallExpression(путь) {
        const вызываемое = путь.node.callee
        if (вызываемое.type !== 'Identifier') return
        const имя = вызываемое.name
        const заперт = путь.findParent((предок) => {
          if (!предок.isConditionalExpression()) return false
          const условие = предок.get('test')
          let читаетНастройку = читаетПолеНастроек(условие.node)
          // `traverse` обходит потомков, но не сам узел, а условие сплошь и рядом само и
          // есть `настройки.сборОбратнойСвязи` — узел проверяется отдельной строкой выше.
          условие.traverse({
            MemberExpression(вложенное) {
              if (читаетПолеНастроек(вложенное.node)) читаетНастройку = true
            },
          })
          return читаетНастройку
        })
        const ключ = `${экран}.${имя}`
        заперты.set(ключ, (заперты.get(ключ) ?? true) && заперт != null)
      },
    })
  }
  return new Set([...заперты].filter(([, заперт]) => заперт).map(([имя]) => имя))
}

/**
 * Виды экранов, каждый переход к которым привязан в `App.tsx` к запертому колбэку.
 *
 * Свободный переход у вида хотя бы один — вид достижим при любых настройках, и в результат
 * он не попадает.
 */
function условноДостижимые(): string[] {
  const заперты = запертыеКолбэки()
  const свободные = new Set<string>()
  const подЗамком = new Set<string>()
  traverse(разобрать('App.tsx'), {
    CallExpression(путь) {
      const вызываемое = путь.node.callee
      if (вызываемое.type !== 'Identifier' || вызываемое.name !== 'push') return
      const свойство = путь.findParent((предок) => {
        if (!предок.isJSXAttribute()) return false
        if (предок.node.name.type !== 'JSXIdentifier') return false
        const элемент = предок.parentPath
        if (!элемент.isJSXOpeningElement()) return false
        const имяЭлемента = элемент.node.name
        if (имяЭлемента.type !== 'JSXIdentifier') return false
        return заперты.has(`${имяЭлемента.name}.${предок.node.name.name}`)
      })
      for (const аргумент of путь.node.arguments) {
        if (аргумент.type !== 'ObjectExpression') continue
        for (const поле of аргумент.properties) {
          if (поле.type !== 'ObjectProperty') continue
          if (поле.key.type !== 'Identifier' || поле.key.name !== 'вид') continue
          if (поле.value.type !== 'StringLiteral') continue
          ;(свойство == null ? свободные : подЗамком).add(поле.value.value)
        }
      }
    },
  })
  return [...подЗамком].filter((вид) => !свободные.has(вид))
}

describe('достижимость экранов §6.6', () => {
  it('на каждый экран, кроме дна стека, есть переход в исходниках', () => {
    const переходы = видыПереходов()
    const недостижимые = видыЭкранов().filter((вид) => вид !== ДНО && !переходы.has(вид))
    expect(недостижимые).toEqual([])
  })

  it('ни один экран не заперт значением настройки', () => {
    expect(условноДостижимые()).toEqual([])
  })

  it('виды экранов вообще нашлись — тест не проходит на пустом разборе', () => {
    expect(видыЭкранов().length).toBeGreaterThan(1)
  })

  it('анализ запертых колбэков что-то находит — иначе предыдущий тест зелен впустую', () => {
    // `DetailsScreen.наНастройки` останется запертым и после правки: там переход живёт
    // внутри тернарника по `настройки.сборОбратнойСвязи` и по §5.3 так и задуман. Проверка
    // ловит поломку самого разбора, при которой множество запертых схлопнулось бы в пустое.
    expect(запертыеКолбэки().size).toBeGreaterThan(0)
  })
})
