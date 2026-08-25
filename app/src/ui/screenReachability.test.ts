import { parse } from '@babel/parser'
import traverse from '@babel/traverse'
import { readFileSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
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

describe('достижимость экранов §6.6', () => {
  it('на каждый экран, кроме дна стека, есть переход в исходниках', () => {
    const переходы = видыПереходов()
    const недостижимые = видыЭкранов().filter((вид) => вид !== ДНО && !переходы.has(вид))
    expect(недостижимые).toEqual([])
  })

  it('виды экранов вообще нашлись — тест не проходит на пустом разборе', () => {
    expect(видыЭкранов().length).toBeGreaterThan(1)
  })
})
