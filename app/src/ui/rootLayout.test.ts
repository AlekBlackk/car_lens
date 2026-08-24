import { parse } from '@babel/parser'
import traverse from '@babel/traverse'
import { readFileSync } from 'node:fs'
import { describe, expect, test } from 'vitest'

/**
 * Стража против регрессии, стоившей приёмки M-1.6 на устройстве: корневой `View`,
 * в котором лежит `CameraScreen`, не должен центрировать детей. `alignItems: 'center'`
 * заменяет ребёнку растяжение на `align-self`, тот берёт ширину по содержимому, а всё
 * содержимое экрана камеры абсолютное — ширина выходит нулевой. Видно это было не как
 * пустой экран, а как вертикальные полосы: текст переносился по одному символу.
 *
 * Проверка статическая, потому что рендерить в этом проекте нечем: тесты гоняются в Node
 * без react-test-renderer. Тот же приём, что у `workletClosure.test.ts`, — разобрать
 * настоящий исходник и спросить его о факте, а не о намерении.
 */
const ФАЙЛ = 'App.tsx'

const ЗАПРЕЩЁННЫЕ = ['alignItems', 'justifyContent']

function разобрать(файл: string) {
  return parse(readFileSync(файл, 'utf8'), {
    sourceType: 'module',
    plugins: ['typescript', 'jsx'],
  })
}

/** Имя стиля у `<View style={styles.X}>`, внутри которого стоит `<CameraScreen>`. */
function стильКорня(файл: string): string | null {
  const дерево = разобрать(файл)
  let имя: string | null = null
  traverse(дерево, {
    JSXElement(путь) {
      const открывающий = путь.node.openingElement
      if (открывающий.name.type !== 'JSXIdentifier' || открывающий.name.name !== 'View') return
      const естьКамера = путь.node.children.some(
        (ребёнок) =>
          ребёнок.type === 'JSXElement' &&
          ребёнок.openingElement.name.type === 'JSXIdentifier' &&
          ребёнок.openingElement.name.name === 'CameraScreen',
      )
      if (!естьКамера) return
      for (const атрибут of открывающий.attributes) {
        if (атрибут.type !== 'JSXAttribute' || атрибут.name.name !== 'style') continue
        const значение = атрибут.value
        if (
          значение?.type === 'JSXExpressionContainer' &&
          значение.expression.type === 'MemberExpression' &&
          значение.expression.property.type === 'Identifier'
        ) {
          имя = значение.expression.property.name
        }
      }
    },
  })
  return имя
}

/** Ключи стиля с указанным именем из `StyleSheet.create({...})`. */
function ключиСтиля(файл: string, имя: string): string[] {
  const дерево = разобрать(файл)
  const ключи: string[] = []
  traverse(дерево, {
    ObjectProperty(путь) {
      const ключ = путь.node.key
      const подходит =
        (ключ.type === 'Identifier' && ключ.name === имя) ||
        (ключ.type === 'StringLiteral' && ключ.value === имя)
      if (!подходит || путь.node.value.type !== 'ObjectExpression') return
      for (const поле of путь.node.value.properties) {
        if (поле.type === 'ObjectProperty' && поле.key.type === 'Identifier') {
          ключи.push(поле.key.name)
        }
      }
    },
  })
  return ключи
}

describe('корень приложения не центрирует экраны', () => {
  test('корневой View найден и у него есть именованный стиль', () => {
    expect(стильКорня(ФАЙЛ)).not.toBeNull()
  })

  test('стиль корня не центрирует детей', () => {
    const имя = стильКорня(ФАЙЛ)!
    const ключи = ключиСтиля(ФАЙЛ, имя)
    expect(ключи).not.toHaveLength(0)
    for (const запрещённый of ЗАПРЕЩЁННЫЕ) {
      expect(ключи).not.toContain(запрещённый)
    }
  })

  test('методика: центрирующий стиль boot-состояний тест бы поймал', () => {
    expect(ключиСтиля(ФАЙЛ, 'экран')).toContain('alignItems')
  })
})
