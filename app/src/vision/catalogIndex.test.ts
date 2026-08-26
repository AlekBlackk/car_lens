import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
// Явный импорт URL из node:url — глобальный `URL` в типах React Native (см.
// @types/react-native) несовместим с сигнатурой node:fs.readFileSync(URL), а
// import.meta.url в рантайме даёт настоящий URL из движка node/vitest.
import { URL } from 'node:url'

import { expect, test } from 'vitest'

import { IndexFormatError, loadIndex } from './catalogIndex.js'

const sha256 = (data: Uint8Array): string => createHash('sha256').update(data).digest('hex')

const fixture = (name: string): Uint8Array =>
  new Uint8Array(readFileSync(new URL(`../../../tests/fixtures/index/${name}`, import.meta.url)))

const bin = (): Uint8Array => fixture('catalog_index.bin')
const meta = (): Uint8Array => fixture('catalog_meta.json')

test('разбирает заголовок фикстуры', () => {
  const index = loadIndex(bin(), meta(), sha256)
  expect(index.header.formatVersion).toBe(1)
  expect(index.header.numClasses).toBe(2)
  expect(index.header.centroidsPerClass).toBe(4)
  expect(index.header.embeddingDim).toBe(256)
  expect(index.header.dtype).toBe(1)
  expect(index.header.normalized).toBe(true)
})

test('классы приходят в порядке индексов', () => {
  const index = loadIndex(bin(), meta(), sha256)
  expect(index.classes.length).toBe(2)
  expect(index.classes[0]?.class_id).toBe('acme__one')
  expect(index.classes[0]?.members).toEqual(['acme__one'])
  expect(index.classes[0]?.brand).toBe('Acme')
  expect(index.classes[1]?.display).toBe('Acme Two / Beta Two')
})

test('у разнобрендовой группы марки нет, а участники на месте', () => {
  const group = loadIndex(bin(), meta(), sha256).classes[1]
  expect(group?.class_id).toBe('grp__twins')
  expect(group?.members).toEqual(['acme__two', 'beta__two'])
  expect(group?.brand).toBeNull()
})

test('центроиды нормализованы и не копируются лишний раз', () => {
  const index = loadIndex(bin(), meta(), sha256)
  for (let c = 0; c < index.header.numClasses; c += 1) {
    for (let k = 0; k < index.header.centroidsPerClass; k += 1) {
      const vector = index.centroid(c, k)
      expect(vector.length).toBe(256)
      const norm = Math.hypot(...vector)
      expect(Math.abs(norm - 1) < 1e-5).toBe(true)
    }
  }
})

test('match находит класс собственного центроида', () => {
  const index = loadIndex(bin(), meta(), sha256)
  const target = index.centroid(1, 2)
  const found = index.match(Float32Array.from(target))
  expect(found.classIdx).toBe(1)
  expect(found.sim > 0.999).toBe(true)
})

test('centroid отвергает индексы вне диапазона', () => {
  const index = loadIndex(bin(), meta(), sha256)
  expect(() => index.centroid(2, 0)).toThrow(IndexFormatError)
  expect(() => index.centroid(0, 4)).toThrow(IndexFormatError)
})

test('match отвергает вектор чужой размерности', () => {
  const index = loadIndex(bin(), meta(), sha256)
  expect(() => index.match(new Float32Array(8))).toThrow(IndexFormatError)
})

const damaged = (patch: (bytes: Uint8Array) => void): Uint8Array => {
  const bytes = bin()
  patch(bytes)
  return bytes
}

test('чужой magic отвергается', () => {
  const bytes = damaged((b) => {
    b[0] = 'X'.charCodeAt(0)
  })
  expect(() => loadIndex(bytes, meta(), sha256)).toThrow(/magic/)
})

test('чужая версия формата отвергается', () => {
  const bytes = damaged((b) => {
    new DataView(b.buffer).setUint16(4, 2, true)
  })
  expect(() => loadIndex(bytes, meta(), sha256)).toThrow(/format_version/)
})

test('float16 отвергается явной ошибкой', () => {
  const bytes = damaged((b) => {
    b[16] = 0
  })
  expect(() => loadIndex(bytes, meta(), sha256)).toThrow(/float16/)
})

test('ненулевой reserved отвергается', () => {
  const bytes = damaged((b) => {
    b[18] = 1
  })
  expect(() => loadIndex(bytes, meta(), sha256)).toThrow(/reserved/)
})

test('несходящаяся длина отвергается', () => {
  const bytes = bin().slice(0, bin().length - 4)
  expect(() => loadIndex(bytes, meta(), sha256)).toThrow(/длина/)
})

test('чужая мета отвергается по sha256', () => {
  const bytes = new Uint8Array([...meta(), 0x20])
  expect(() => loadIndex(bin(), bytes, sha256)).toThrow(/sha256/)
})

test('число классов в мете сверяется с заголовком', () => {
  const document = JSON.parse(new TextDecoder().decode(meta())) as { classes: unknown[] }
  document.classes = document.classes.slice(0, 1)
  const patched = new TextEncoder().encode(JSON.stringify(document))
  const header = bin().slice(0, 64)
  new Uint8Array(header.buffer, 20, 32).set(Uint8Array.from(Buffer.from(sha256(patched), 'hex')))
  const bytes = new Uint8Array([...header, ...bin().slice(64)])
  expect(() => loadIndex(bytes, patched, sha256)).toThrow(/classes/)
})
