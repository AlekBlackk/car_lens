import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import test from 'node:test'

import { IndexFormatError, loadIndex } from './catalogIndex.js'

const sha256 = (data: Uint8Array): string => createHash('sha256').update(data).digest('hex')

const fixture = (name: string): Uint8Array =>
  new Uint8Array(readFileSync(new URL(`../../../tests/fixtures/index/${name}`, import.meta.url)))

const bin = (): Uint8Array => fixture('catalog_index.bin')
const meta = (): Uint8Array => fixture('catalog_meta.json')

test('разбирает заголовок фикстуры', () => {
  const index = loadIndex(bin(), meta(), sha256)
  assert.equal(index.header.formatVersion, 1)
  assert.equal(index.header.numClasses, 2)
  assert.equal(index.header.centroidsPerClass, 4)
  assert.equal(index.header.embeddingDim, 256)
  assert.equal(index.header.dtype, 1)
  assert.equal(index.header.normalized, true)
})

test('классы приходят в порядке индексов', () => {
  const index = loadIndex(bin(), meta(), sha256)
  assert.equal(index.classes.length, 2)
  assert.equal(index.classes[0]?.model_id, 'acme__one')
  assert.equal(index.classes[1]?.display, 'Acme Two')
})

test('центроиды нормализованы и не копируются лишний раз', () => {
  const index = loadIndex(bin(), meta(), sha256)
  for (let c = 0; c < index.header.numClasses; c += 1) {
    for (let k = 0; k < index.header.centroidsPerClass; k += 1) {
      const vector = index.centroid(c, k)
      assert.equal(vector.length, 256)
      const norm = Math.hypot(...vector)
      assert.ok(Math.abs(norm - 1) < 1e-5, `норма ${norm}`)
    }
  }
})

test('match находит класс собственного центроида', () => {
  const index = loadIndex(bin(), meta(), sha256)
  const target = index.centroid(1, 2)
  const found = index.match(Float32Array.from(target))
  assert.equal(found.classIdx, 1)
  assert.ok(found.sim > 0.999, `sim ${found.sim}`)
})

test('centroid отвергает индексы вне диапазона', () => {
  const index = loadIndex(bin(), meta(), sha256)
  assert.throws(() => index.centroid(2, 0), IndexFormatError)
  assert.throws(() => index.centroid(0, 4), IndexFormatError)
})

test('match отвергает вектор чужой размерности', () => {
  const index = loadIndex(bin(), meta(), sha256)
  assert.throws(() => index.match(new Float32Array(8)), IndexFormatError)
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
  assert.throws(() => loadIndex(bytes, meta(), sha256), /magic/)
})

test('чужая версия формата отвергается', () => {
  const bytes = damaged((b) => {
    new DataView(b.buffer).setUint16(4, 2, true)
  })
  assert.throws(() => loadIndex(bytes, meta(), sha256), /format_version/)
})

test('float16 отвергается явной ошибкой', () => {
  const bytes = damaged((b) => {
    b[16] = 0
  })
  assert.throws(() => loadIndex(bytes, meta(), sha256), /float16/)
})

test('ненулевой reserved отвергается', () => {
  const bytes = damaged((b) => {
    b[18] = 1
  })
  assert.throws(() => loadIndex(bytes, meta(), sha256), /reserved/)
})

test('несходящаяся длина отвергается', () => {
  const bytes = bin().slice(0, bin().length - 4)
  assert.throws(() => loadIndex(bytes, meta(), sha256), /длина/)
})

test('чужая мета отвергается по sha256', () => {
  const bytes = new Uint8Array([...meta(), 0x20])
  assert.throws(() => loadIndex(bin(), bytes, sha256), /sha256/)
})

test('число классов в мете сверяется с заголовком', () => {
  const document = JSON.parse(new TextDecoder().decode(meta())) as { classes: unknown[] }
  document.classes = document.classes.slice(0, 1)
  const patched = new TextEncoder().encode(JSON.stringify(document))
  const header = bin().slice(0, 64)
  new Uint8Array(header.buffer, 20, 32).set(Uint8Array.from(Buffer.from(sha256(patched), 'hex')))
  const bytes = new Uint8Array([...header, ...bin().slice(64)])
  assert.throws(() => loadIndex(bytes, patched, sha256), /classes/)
})
