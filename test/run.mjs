/**
 * test/run.mjs — the plugin's Node test suite (`npm test`).
 *
 * Two levels, both without a DSH runtime:
 *
 *   A. the pure layer — argv construction, summary parsing, result rendering, and
 *      the descriptor this plugin hands to `ctx.tools.register`. The descriptor is
 *      checked against the REAL tool registry's JSON-Schema validator
 *      (`@deepseek-ai/dsh-tools`) when that package is resolvable, so a schema the
 *      harness would reject cannot pass here.
 *   B. the real thing — `apply()` against a fake but faithful ctx, whose
 *      subprocess service is backed by node:child_process. That means the Python
 *      interpreter is really located, really spawned, and real images are really
 *      written; the assertions are on files on disk, not on the code path.
 *
 * The Python suites (test/test_remove_watermark.py, test/test_signature_and_lattice.py)
 * measure detection quality; this file proves the plugin half is wired correctly.
 *
 * Run: node test/run.mjs
 */
import { strict as assert } from 'node:assert'
import { spawn } from 'node:child_process'
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, relative } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const ROOT = fileURLToPath(new URL('..', import.meta.url))
const SCRIPT = join(ROOT, 'src', 'remove_watermark.py')

const results = []
function check(name, ok, detail) {
  results.push({ name, ok, detail })
  console.log(`  [${ok ? 'PASS' : 'FAIL'}] ${name.padEnd(30)} ${detail ?? ''}`)
}

// ---------------------------------------------------------------- fake runtime
/**
 * A subprocess service with the semantics this plugin relies on: argv is never
 * shell-interpreted, collected stdout/stderr are readable after exit via
 * readFrom(0), and `done` resolves with the exit facts.
 */
function makeSubprocess() {
  return {
    async resolveExecutable(command) {
      const r = await runProcess(process.platform === 'win32' ? 'where' : 'which', [command], ROOT)
      if (r.code !== 0) return undefined
      const first = r.stdout.split(/\r?\n/).map((s) => s.trim()).filter(Boolean)[0]
      return first || undefined
    },
    spawn(spec) {
      const child = spawn(spec.argv[0], spec.argv.slice(1), {
        cwd: spec.cwd,
        env: { ...process.env, ...(spec.env ?? {}) },
        stdio: ['ignore', 'pipe', 'pipe'],
        windowsHide: true,
      })
      let out = ''
      let err = ''
      child.stdout.on('data', (d) => { out += String(d) })
      child.stderr.on('data', (d) => { err += String(d) })
      const done = new Promise((resolve, reject) => {
        child.on('error', reject)
        child.on('close', (code) => resolve({ exitCode: code, signal: null }))
      })
      const reader = (text) => ({ readFrom: () => ({ text, nextOffset: text.length, lossy: false }) })
      return {
        collected: { stdout: reader(out), stderr: reader(err) },
        done,
        terminate() { child.kill() },
        async waitForExit() { await done; return true },
        // test-only accessors, refreshed on read
        get liveStdout() { return out },
        get liveStderr() { return err },
      }
    },
  }
}

function runProcess(cmd, args, cwd) {
  return new Promise((resolve) => {
    const child = spawn(cmd, args, { cwd, windowsHide: true })
    let stdout = ''
    child.stdout?.on('data', (d) => { stdout += String(d) })
    child.on('error', () => resolve({ code: -1, stdout: '' }))
    child.on('close', (code) => resolve({ code, stdout }))
  })
}

/** The real subprocess service returns readers that reflect all output at exit. */
function makeSubprocessFixed() {
  const base = makeSubprocess()
  return {
    resolveExecutable: base.resolveExecutable.bind(base),
    spawn(spec) {
      const handle = base.spawn(spec)
      const readers = {}
      const wrap = (key, get) => ({
        readFrom: () => ({ text: get(), nextOffset: get().length, lossy: false }),
      })
      readers.stdout = wrap('stdout', () => handle.liveStdout)
      readers.stderr = wrap('stderr', () => handle.liveStderr)
      return {
        collected: readers,
        done: handle.done,
        terminate: handle.terminate,
        waitForExit: handle.waitForExit,
      }
    },
  }
}

/** A filesystem service with the semantics the plugin uses (resolve + stat/read). */
function makeFs() {
  return {
    async resolve(p, _opts) { return { path: p } },
    processPath(target) { return target.path },
    async stat(target) {
      const info = existsSync(target.path) ? statSyncSafe(target.path) : undefined
      return info
    },
    async readText(target) { return readFileSync(target.path, 'utf8') },
  }
}

function statSyncSafe(p) {
  try {
    const { statSync } = require('node:fs')
    const s = statSync(p)
    return { type: s.isFile() ? 'file' : s.isDirectory() ? 'directory' : 'other', size: s.size }
  } catch {
    return undefined
  }
}

// A synthetic image generator: enough to exercise the real pipeline (the Python
// suites own detection quality, this owns the wiring).
function writePng(path, w, h) {
  // Minimal PNG encoder (truecolour, no compression beyond zlib stored blocks) so
  // this suite has no image dependency.
  const { deflateSync } = require('node:zlib')
  const raw = Buffer.alloc((w * 3 + 1) * h)
  let o = 0
  for (let y = 0; y < h; y++) {
    raw[o++] = 0
    for (let x = 0; x < w; x++) {
      const base = (x + y) % 2 === 0 ? 30 : 34
      // a bright block that looks nothing like a watermark, plus a bright square
      const inBlock = x > w - 60 && y > h - 60
      raw[o++] = inBlock ? 240 : base
      raw[o++] = inBlock ? 240 : base
      raw[o++] = inBlock ? 240 : base
    }
  }
  const chunks = []
  const chunk = (type, data) => {
    const len = Buffer.alloc(4)
    len.writeUInt32BE(data.length)
    const body = Buffer.concat([Buffer.from(type, 'ascii'), data])
    const crc = Buffer.alloc(4)
    crc.writeUInt32BE(crc32(body) >>> 0)
    return Buffer.concat([len, body, crc])
  }
  const ihdr = Buffer.alloc(13)
  ihdr.writeUInt32BE(w, 0)
  ihdr.writeUInt32BE(h, 4)
  ihdr[8] = 8
  ihdr[9] = 2
  chunks.push(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]))
  chunks.push(chunk('IHDR', ihdr))
  chunks.push(chunk('IDAT', deflateSync(raw)))
  chunks.push(chunk('IEND', Buffer.alloc(0)))
  writeFileSync(path, Buffer.concat(chunks))
}

let CRC_TABLE
function crc32(buf) {
  if (!CRC_TABLE) {
    CRC_TABLE = new Int32Array(256)
    for (let n = 0; n < 256; n++) {
      let c = n
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
      CRC_TABLE[n] = c
    }
  }
  let c = -1
  for (const b of buf) c = CRC_TABLE[(c ^ b) & 0xff] ^ (c >>> 8)
  return c ^ -1
}

const require = (await import('node:module')).createRequire(import.meta.url)

// ------------------------------------------------------------------------ tests
const mod = await import(pathToFileURL(join(ROOT, 'lib', 'index.js')).href)

console.log('>>> host layer test')
check('plugin_shape', mod.name === 'dsh-watermark'
  && Array.isArray(mod.inject) && mod.inject.includes('tools')
  && typeof mod.apply === 'function',
  `name=${mod.name} inject=[${mod.inject}]`)

// A. pure layer
const argv = mod.buildArgv({
  paths: ['a.png', 'b/'],
  search: 'bottom-right',
  rect: ['10,20,30,40'],
  restore: true,
  template: 'sig.png',
  kSigma: 2,
  dryRun: false,
  strategy: 'template',
  nonsense: 'ignored',
}, SCRIPT, 'report.json')
check('buildArgv_order', argv[0] === SCRIPT && argv[1] === 'a.png' && argv[2] === 'b/'
  && argv.includes('--rect') && argv.includes('--restore')
  && argv[argv.length - 2] === '--json-out' && argv[argv.length - 1] === 'report.json',
  argv.slice(1).join(' '))
check('buildArgv_skips_unknown', !argv.includes('--nonsense') && !argv.includes('ignored')
  && !argv.includes('--dry-run'),
  'unknown keys and false booleans are not passed')

const summary = mod.parseSummary('noise\n>>> remove_watermark: 2 file(s)\n[SUMMARY] ok 2 / fail 0 / total 2\n')
check('parseSummary', summary?.ok === 2 && summary.total === 2, JSON.stringify(summary))
check('parseSummary_absent', mod.parseSummary('nothing here') === undefined, 'returns undefined')

const rendered = mod.renderResult({
  ok: true, total: 2, succeeded: 1, failed: 1, strategy: 'single', dryRun: false,
  outputDir: '/out', files: [
    { name: 'a.png', ok: true, changedPct: 1.25, confidence: 4.2, note: 'inpainted 100 px' },
    { name: 'b.png', ok: false, note: 'cannot read' },
  ],
  failures: [{ name: 'b.png', reason: 'cannot read' }],
})
check('renderResult', rendered.includes('成功 1/2') && rendered.includes('a.png')
  && rendered.includes('conf 4.2') && rendered.includes('b.png'),
  rendered.split('\n')[0])

// The descriptor must satisfy the REAL registry's JSON-Schema validation. The
// harness package is not a dependency of this repo, so it is imported when it is
// resolvable and skipped (loudly, never silently passed) when it is not. Point
// DSH_TOOLS_PATH at the deployment's `@deepseek-ai/dsh-tools/lib/index.js` to run
// this check anywhere:
//   DSH_TOOLS_PATH=<dsh install>/node_modules/@deepseek-ai/dsh-tools/lib/index.js node test/run.mjs
let schemaCheck = 'skipped: @deepseek-ai/dsh-tools not resolvable'
let schemaOk = true
{
  let dshTools
  const tried = []
  for (const spec of ['@deepseek-ai/dsh-tools', process.env.DSH_TOOLS_PATH]) {
    if (!spec) continue
    tried.push(spec)
    try {
      dshTools = spec.startsWith('@')
        ? await import(spec)
        : await import(pathToFileURL(spec).href)
      break
    } catch (error) {
      schemaCheck = `skipped: ${String(error).split('\n')[0].slice(0, 110)}`
    }
  }
  if (dshTools) {
    // Exercise the two contracts the real registry enforces on a definition:
    //   * register() calls assertSupportedJsonSchema(output.schema) itself;
    //   * parameters is consumed as a raw JSON Schema, rendered into the model's
    //     TS/Python tool signature. Both are the harness's own code, not a copy.
    try {
      const captured = {}
      mod.apply(makeCtx(captured), {})
      const tool = captured.tool
      dshTools.assertSupportedJsonSchema(tool.output.schema)
      const ts = dshTools.jsonSchemaToTs?.(tool.parameters)
      const py = dshTools.jsonSchemaToPy?.(tool.parameters)
      const rendered = `${ts ?? ''}${py ?? ''}`
      if (typeof ts === 'string' && !rendered.includes('paths')) {
        throw new Error('the parameters schema rendered without its required `paths`')
      }
      schemaCheck = `output.schema accepted; parameters rendered by the harness `
        + `(${Object.keys(tool.parameters.properties).length} params)`
    } catch (error) {
      schemaOk = false
      schemaCheck = `rejected: ${String(error).split('\n')[0].slice(0, 160)}`
    }
  }
}
check('descriptor_json_schema', schemaOk, schemaCheck)

// The structural check always runs, so a missing harness package cannot turn the
// descriptor contract into an untested claim.
{
  const captured = {}
  mod.apply(makeCtx(captured), {})
  const tool = captured.tool
  const props = tool.parameters.properties ?? {}
  const problems = []
  if (tool.parameters.type !== 'object') problems.push('parameters must be an object schema')
  if (!Array.isArray(tool.parameters.required) || tool.parameters.required.length === 0) {
    problems.push('no required parameter')
  }
  for (const key of tool.parameters.required ?? []) {
    if (!(key in props)) problems.push(`required "${key}" is not declared`)
  }
  for (const [key, spec] of Object.entries(props)) {
    if (typeof spec.type !== 'string') problems.push(`"${key}" has no type`)
  }
  if (tool.parameters.additionalProperties !== false) {
    problems.push('additionalProperties must be false so a stray key cannot be passed')
  }
  if (typeof tool.description !== 'string' || tool.description.length < 40) {
    problems.push('description too thin for the model to choose this tool')
  }
  if (typeof tool.output?.render !== 'function') problems.push('output.render missing')
  check('descriptor_shape', problems.length === 0,
    problems.length ? problems.join('; ') : `${Object.keys(props).length} typed params, `
      + `required=[${tool.parameters.required.join(',')}]`)
}

function makeCtx(captured) {
  const disposers = []
  return {
    tools: {
      register(tool) {
        captured.tool = tool
        return () => {}
      },
    },
    fs: makeFs(),
    subprocess: makeSubprocessFixed(),
    effect(fn) { disposers.push(fn) },
    logger: { warn: (m) => console.log('       (plugin warn)', m.slice(0, 120)) },
  }
}

// B. end to end through apply()
const tmp = mkdtempSync(join(tmpdir(), 'dsh-wm-node-'))
try {
  const inDir = join(tmp, 'in')
  const { mkdirSync } = await import('node:fs')
  mkdirSync(inDir, { recursive: true })
  const img = join(inDir, 'shot.png')
  writePng(img, 160, 120)

  const captured = {}
  mod.apply(makeCtx(captured), {})
  const tool = captured.tool
  check('tool_registered', tool?.name === 'remove_watermark' && typeof tool.execute === 'function',
    `name=${tool?.name} timeoutMs=${tool?.timeoutMs}`)

  const py = await mod.detectPython(makeSubprocessFixed())
  check('python_detected', Boolean(py),
    py ? `${py.exe} cv2 ${py.version}` : 'no interpreter with cv2+numpy found')

  if (py) {
    const res = await tool.execute({ paths: [img], search: 'all' }, {})
    const out = join(inDir, 'clean', 'shot-clean.png')
    check('e2e_run_ok', res.ok === true && res.total === 1 && res.exitCode === 0,
      `ok=${res.ok} total=${res.total} succeeded=${res.succeeded} exit=${res.exitCode}`)
    // Detail deliberately relative to the run's temp root: an absolute path would
    // carry the random mkdtemp suffix and make this output non-reproducible.
    check('e2e_output_exists', existsSync(out), `${relative(tmp, out)} (under the run's temp dir)`
      + `, ${statSyncSafe(out)?.size ?? 0} bytes`)
    check('e2e_result_shape', Array.isArray(res.files) && res.files[0].name === 'shot.png'
      && typeof res.files[0].changedPct === 'number' && res.python === py.version,
      JSON.stringify({ name: res.files?.[0]?.name, changedPct: res.files?.[0]?.changedPct,
        maskPx: res.files?.[0]?.maskPx, confidence: res.files?.[0]?.confidence }))

    // a claimed success that is not on disk must be reported as a failure
    const lying = await mod.runRemoval({ paths: [img] }, {
      subprocess: makeSubprocessFixed(),
      python: py,
      cwd: ROOT,
      scriptPath: SCRIPT,
      reportPath: join(tmp, 'fake-report.json'),
      readFile: async () => JSON.stringify({
        strategy: 'single', summary: { ok: 1, fail: 0, total: 1 },
        files: [{ input: 'x.png', output: join(tmp, 'missing.png'), ok: true, note: 'inpainted' }],
      }),
      statPath: async (p) => (existsSync(p) ? { type: 'file' } : undefined),
    })
    check('e2e_missing_output_caught', lying.succeeded === 0 && lying.failed === 1
      && lying.files[0].note.includes('not on disk'),
      `succeeded=${lying.succeeded} failed=${lying.failed}`)

    const dry = await tool.execute({ paths: [img], dryRun: true }, {})
    check('e2e_dry_run', dry.ok === true && dry.dryRun === true && dry.outputDir === undefined,
      `dryRun=${dry.dryRun} files=${dry.files?.length}`)

    const bad = await tool.execute({ paths: [join(tmp, 'nope.png')] }, {})
    check('e2e_missing_input', bad.ok === false && String(bad.error).length > 0,
      String(bad.error).split('\n')[0].slice(0, 100))
  } else {
    check('e2e_run_ok', false, 'skipped: no interpreter')
  }
} finally {
  rmSync(tmp, { recursive: true, force: true })
}

const passed = results.filter((r) => r.ok).length
const failed = results.length - passed
console.log('')
console.log(`[SUMMARY] passed ${passed} / failed ${failed}`)
process.exit(failed === 0 ? 0 : 1)
