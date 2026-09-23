/**
 * dsh-watermark — host half.
 *
 * Registers one tool, `remove_watermark`, which drives the algorithm in
 * `src/remove_watermark.py`. What this file is responsible for, and nothing else:
 *
 *   - validating the tool's arguments and translating them to argv
 *     (`buildArgv`, one pure function, unit-tested without a runtime);
 *   - locating a Python interpreter that actually has cv2 + numpy, and saying so
 *     plainly when there is none (the algorithm is NOT re-implemented in Node: it
 *     was measured in Python against a synthetic suite, and a rewrite would have
 *     to earn all of that evidence again);
 *   - spawning it, parsing the JSON report it writes, and turning that report into
 *     a tool result — including refusing to call a run successful when the tool
 *     wrote nothing (`--dry-run` aside, every ok file must exist on disk).
 *
 * Hard dependencies (inject): tools / fs / subprocess. If one is missing the
 * plugin should not activate at all, so Cordis waits for them rather than apply()
 * checking for undefined everywhere.
 *
 * Loaded as a plain ES module with no bare imports, so a `link:` install works:
 * a plugin that resolves `@deepseek-ai/dsh-tools` from its own real path cannot
 * (the package lives in the harness deployment, not next to the plugin).
 */

import { fileURLToPath } from 'node:url'

/** Same id as package.json's dsh.bundle.patch / cordis.patch.yml. */
export const name = 'dsh-watermark'

export const inject = ['tools', 'fs', 'subprocess']

/** The algorithm, addressed relative to this module rather than to cwd. */
export const SCRIPT_PATH = fileURLToPath(new URL('../src/remove_watermark.py', import.meta.url))

/** Candidate interpreters, in the order they are tried. */
export const PYTHON_CANDIDATES = ['python', 'python3', 'py']

/** Exit codes the script documents. */
export const EXIT_OK = 0
export const EXIT_FILE_FAILED = 1
export const EXIT_BAD_ARGS = 2

/** Argument names the tool accepts one-to-one, in a stable order. */
const FLAG_MAP = [
  ['search', '--search', 'string'],
  ['template', '--template', 'string'],
  ['strategy', '--strategy', 'string'],
  ['period', '--period', 'string'],
  ['color', '--color', 'string'],
  ['outdir', '--outdir', 'string'],
  ['maskOutDir', '--mask-out-dir', 'string'],
  ['suffix', '--suffix', 'string'],
  ['outExt', '--out-ext', 'string'],
  ['mode', '--mode', 'string'],
  ['contrastDelta', '--contrast-delta', 'number'],
  ['kSigma', '--k-sigma', 'number'],
  ['saliencyRatio', '--saliency-ratio', 'number'],
  ['minSaliency', '--min-saliency', 'number'],
  ['collapseRatio', '--collapse-ratio', 'number'],
  ['collapseZ', '--collapse-z', 'number'],
  ['minBlob', '--min-blob', 'number'],
  ['minSeedBlob', '--min-seed-blob', 'number'],
  ['coreDelta', '--core-delta', 'number'],
  ['signConsistency', '--sign-consistency', 'number'],
  ['matchWindow', '--match-window', 'number'],
  ['window', '--window', 'number'],
  ['dilate', '--dilate', 'number'],
  ['radius', '--radius', 'number'],
  ['alpha', '--alpha', 'number'],
  ['opaqueAlpha', '--opaque-alpha', 'number'],
  ['maxAreaFrac', '--max-area-frac', 'number'],
  ['maxFillRatio', '--max-fill-ratio', 'number'],
  ['multiRatio', '--multi-ratio', 'number'],
  ['multiMinLocal', '--multi-min-local', 'number'],
]

/** Boolean flags, passed as write-only switches. */
const BOOL_MAP = [
  ['restore', '--restore'],
  ['overwrite', '--overwrite'],
  ['dryRun', '--dry-run'],
  ['autoScales', '--auto-scales'],
  ['noCollapseDetect', '--no-collapse-detect'],
]

/**
 * Translate validated tool arguments into the script's argv (without the
 * interpreter). Pure: no IO, no globals — this is the part worth unit-testing.
 *
 * @param args - tool arguments (already validated by the tool's JSON Schema)
 * @param scriptPath - absolute path to remove_watermark.py
 * @param reportPath - where the script should write its JSON report
 * @returns argv for the interpreter
 */
export function buildArgv(args, scriptPath, reportPath) {
  const argv = [scriptPath]
  const paths = Array.isArray(args.paths) ? args.paths : [args.paths]
  for (const p of paths) {
    if (typeof p === 'string' && p.length > 0) argv.push(p)
  }
  if (Array.isArray(args.rect)) {
    for (const r of args.rect) if (typeof r === 'string' && r.length > 0) argv.push('--rect', r)
  }
  if (Array.isArray(args.learn)) {
    for (const pair of args.learn) {
      if (Array.isArray(pair) && pair.length === 2) argv.push('--learn', pair[0], pair[1])
    }
  }
  for (const [key, flag, kind] of FLAG_MAP) {
    const value = args[key]
    if (value === undefined || value === null || value === '') continue
    if (kind === 'number' && !Number.isFinite(value)) continue
    argv.push(flag, String(value))
  }
  for (const [key, flag] of BOOL_MAP) {
    if (args[key] === true) argv.push(flag)
  }
  argv.push('--json-out', reportPath)
  return argv
}

/**
 * Parse the script's stdout for its one machine-readable summary line.
 * Returns undefined rather than throwing: the JSON report is the contract, this
 * is only used for diagnostics when the report is missing.
 * @param stdout - the child's collected stdout
 */
export function parseSummary(stdout) {
  if (typeof stdout !== 'string') return undefined
  for (const line of stdout.split(/\r?\n/)) {
    const m = /^\[SUMMARY\] ok (\d+) \/ fail (\d+) \/ total (\d+)$/.exec(line.trim())
    if (m) return { ok: Number(m[1]), fail: Number(m[2]), total: Number(m[3]) }
  }
  return undefined
}

/**
 * Render the tool result for the model/user: a compact table plus the numbers a
 * caller needs to judge whether this run is trustworthy. `confidence` is the
 * detector's own mean residual of the accepted pixels over the noise floor (>= 1,
 * higher is better); it is NOT a hit rate — a true hit rate needs ground truth,
 * which a single run does not have. The suite in test/ measures that instead.
 * @param value - the tool's structured result
 */
export function renderResult(value) {
  if (value.ok === false) return `remove_watermark failed: ${value.error ?? 'unknown error'}`
  const lines = []
  lines.push(`去水印：成功 ${value.succeeded}/${value.total} 张｜策略 ${value.strategy}` +
    `${value.restore ? '（精确还原）' : ''}${value.dryRun ? '（dry run）' : ''}`)
  if (value.outputDir) lines.push(`输出目录：${value.outputDir}`)
  for (const f of value.files ?? []) {
    const parts = [`${f.ok ? '✓' : '✗'} ${f.name}`]
    if (typeof f.changedPct === 'number') parts.push(`${f.changedPct}%`)
    if (typeof f.confidence === 'number' && f.confidence > 0) parts.push(`conf ${f.confidence}`)
    if (f.note) parts.push(f.note)
    lines.push('  ' + parts.join(' · '))
  }
  for (const f of value.failures ?? []) lines.push(`  ✗ ${f.name}: ${f.reason}`)
  if (value.warnings?.length) lines.push('警告：' + value.warnings.join('；'))
  return lines.join('\n')
}

/** One interpreter probe: does this command exist AND import cv2 + numpy? */
async function probePython(subprocess, command) {
  try {
    const exe = await subprocess.resolveExecutable(command)
    if (!exe) return undefined
    const handle = subprocess.spawn({
      argv: [exe, '-c', 'import cv2, numpy; print(cv2.__version__)'],
      cwd: process.cwd(),
      stdio: { stdin: 'ignore', stdout: { maxBytes: 4096 }, stderr: { maxBytes: 4096 } },
      graceMs: 5000,
    })
    const outcome = await handle.done
    if (outcome.exitCode !== 0) return undefined
    const read = await handle.collected.stdout.readFrom(0)
    return { exe, version: read.text.trim() }
  } catch {
    return undefined
  }
}

/**
 * Find a usable interpreter once per plugin lifetime.
 * @param subprocess - the subprocess service
 * @param candidates - command names to try in order
 * @returns {exe, version} or undefined
 */
export async function detectPython(subprocess, candidates = PYTHON_CANDIDATES) {
  for (const candidate of candidates) {
    const found = await probePython(subprocess, candidate)
    if (found) return found
  }
  return undefined
}

/** Collect a collected-output reader into text (bounded by its own cap). */
async function drain(reader) {
  if (!reader) return ''
  const read = reader.readFrom(0)
  return read?.text ?? ''
}

/**
 * Run one removal, from arguments to a structured result. Exported so the test
 * suite can drive the real thing with a real interpreter and no DSH runtime.
 *
 * @param args - validated tool arguments
 * @param deps - { subprocess, python, cwd, signal }
 * @returns the tool's structured result
 */
export async function runRemoval(args, deps) {
  const { subprocess, python, cwd } = deps
  const reportPath = deps.reportPath
  const argv = buildArgv(args, deps.scriptPath, reportPath)
  const handle = subprocess.spawn({
    argv: [python.exe, '-X', 'utf8', ...argv],
    cwd,
    stdio: {
      stdin: 'ignore',
      stdout: { maxBytes: 4_000_000 },
      stderr: { maxBytes: 1_000_000 },
    },
    graceMs: 10_000,
    env: { PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8' },
    signal: deps.signal,
  })
  let outcome
  try {
    outcome = await handle.done
  } catch (error) {
    return { ok: false, error: `could not start ${python.exe}: ${String(error)}` }
  }
  const [stdout, stderr] = await Promise.all([
    drain(handle.collected.stdout),
    drain(handle.collected.stderr),
  ])
  const summary = parseSummary(stdout)
  let report
  try {
    report = JSON.parse(await deps.readFile(reportPath))
  } catch {
    report = undefined
  }
  if (!report) {
    const tail = (stderr || stdout || '(no output)').trim().split(/\r?\n/).slice(-12).join('\n')
    return {
      ok: false,
      error: `the watermark tool exited ${outcome.exitCode} without writing a report\n${tail}`,
      exitCode: outcome.exitCode,
      stdout: stdout.slice(-2000),
    }
  }
  const files = (report.files ?? []).map((f) => ({
    name: f.input.split(/[\\/]/).pop(),
    ok: f.ok,
    output: f.output ?? undefined,
    note: f.note,
    changedPct: f.changedPct,
    maskPx: f.maskPx,
    confidence: f.confidence,
    strategy: f.strategy,
    solvedPx: f.solvedPx,
  }))
  const failures = files.filter((f) => !f.ok).map((f) => ({ name: f.name, reason: f.note }))
  const warnings = []
  if (summary && report.summary && summary.ok !== report.summary.ok) {
    warnings.push('stdout summary disagrees with the JSON report; the report was used')
  }
  // Refuse to call a run successful when the files it claims are not on disk. This
  // is the failure mode that makes a plugin like this untrustworthy: the tool
  // reports success and the output directory is empty.
  if (!report.dryRun && deps.statPath) {
    for (const f of files) {
      if (!f.ok || !f.output) continue
      let info
      try {
        info = await deps.statPath(f.output)
      } catch {
        info = undefined
      }
      if (!info || info.type !== 'file') {
        f.ok = false
        f.note = `reported written but not on disk: ${f.output}`
        if (!failures.some((x) => x.name === f.name)) {
          failures.push({ name: f.name, reason: f.note })
        }
      }
    }
  }
  const succeeded = files.filter((f) => f.ok).length
  const outputDir = report.dryRun ? undefined : dirOf(files.find((f) => f.ok && f.output)?.output)
  return {
    ok: true,
    total: report.summary?.total ?? files.length,
    succeeded,
    failed: failures.length,
    strategy: report.strategy,
    restore: Boolean(report.restore),
    dryRun: Boolean(report.dryRun),
    period: report.period ?? undefined,
    outputDir,
    python: python.version,
    files,
    failures,
    warnings,
    exitCode: outcome.exitCode,
  }
}

/** Directory part of a path in either separator convention. */
function dirOf(p) {
  if (typeof p !== 'string') return undefined
  const i = Math.max(p.lastIndexOf('/'), p.lastIndexOf('\\'))
  return i < 0 ? undefined : p.slice(0, i)
}

/**
 * Install the tool.
 * @param ctx - plugin context (tools + fs + subprocess injected)
 * @param config - { python?, timeoutMs? }
 */
export function apply(ctx, config = {}) {
  const tools = ctx.tools
  const fs = ctx.fs
  const subprocess = ctx.subprocess
  const scriptPath = config.script ?? SCRIPT_PATH
  let pythonPromise

  const getPython = () => {
    if (!pythonPromise) {
      pythonPromise = config.python
        ? probePython(subprocess, config.python)
        : detectPython(subprocess)
      pythonPromise.then((found) => {
        if (!found) {
          ctx.logger?.warn?.(
            'dsh-watermark: no Python with cv2 + numpy found; install opencv-python and '
            + 'numpy, or set the plugin option `python` to the interpreter path')
        }
      })
    }
    return pythonPromise
  }

  const disposeTool = tools.register({
    name: 'remove_watermark',
    description: [
      'Remove watermarks from image files or a whole directory tree (originals are never '
      + 'modified; outputs go to <dir>/clean/<name>-clean.png). Handles corner marks, any '
      + 'position, light/dark/coloured marks, semi-transparent marks and tiled marks. ',
      'Just point it at the folder: with the default `strategy: "auto"` it uses cross-image '
      + 'evidence whenever >=3 of the inputs share one size (by far the most reliable mode, '
      + 'and the one for a batch from a single platform), and falls back to single-image '
      + 'detection otherwise. Searching defaults to the four corners, where platform marks '
      + 'live. ',
      '`template` + `restore` inverts a known semi-transparent mark exactly instead of '
      + 'painting over it (obtain such a signature by running the script once with --learn on '
      + 'a marked/clean pair of the same picture). ',
      'Read the reported changed% and confidence, and use dryRun + maskOutDir before trusting '
      + 'a batch: the tool marks a run as suspicious when the mask looks like the photo rather '
      + 'than a watermark, and single-image detection on a busy photo does get it wrong.',
    ].join(''),
    parameters: {
      type: 'object',
      additionalProperties: false,
      required: ['paths'],
      properties: {
        paths: {
          type: 'array',
          items: { type: 'string' },
          description: 'Files or directories to process. Directories are walked. Needs at '
            + 'least 3 images of the same size that share one watermark.',
        },
        search: {
          type: 'string',
          enum: ['auto', 'all', 'bottom-right', 'bottom-left', 'top-right', 'top-left', 'bottom',
            'top', 'corners', 'none'],
          description: 'Where the detector may look. Default "auto" = the four corners, where '
            + 'platform marks live (searching the whole frame measurably damages real photos), '
            + 'and the whole frame when a template says where the mark is. Pass "all" for a '
            + 'centred mark.',
        },
        rect: {
          type: 'array',
          items: { type: 'string' },
          description: 'Exact boxes "x,y,w,h" (repeatable) when you know where the mark is.',
        },
        template: {
          type: 'string',
          description: 'The mark itself: a signature PNG from --learn (RGBA, placed via its '
            + '.json sidecar) or a shape image where WHITE is the mark. With this the batch '
            + 'requirement does not apply.',
        },
        restore: {
          type: 'boolean',
          description: 'With template: invert the overlay exactly: '
            + 'original = (observed - a*color)/(1 - a). Opaque pixels are inpainted instead.',
        },
        strategy: {
          type: 'string',
          enum: ['auto', 'multi', 'template', 'single'],
          description: 'auto and multi are the same thing: shared evidence from >=3 same-size '
            + 'frames. "single" is removed and exits with an explanation.',
        },
        color: { type: 'string', description: 'Mark colour B,G,R when the template has no colour.' },
        alpha: {
          type: 'number',
          description: 'Peak alpha for a shape-only template; default: calibrated from the image.',
        },
        outdir: { type: 'string', description: 'Output directory (default <input dir>/clean).' },
        suffix: { type: 'string', description: 'Output name suffix (default -clean).' },
        outExt: {
          type: 'string',
          description: 'Output extension, e.g. "png". Use png for JPEG inputs: writing JPEG '
            + 're-compresses every pixel, so the output stops being a record of what changed.',
        },
        maskOutDir: { type: 'string', description: 'Write each mask here for inspection.' },
        dryRun: { type: 'boolean', description: 'Report what would change; write nothing.' },
        overwrite: { type: 'boolean', description: 'Overwrite existing outputs.' },
        multiRatio: {
          type: 'number',
          description: 'How much less a candidate must vary across frames than its own '
            + 'neighbourhood (default 0.65; measured marks 0.48-0.57, background 0.73-0.89).',
        },
        minBlob: { type: 'number', description: 'Drop components smaller than this (default 40).' },
        window: { type: 'number', description: 'Local window radius (default 31).' },
        dilate: { type: 'number', description: 'Grow the mask by this many px (default 3).' },
        radius: { type: 'number', description: 'Inpaint radius (default 5).' },
        mode: { type: 'string', enum: ['telea', 'ns'], description: 'Inpainting method.' },
        maxComponents: {
          type: 'number',
          description: 'More accepted components than this and the run is reported as '
            + 'suspicious: a watermark is one small group of pixels (default 10).',
        },
        maxMaskFrac: {
          type: 'number',
          description: 'Same, for the share of the search area the mask covers (default 0.08).',
        },
      },
    },
    timeoutMs: config.timeoutMs ?? 900_000,
    output: {
      schema: {
        type: 'object',
        additionalProperties: true,
        properties: {
          ok: { type: 'boolean' },
          total: { type: 'number' },
          succeeded: { type: 'number' },
          failed: { type: 'number' },
          error: { type: 'string' },
        },
      },
      render: (_args, value) => [{ type: 'text', text: renderResult(value) }],
    },

    async execute(args, exec) {
      const cwd = exec?.agent?.session?.header?.cwd ?? process.cwd()
      const python = await getPython()
      if (!python) {
        return {
          ok: false,
          error: 'no Python interpreter with opencv-python (cv2) and numpy was found. '
            + 'Install them (`pip install opencv-python numpy`) or point the plugin option '
            + '`python` at an interpreter that has them. The algorithm is not re-implemented '
            + 'in Node, so the tool cannot run without it.',
        }
      }
      // The report path lives in the temp directory: it is scratch, must not collide
      // between concurrent sessions, and must not appear in the user's output tree.
      const tmp = process.env.TEMP || process.env.TMPDIR || process.env.TMP || '.'
      const reportPath = `${tmp}/dsh-watermark-report-${process.pid}-${Date.now()}.json`
      const reportTarget = await fs.resolve(reportPath, { cwd, signal: exec?.signal })
      return await runRemoval(args, {
        subprocess,
        python,
        cwd,
        signal: exec?.signal,
        scriptPath,
        reportPath: fs.processPath(reportTarget),
        readFile: () => fs.readText(reportTarget, exec?.signal),
        statPath: async (p) => {
          try {
            return await fs.stat(await fs.resolve(p, { cwd, signal: exec?.signal }),
              exec?.signal)
          } catch {
            return undefined
          }
        },
      })
    },
  })

  ctx.effect(() => disposeTool, 'dsh-watermark: remove_watermark tool')
}

export default { name, inject, apply }
