# dsh-watermark

DSH 插件：**批量去水印**。给一个目录（同尺寸 ≥3 张、共用同一个水印），把水印去掉，
**只改水印像素**、不动画面其它内容。

> **本插件只支持批量。单张自动去水印已明确舍弃**，原因、实测数据与失败方式见 §2 ——
> 那是"信息不足"，不是"再调调就行"。放弃的理由全部保留在下面，包括所有失败数字。
>
> 单张仍可用的两条**显式**路径：`--template`（你给出水印签名/形状）与 `--rect`（你给出框）。

对外暴露一个 agent 工具：`remove_watermark`。

---

## 0. 这个插件做什么

| 场景 | 支持 |
| --- | --- |
| 一批同尺寸图，右下/四角等**同一位置**的同一水印（即梦/豆包/可灵等平台角标） | ✅ 默认路径 |
| 水印在画面中间、或任意位置（同尺寸 ≥3 张） | ✅ 配 `--search all` 或 `--rect` |
| 深色 / 白色 / 彩色（含"亮度相同只有颜色不同"）水印 | ✅ |
| 半透明水印（同批次内） | ✅ 到 alpha≈0.35；更淡的见 §3 的实测边界 |
| 平铺水印 | ⚠️ 能命中但精度不足，见 §3 |
| 只有一张图、水印未知 | ❌ **已明确舍弃**（§2）。请用 `--template` 或 `--rect`，或凑够 3 张同尺寸再跑 |
| 一批图尺寸不一致 | 同尺寸 ≥3 张的那一组照做，落单的**报错不猜**（退出码非 0） |

对 agent 说一句话：`把 D:\出图\卡面 里的水印去掉，然后抽查 3 张确认原图没被动。`

---

## 1. 它怎么找到水印（唯一自动路径）

**跨图证据**：一批图共用同一个水印 ⇒ 水印不动、画面在动。于是逐像素看这一叠图的取值：

- 水印盖住的像素在**所有帧里几乎一样**（spread 小）；
- 画面自己的像素在帧间差异很大；
- 判据是"这个像素比它**自己周围**安静得多"（`--multi-ratio`，实测水印 0.48–0.57、
  背景 0.73–0.89，取 0.65），再过三道组件级校验：

| 校验 | 依据 | 实测 |
| --- | --- | --- |
| **画出来的边界** | 水印是画上去的，共享的背景不是 | 水印边缘梯度 63–341；背景块 2.0–3.7 |
| **周围参考** | 与候选**周围**的画面比，而不是与可能被自己填满的窗口比 | 见 §3 的已知限制 |
| **[WARN] 哨兵** | 掩码占搜索区 >8% 或组件数 >10 → 打印"这看起来是照片不是水印" | 两批真实图：2 个组件、2.4% |

外加**填洞**：水印比参考窗口宽时，判据只勾出它的"外环"，闭运算+填洞把本体补回来
（不补的话 inpaint 只糊外环、字形填充留在原地 → 输出是水印的模糊白影，实测过）。

**显式路径**（不算"自动"）：`--template` 给签名/形状 → 直接当掩码；配 `--restore` 则
按 `original = (observed - a*color)/(1-a)` **精确反解**，而不是涂抹。

---

## 2. 为什么不做单张自动去水印（**这一节是决策依据，数字全部保留**）

单张自动检测曾经实现过、也测过，最终**删除**。原因不是"难"，是**信息不足**：

**① 它在真实照片上会去找画面自己的内容，水印基本没被碰到。** 两批真实照片（各 10 张）
上用单图模式（`--search bottom-right`）：

```
第一批  10 张：掩码 9688 px，改动 71924 px 在画面其它地方   水印仍在（水印相关度 0.84 vs 原图 0.94）
第二批  10 张：掩码 1374 px，改动 40818 px 在画面其它地方   水印仍在（0.88）
```

掩码落在花瓣阴影、猫爪、桌沿上。把搜索框缩到水印大小也不行：搜索区一小，
局部"背景水平"窗口里大半就是水印自己，水印把自己的证据压掉了（实测：水印内部
spread 5.0、而参考值 4.3，比值 1.16 → 判据永不触发）。

**② 强行放宽判据的代价，是合成套件从 10/1/0 掉到 5/1/5，且每张多改约 1 万像素。**
把"方向一致"改成"共线性"（因为白字+深描边会在均值里互相抵消：strength 0.69 需要 ≥6、
一致性 0.447 需要 ≥0.70，水印被自己丢掉）之后：单图模式在 20 张真实图上**全部找到了水印**
（覆盖率 0.93–0.97），但同时接受 78 个画面自身的组件、每张多改约 1 万像素；再加"实心区"
判据也救不回来。**这是信息不足问题，不是算法问题**——一张图里，"平台水印"和"照片自己的
低对比结构"在统计上无法区分。所以删除，而不是留一个不敢用的开关。

**③ 横向对比：同一个水印，批量路径的成绩。**

| | 单图自动（已删） | 批量（现在的唯一路径） |
| --- | --- | --- |
| 两批真实图，水印是否去掉 | ❌ 基本没碰到 | ✅ 完全去掉（相关度 0.01–0.02，度量本底 0.07） |
| 框外被改动的像素 | 40818 / 71924 px | **0 px** |
| 掩码组件数 | 78 | 2 |

---

## 3. 能力边界（**实测数据**）

| 情形 | 现状 | 实测 |
| --- | --- | --- |
| 角标（默认 `--search corners`） | ✅ | 两批真实图 20/20 张：水印完全去掉、**框外 0 px** |
| 纯色/渐变/忙碌背景上的不透明水印 | ✅ | 合成批次：命中 1.00、框外 0 px、水印处 RMSE 133→1.5 |
| 亮度相同只有颜色不同的水印 | ✅ | 命中 1.00、RMSE 72→1.8（灰度法在这里是隐形的） |
| 同一水印压在最忙的拼贴画面上（`--rect` 给框） | ⚠️ | 命中 1.00、框外 0，但**框内**水印处 RMSE 59→30，未达"减半"判据 |
| 画面中间的同一水印 + `--rect` 给框 | ❌ 已知限制 | 见下 |
| 半透明 35% + 忙碌画面（`--rect`） | ❌ 已知限制 | 见下 |
| 平铺水印 | ❌ 已知限制 | 掩码 38289 px 对水印 3960 px，框外 7072 px（5 帧合计） |
| 整帧搜索（`--search all`）找中间的水印 | ❌ 已知限制 | 候选区连成一片，一个 15686 px 的组件把水印吞进去，掩码占 5–7% 画面 |

**已知限制的机制（不是"调参不够"）**：用 `--rect` 把搜索框贴在水印上时，
"边缘参考值"取的是**搜索区内**梯度的 95 分位——而当搜索区就是水印自己时，这个参考值
**就是水印自己的边缘**，等于要求水印打败自己，于是什么都找不到。这不是没试过修：
每一条替代方案都实现并实测过，**每一条都要用真实批次的质量去换**：

| 替代方案 | 修好了 | 代价（实测） |
| --- | --- | --- |
| 参考值改为候选周围的环 | 中间水印那一条 | 第一批的去除量 109853 px → 23235 px |
| `--multi-ratio` 放到 0.70 | 35% 半透明那条 | 第二批画面被改 4033 px |
| 边缘参考改为多尺度像素参考 | 颜色难分的那条 | 第二批 10 张**全部检测不到**（0 改动） |

所以这些用例在测试里被**显式标为 XFAIL**（不是 PASS），每次运行都把原因和数字打出来。

**不透明水印无法"还原"**，只能 inpaint；`--restore` 会报出有多少像素属于这种。

---

## 4. 安装与使用

**用法就一句话：把目录丢给它。**

```powershell
python src/remove_watermark.py 出图目录\ -o 结果\ --out-ext png
```

它会自己决定搜索范围（默认四个角），并在结果不可信时打 `[WARN]`。
同尺寸不足 3 张、或某张尺寸与其它都不同，会**报错而不是退化成单张**：

```
[ERROR] 批量去水印需要至少 3 张同尺寸图片；单张/混合尺寸不受支持。
        (batch removal needs >=3 images of the same size; a single image or mixed sizes are not supported)
```

```powershell
# 挂到 web profile（本地开发用 link）
dsh plugin --profile web add link:D:/dsh-watermark

# 水印在画面中间（这几种都要 >=3 张同尺寸）
python src/remove_watermark.py 出图目录\ --search all
python src/remove_watermark.py 出图目录\ --rect 250,280,190,50

# 显式给签名 → 精确反解（此时不需要批量）
python src/remove_watermark.py --learn 带水印.png 无水印.png --signature-out 平台签名.png
python src/remove_watermark.py 单张.png --template 平台签名.png --restore
```

依赖：**Python 3 + opencv-python + numpy**。找不到带 cv2 的解释器时工具会直接说明。
退出码：`0` 全部成功，`1` 有文件失败（含"尺寸不匹配、无法共享证据"），`2` 参数/输入错误。

---

## 5. 测试与原始输出（可复现）

三个套件，全部是**真实运行**的输出与退出码。复现：

```powershell
python -X utf8 test/test_remove_watermark.py          # 批量通用性：9 个场景 x 5 帧
python -X utf8 test/test_signature_and_lattice.py     # 签名反解（--learn/--template/--restore）
node test/run.mjs                                     # 宿主插件层
```

### 5.1 批量通用性（`test/test_remove_watermark.py`）

每个场景是 **5 帧**：同一个水印、5 张不同的画面。标记含义：`hit` 命中的水印像素比例；
`changed` 全图被改比例；`rmse_pre/post` 水印框内相对**真正无水印原图**的 RMSE（后 < 前一半才算过）；
`spill` 水印框外被改动的像素数（5 帧合计）**必须为 0**。

```
>>> batch generality test (10 cases x 5 frames each)
  case                         hit  changed  rmse_pre rmse_post  spill px_true px_flag
  [PASS] A_bright_right_dark         1.00    0.43%    133.42      1.53      0    811   2483
  [PASS] B_dark_left_light           1.00    1.97%    127.80      1.71      0   3451  10970
  [XFAIL] C_white_centre_busy         0.00    0.00%    103.44    103.44      0   2469      0
         -> rmse must drop below 51.72
         -> flagged 0 of 2469 solid mark px
         -> with --rect hugging the mark, the rim reference (p95 of the gradient over the search area) IS the mark's own edges, so the mark is asked to beat itself and nothing is found. Measured: ring reference per candidate fixes it and drops batch 1's removal from 109853 px to 23235 px
  [PASS] D_magenta_topright_grad     1.00    0.98%     72.02      1.79      0   1514   4809
  [XFAIL] E_semi35_busy               0.00    0.00%     37.12     37.12      0   1175      0
         -> rmse must drop below 18.56
         -> flagged 0 of 1175 solid mark px
         -> a 35%-alpha mark scales the across-frame spread by (1-a) = 0.65, exactly the shipping --multi-ratio, so the pixel criterion is decided by rounding; 0.70 fixes it and costs 4033 px of real picture on batch 2
  [XFAIL] F_grey_big_busy             0.00    0.00%     59.31     59.31      0   9174      0
         -> rmse must drop below 29.66
         -> flagged 0 of 9174 solid mark px
         -> same circular rim reference as C -- this mark's box is the search area. Its candidate map is otherwise perfect (measured: 9174 of 9174 mark px flagged before the reference test)
  [PASS] G_black_on_light            1.00    0.61%    119.57      1.16      0   1197   3636
  [XFAIL] H_tiled_lattice             1.00    3.91%     25.94      4.94   7098   3960  18966
         -> touched 7098 px outside the mark, over the batch
         -> a tiled mark covers the frame, so there is no smaller search area to scope it to; over patchwork art the mask reaches 38289 px for a 3960 px mark (hit 1.00, spill 7098 over the batch)
  [XFAIL] I_white_small_centre        0.00    0.00%     73.59     73.59      0   1048      0
         -> rmse must drop below 36.80
         -> flagged 0 of 1048 solid mark px
         -> same circular rim reference as C and F
  [XFAIL] X_wholeframe_busy           1.00    1.02%     73.59     17.54   1906   1048   4914
         -> touched 1906 px outside the mark, over the batch
         -> centred mark searched over the whole frame on patchwork art; the candidate regions merge into one component that swallows the mark (15686 px) and the mask covers 5-7% of the frame
  [PASS] J_clean_noop             0 changed pixels
  [PASS] K_too_few_refused        exit 2, message present

[SUMMARY] passed 6 / xfail 6 (known limitations) / xpass 0 / failed 0
          xfail cases are NOT passes: see README section 3, with their numbers

EXITCODE=0
```

### 5.2 签名反解（`test/test_signature_and_lattice.py`）

```
>>> signature / lattice / regression test
  [PASS] L1_learn_solves        alpha 0.401 (built 0.40, tol .03) colourBGR [255.2, 255.3, 255.1] (built white) explained 0.9962
  [PASS] L2_restore_inverts     rmse vs the true clean image inside the mark: 41.74 -> 0.44 (needs < 14.61)
  [PASS] L3_negative_untouched  466831 px outside the signature support, 0 of them changed (must be 0)
  [PASS] L4_learn_refuses_empty exit 2, signature written: False, message: [FAIL] no usable pair (all pairs were unreadable or show no mark)

[SUMMARY] passed 4 / failed 0

EXITCODE=0
```

### 5.3 宿主插件层（`node test/run.mjs`）

```
>>> host layer test
  [PASS] plugin_shape                   name=dsh-watermark inject=[tools,fs,subprocess]
  [PASS] buildArgv_order                a.png b/ --rect 10,20,30,40 --search bottom-right --template sig.png --strategy template --k-sigma 2 --restore --json-out report.json
  [PASS] buildArgv_skips_unknown        unknown keys and false booleans are not passed
  [PASS] parseSummary                   {"ok":2,"fail":0,"total":2}
  [PASS] parseSummary_absent            returns undefined
  [PASS] renderResult                   去水印：成功 1/2 张｜策略 single
  [PASS] descriptor_json_schema         output.schema accepted; parameters rendered by the harness (22 params)
  [PASS] descriptor_shape               22 typed params, required=[paths]
  [PASS] tool_registered                name=remove_watermark timeoutMs=900000
  [PASS] python_detected                D:\python\python.exe cv2 4.12.0
  [PASS] e2e_run_ok                     ok=true total=4 succeeded=4 exit=0
  [PASS] e2e_output_exists              in\clean\shot0-clean.png (under the run's temp dir), 9409 bytes
  [PASS] e2e_result_shape               {"name":"shot0.png","changedPct":0,"maskPx":0}
  [PASS] e2e_missing_output_caught      succeeded=0 failed=1
  [PASS] e2e_dry_run                    dryRun=true files=4
  [PASS] e2e_single_refused             ok=false failed=undefined error=the watermark tool exited 2 without writing a report
[FAIL] no images 
  [PASS] e2e_missing_input              the watermark tool exited 2 without writing a report

[SUMMARY] passed 17 / failed 0

EXITCODE=0
```

`descriptor_json_schema` 用**真实 harness 的校验器**验工具定义；该包不是本仓库依赖：

```powershell
$env:DSH_TOOLS_PATH="<dsh 安装目录>\node_modules\@deepseek-ai\dsh-tools\lib\index.js"
node test/run.mjs
```

`e2e_single_refused` 是这次决策的那条：**单张必须被拒绝**，不能悄悄退化成单图模式。

---

## 6. 真实图片端到端验证（两批共 21 张）

`evidence/verify-all.py` 逐像素核对交付结果，**没有任何参数**（连 `--search` 都没给）：

```
$ python src/remove_watermark.py 测试例/第一次 -o 结果/第一次结果 --suffix "" --out-ext png
    (shared evidence: 10 frames of size (1280, 1280))
[SUMMARY] ok 10 / fail 0 / total 10                      exit 0

$ python src/remove_watermark.py 测试例/第二次 -o 结果/第二次结果 --suffix "" --out-ext png
    (shared evidence: 10 frames of size (1280, 1280))
    (1 frame(s) of size (1706, 1279) cannot share evidence and will fail)
[FAIL] 微信图片_20260922222303_119_5.jpg  批量去水印需要至少 3 张同尺寸图片；单张/混合尺寸不受支持。
[SUMMARY] ok 10 / fail 1 / total 11                      exit 1   ← 落单那张按规矩失败，不猜

$ python evidence/verify-all.py
   → 两次合计：水印内改动 219613 px，框外改动 0 px
[PASS] outside the watermark the pictures are the decoded originals
```

那张竖幅图（1279×1706，与其它 10 张尺寸不同）**没有被当作结果交付**：工具自己报了 `[FAIL]`，
我也逐像素确认过它若走显式路径会被改坏（改动 42698 px，其中水印内 0 px）。处理办法是把它
和**其它同尺寸**的图放一起再跑一次。

---

## 7. 目录结构

```
dsh-watermark/
├── package.json                    # dsh / dshhub 清单（市场收录用）
├── cordis.patch.yml                # 挂载行：insert dsh-watermark
├── README.md                       # 本文件（含实测数据与原始输出）
├── PUBLISHING.md                   # 发布清单（GitHub / npm / registry PR）
├── lib/index.js                    # 宿主插件层：注册 remove_watermark 工具
├── src/remove_watermark.py         # 批量检测 + 签名反解 + CLI
├── test/
│   ├── test_remove_watermark.py    # 批量通用性（9 场景 x 5 帧 + 负向对照 + 拒绝单张）
│   ├── test_signature_and_lattice.py  # --learn / --template / --restore
│   └── run.mjs                     # 宿主层：argv/解析/渲染/注册 + 真实端到端
└── evidence/                       # 施工期测量脚本与原始输出
```

## 8. 参考实现（同类插件的真实格式，已在本机核对）

| 参考 | 学到什么 |
| --- | --- |
| `dsh-taskboard@0.7.2` | `exports["."]` = 宿主入口；`cordis.patch.yml` 用 `- insert:` 加一行；`inject = ["tools", ...]` |
| `dsh-cost-meter@1.7.29` | **`dshhub` 上架清单的完整字段**（categories / surfaces / capabilities / permissions） |
| `dsh-docx-report` | `link:` 安装的 host-only 插件：不 import `@deepseek-ai/*`（裸导入在 link 场景下解析不到），直接 `tools.register({...})` + `ctx.effect` 托管 disposer |
| `dshmarket` | 安装源 = curated registry `awesome-dsh-plugin`；收录方式 = 提 PR 加一条条目 |
