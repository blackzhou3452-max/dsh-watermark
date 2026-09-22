# dsh-watermark

DSH 插件：**批量去水印**。给一个目录，把水印去掉，**只改水印像素**、不动画面其它内容。

> 状态：**宿主插件层已实现并在真实 harness 里装过、核心算法已重写并用合成样本实测、测试原始输出见 §5**。
> 已实现的：启发式单图检测、精确签名反解（`--learn` / `--restore`）、多图叠加、平铺周期自动估计。
> 未达标并已定位的：**1 个用例**（半透明 35% 水印压在噪声忙碌画面上），见 §3。

对外暴露一个 agent 工具：`remove_watermark`。

---

## 0. 这个插件做什么

| 场景 | 支持 |
| --- | --- |
| 即梦 / 可灵 / 豆包等国产生图平台右下角角标 | ✅ |
| 任意位置（中间、四角、边缘） | ✅ |
| 深色水印、彩色水印（含"亮度相同只有颜色不同"的水印） | ✅ |
| 平铺水印 | ✅ 周期可自动估计 |
| 一批图**同一个水印、同一位置** | ✅ 且效果最好（`--strategy multi`） |
| 你知道水印长什么样（有带 alpha 的 PNG，或有一对"带水印/无水印"图） | ✅ **精确反解**，不是涂抹 |
| 单张图、水印未知、画面又很忙 | ⚠️ 见 §3 逐条实测边界 |

说一句话就能用：

> 把 `D:\出图\卡面` 里的水印去掉，然后抽查 3 张确认原图没被动。

---

## 1. 三种策略（按可靠性排序，这是本插件的核心设计）

### 策略 A：精确签名（唯一算得上"还原"的模式）

水印是**半透明叠加**时（绝大多数平台角标都是），它就是一个确定的数学关系：

```
observed = (1 - a) * original + a * color        (逐像素，a 与 color 由水印决定)
=>  original = (observed - a * color) / (1 - a)
```

被盖住的像素可以**解出来**。签名的来源：

| 来源 | 做法 | 状态 |
| --- | --- | --- |
| `--learn MARKED CLEAN` | 从一对"同图带水印/无水印"反解 `a` 与 `color`，写成可复用签名 PNG | ✅ 实测：真值 0.40 解出 0.401，模型解释度 0.9962（§5 L1） |
| `--template sig.png --restore` | 用签名精确反解 | ✅ 实测：水印处 RMSE 41.74 → **0.44**（§5 L2） |
| `--template shape.png`（无 alpha） | 单图自标定：先 inpaint 估背景，再用同一套最小二乘拟合 `a`/`color`（结果会打印出来） | ✅ 估计值，不是测量值 |

> **不透明像素（a≈1）没有可恢复的原图**——除法会炸，任何算法都变不出被完全盖住的内容。
> 这些像素会被列出来、改用 inpaint，并在结果里报数量，不藏。

### 策略 B：多图叠加（不需要知道水印长什么样）

一批图共享同一个水印位置时：水印不动、画面在动 ⇒ 逐像素看这一叠图的取值分布，
水印覆盖处分布被压缩。`--strategy multi`（需要 ≥3 张同尺寸图）。

### 策略 C：单图启发式（最通用、最不可靠）

不知道水印、只有一张图时，靠"异常"猜。三重判据，全部围绕**同一个物理事实**：

水印是叠加 ⇒ 残差向量 `r = 观测 − 局部背景 = a*(color − 背景)`，
在整个水印上**方向一致**、强度可比；而画面纹理的残差方向互相抵消。

1. **局部背景用中值而不是均值**——均值窗口会被水印自己的像素污染，窗口只要不比水印大很多，
   残差就会朝 0 塌陷（这是旧版 README §3 记录、但一直没修的那条：大的洋红水印在半径 31 的窗口下
   检测到 **0** 像素）。中值返回窗口内的多数值，水印占比不到一半时背景估计仍然落在背景上。
2. **残差是三维向量**，不是灰度差——亮度与背景相同的彩色水印在灰度里是隐形的，
   在向量里 `r ≈ (+150, −150, +150)`。实测用例 D（洋红压在亮度相近的渐变上）就是被这一条救回来的。
3. **种子必须比"它自己周围的残差水平"强**，不是比 0 强——忙碌画面上画面自身的残差就有 40–90，
   一个绝对阈值分不开。水印的笔画相对**紧邻环境**是异常（实测比值 3–6），画面的色块不是（比值 ~1）。
4. **组件判决**：面积、残差强度、**方向一致性**（一个水印整体压在同一侧；纹理不断翻向），
   以及 **纹理塌陷**——不透明水印会把画面纹理换成纯色，半透明水印把它按 (1−a) 压缩，
   而画面自己的色块保留着自己的纹理。这一条是单图版的策略 B，也是忙碌画面上准确率的来源。

最后接 `cv2.inpaint`（TELEA/NS），水印被它自己的周围替换——不是灰块，不是模糊补丁。

---

## 2. 算法相对于 v0.1 的实测变化（同一套用例）

第一版（`src/remove_watermark.py` 的旧实现）在本仓库的用例上跑不起来：单图路径有一个
`args.sat_min` 未定义的崩溃，随后又暴露出 `core_delta` / `sign_consistency` 两个同样未定义的选项。
把这三个缺陷补掉之后，才是**真正的算法基线 1/11**，然后才是改进：

| | 基线（补掉崩溃后） | 现在 |
| --- | --- | --- |
| 合成用例通过 | **1 / 11** | **10 通过 / 1 已知限制** |
| 单图检测（用例 A–I） | 4 个完全检测不到（B/D/G 命中 0，E 命中 0 且 16.6% 误改） | 8 个命中 1.00，误差全部为 0 溢出 |
| 干净图（负向对照） | 失败：工具崩了，输出没生成 | **0 像素改动** |
| 大号半透明水印 | 旧版 README §3 记录为"检测到 0 像素"（塌陷），一直没修 | 命中 0.77（§5 B1 回归用例，12017 px 水印） |
| 平铺周期 | 必须手工 `--period` | `--period auto` 自动估出 `x=260 y=200`（真值 260/200） |

基线为什么失败（诊断脚本 `evidence/diag.py` 的原始结论）：旧实现的"局部噪声尺度"用的是
`blur(|残差|)`——**水印自己的边缘在抬高那个本该发现它的阈值**。9 个用例里
`contrast=0`：方差项在所有用例上都没投出过一票，检测器恰好在信号最强的地方把自己关掉了。

---

## 3. 能力边界（**实测数据，不要当成"万能"**）

| 情形 | 现状 | 实测 |
| --- | --- | --- |
| 纯色/渐变背景 + 不透明水印（任意深浅、任意位置） | ✅ | 命中 1.00，误差 0 溢出，水印处 RMSE 134→1.1 / 127→1.3 |
| 亮度相同、只有颜色不同的水印 | ✅ | 命中 1.00，RMSE 75.4→1.45（灰度法在这里是隐形的） |
| 忙碌画面 + 不透明水印 | ✅ | 命中 1.00，RMSE 59.5→11.3，0 溢出 |
| 忙碌画面 + 中灰不透明大水印 | ✅ | 命中 1.00，RMSE 71.1→17.7 |
| 平铺水印 | ✅ | 命中 1.00，周期自动估计正确 |
| **半透明 35% + 噪声忙碌画面** | ❌ **已知限制** | 命中 **0.00**（默认参数下一个像素都不动；0 溢出） |
| 单张图、水印未知、画面纹理很重 | ⚠️ | 缩窄 `--search` / 给 `--rect` 是最省事的办法；再不行只能靠策略 A/B |

### 关于那唯一一个 ❌（不遮不掩）

用例 `E_semi35_busy`：白色 `AI`，alpha=0.35，6px 笔画，压在带噪声的色块画面上。

- 它的**单像素**残差是 `0.35 × (255 − 背景)` ≈ 19–79，而该画面的噪声 σ≈23 ⇒ 信噪比 **0.8–3.4**，
  单像素阈值怎么调都分不开；
- 它的**聚合**证据（纹理塌陷）在 6px 笔画的抗锯齿边缘上被摊平了：笔画太细，
  边缘自身的 Laplacian 能量比内部的"压缩"更显眼。实测：笔画上平滑后的能量比中位数 0.94，
  只有 34.6% 的像素低于 0.8 的检测阈值 ⇒ 碎片化 ⇒ 组件显著性 z 值 1.9–5.1，低于 6.0 的门槛。

所以它被显式标成 `XFAIL`，**不计入通过**：测试输出里写的是
`[XFAIL] E_semi35_busy` 和 `xfail 1 (known limitations)`，汇总行还额外打印
"xfail cases are NOT passes"。README 不把它算作通过。

**可用的补救**：这类水印换成 `--strategy multi`（同一批图 watermarked 多张）能直接解决——
它用的是"画面在动、水印不动"的跨图证据，不受单图噪声限制。

---

## 4. 安装与使用

```powershell
# 挂到 web profile（本地开发用 link）
dsh plugin --profile web add link:D:/dsh-watermark

# 或者从 GitHub 装（发布之后）
dsh plugin --profile web add github:<owner>/dsh-watermark
```

依赖：**Python 3 + opencv-python + numpy**（`pip install opencv-python numpy`）。
算法**没有**用 Node 重写：它已经在 Python 里被合成样本实测过，重写要重新挣一遍这些证据；
宿主层只负责找解释器、拼参数、收 JSON 报告、核对产出真的落盘。
找不到带 cv2 的解释器时工具会直接说明，而不是静默失败。

装好后对 agent 说一句话即可；也可以直接用命令行：

```powershell
python src/remove_watermark.py 出图目录\                       # 最省事
python src/remove_watermark.py 出图目录\ --search bottom-right  # 只在右下角找
python src/remove_watermark.py 出图目录\ --dry-run --mask-out-dir masks\   # 先看会改哪里

# 学一次签名，之后一直复用（精确反解）
python src/remove_watermark.py --learn 带水印.png 无水印.png --signature-out 平台签名.png
python src/remove_watermark.py 出图目录\ --template 平台签名.png --restore

python src/remove_watermark.py 出图目录\ --period auto          # 平铺水印
python src/remove_watermark.py 一批图\ --strategy multi          # ≥3 张同位置
python src/remove_watermark.py 一批图\ --strategy multi --rect x,y,w,h -o 结果 --out-ext png
                                                                 # 输出无损 PNG，不重压 JPEG
```

退出码：`0` 全部成功，`1` 有文件失败（或自动周期不可信），`2` 参数/环境错误。

---

## 5. 测试与原始输出（可复现）

三个套件，全部是**真实运行**的输出与退出码。复现：

```powershell
python -X utf8 test/test_remove_watermark.py          # 通用性：11 个合成用例
python -X utf8 test/test_signature_and_lattice.py     # 签名反解 / 周期估计 / 回归
node test/run.mjs                                     # 宿主插件层
```

### 5.1 通用性用例（`test/test_remove_watermark.py`）

```
>>> synthetic generality test (9 cases)
  case                         hit  changed  rmse_pre rmse_post  spill px_true px_flag
  [PASS] A_bright_right_dark         1.00    0.34%    134.42      1.14      0    811   1859
  [PASS] B_dark_left_light           1.00    1.68%    127.30      1.30      0   3451   8745
  [PASS] C_white_centre_busy         1.00    1.47%     59.51     11.41      0   2469   7142
  [PASS] D_magenta_topright_grad     1.00    0.78%     75.41      1.45      0   1514   3807
  [XFAIL] E_semi35_busy               0.00    0.00%     38.94     38.94      0   1175      0
         -> rmse must drop below 19.47
         -> flagged 0 of 1175 solid mark px
         -> known limitation, documented in README section 3
  [PASS] F_grey_big_busy             1.00    2.55%     71.13     17.74      0   9174  12281
  [PASS] G_black_on_light            1.00    0.46%    118.72      0.82      0   1197   2516
  [PASS] H_tiled_lattice             1.00    2.37%     23.17      2.80      0   3960  11448
  [PASS] I_white_small_centre        1.00    0.67%     80.81     18.64      0   1048   3227
  [PASS] J_clean_noop             0 changed pixels
  [PASS] K_multi_image            5/5 outputs (exit 0)

[SUMMARY] passed 10 / xfail 1 (known limitations) / xpass 0 / failed 0
          xfail cases are NOT passes: see README section 3 for each one, with its measured numbers
EXITCODE=0
```

列含义：`hit` = 命中的水印实心像素比例；`changed` = 全图被改动的像素比例；
`rmse_pre/post` = 水印框内相对**真正的无水印原图**的 RMSE（后 < 前的一半才算过）；
`spill` = 水印框（外扩 14px）之外被改动的像素数，**必须为 0**；
`px_true/px_flag` = 真实水印像素数 / 工具标记的像素数。

`spill` 全 0、`J_clean_noop` 0 改动，就是纪律里要求的**负向对照**：无水印的图一个像素都不许动。

### 5.2 签名反解 / 周期估计 / 回归（`test/test_signature_and_lattice.py`）

```
>>> signature / lattice / regression test
  [PASS] L1_learn_solves        alpha 0.401 (built 0.40, tol .03) colourBGR [255.2, 255.3, 255.1] (built white) explained 0.9962
  [PASS] L2_restore_inverts     rmse vs the true clean image inside the mark: 41.74 -> 0.44 (needs < 14.61)
  [PASS] L3_negative_untouched  466831 px outside the signature support, 0 of them changed (must be 0)
  [PASS] L4_learn_refuses_empty exit 2, signature written: False, message: [FAIL] no usable pair (all pairs were unreadable or show no mark)
  [PASS] P1_period_auto         hit 1.00, spill 0, exit 0 | (auto period: x=260 score=0.588 | y=200 score=0.388 -> accepted)
  [PASS] B1_big_semi_transparent hit 0.77 on a 12017 px half-transparent mark (the old detector: 0 flagged px)
  [PASS] B2_negative_untouched  the same busy frame without the mark: 0 changed pixels

[SUMMARY] passed 7 / failed 0
EXITCODE=0
```

- **L1**：真值 alpha=0.40、白色，解出 0.401 / `[255.2, 255.3, 255.1]`，模型解释度 0.9962——这是**绝对校验**，不是"跑通了"。
- **L2**：用学到的签名反解，水印处 RMSE 41.74 → **0.44**（判据是 < 14.61）。
  修之前是 41.74 → 47.13：签名是按水印自身尺寸写的，缺了 sidecar 里的位置就会被拉伸到整幅，
  **越修越糟还报成功**。现在没有 sidecar 就报错，不猜。
- **L3**：签名覆盖范围之外 466831 像素，**0 个被改动**（逐位相同）。
- **P1**：`--period auto` 从图里估出 `x=260 y=200`（真值 260/200），自相关峰值 0.588 / 0.388，
  命中 1.00、0 溢出；估不出来时工具会**拒绝猜**并退出 1，而不是压一个错的网格上去改干净像素。
- **B1**：README §3 记录过的那个"大号半透明水印检测塌陷"，现在命中 0.77。

### 5.3 宿主插件层（`node test/run.mjs`）

```
>>> host layer test
  [PASS] plugin_shape                   name=dsh-watermark inject=[tools,fs,subprocess]
  [PASS] buildArgv_order                a.png b/ --rect 10,20,30,40 --search bottom-right --template sig.png --strategy template --k-sigma 2 --restore --json-out report.json
  [PASS] buildArgv_skips_unknown        unknown keys and false booleans are not passed
  [PASS] parseSummary                   {"ok":2,"fail":0,"total":2}
  [PASS] parseSummary_absent            returns undefined
  [PASS] renderResult                   去水印：成功 1/2 张｜策略 single
  [PASS] descriptor_json_schema         output.schema accepted; parameters rendered by the harness (26 params)
  [PASS] descriptor_shape               26 typed params, required=[paths]
  [PASS] tool_registered                name=remove_watermark timeoutMs=900000
  [PASS] python_detected                D:\python\python.exe cv2 4.12.0
  [PASS] e2e_run_ok                     ok=true total=1 succeeded=1 exit=0
  [PASS] e2e_output_exists              in\clean\shot-clean.png (under the run's temp dir), 12073 bytes
  [PASS] e2e_result_shape               {"name":"shot.png","changedPct":7.3802,"maskPx":1417,"confidence":3.21}
  [PASS] e2e_missing_output_caught      succeeded=0 failed=1
  [PASS] e2e_dry_run                    dryRun=true files=1
  [PASS] e2e_missing_input              the watermark tool exited 2 without writing a report

[SUMMARY] passed 16 / failed 0
EXITCODE=0
```

`descriptor_json_schema` 用**真实 harness 的校验器**（`@deepseek-ai/dsh-tools` 的
`assertSupportedJsonSchema` 与参数渲染）验工具定义。该包不是本仓库依赖，所以：

```powershell
# 想在别处也真验一遍（否则该项会明确打印 skipped，不会假装通过）
$env:DSH_TOOLS_PATH="<dsh 安装目录>\node_modules\@deepseek-ai\dsh-tools\lib\index.js"
node test/run.mjs
```

`e2e_missing_output_caught` 是防"假成功"的那一条：子进程报告写成功、但文件不在盘上，
必须被报成失败（这条如果回归，插件就会开始骗人）。
`e2e_run_ok` 等用例走的是**真实**路径：真找解释器、真 spawn、真写图，断言落在磁盘上。

### 5.4 安装验证（真实 harness）

```
$ dsh plugin --profile wmtest add link:D:/dsh-watermark
dependencies:
+ dsh-watermark link:D:/dsh-watermark
dsh: initialized profile wmtest at D:\dsh-home\profiles\wmtest

$ dsh --profile wmtest --dump-config | grep -A1 dsh-watermark
# == dsh-watermark
- id: dsh-watermark
  name: dsh-watermark
```

即：`dsh plugin add` 同时写好依赖与 `dsh.profile.bundles`，`cordis.patch.yml` 的挂载行
真的进了合成后的 profile 树。（这个 `wmtest` 是一次性验证用 profile，验完已经删除，
没有动你的 `web` profile。）

### 5.5 真实图片实测：10 张豆包 AI 生成图（`测试例/` → `结果/`）

输入是 10 张真实 JPEG（1280×1280），水印都是右下角同一位置的「豆包AI生成」：
白字 + 细灰描边，字形范围 x 1058–1257 / y 1211–1253，**10 张完全一致**。

先测清楚这是什么水印：字形填充色在 10 张里都是 **248–250**（标准差 ~2），而水印下方的
背景在 **145–198** 之间变化 ⇒ 反推 alpha ≈ 0.9，即**近乎不透明**。两个后果：
① 只能用 inpaint（除以 1−a = 0.1 会把噪声放大 10 倍，"精确反解"在这里没有意义）；
② 跨图看水印像素几乎恒定 ⇒ 这正是 `--strategy multi` 最擅长的情况。

**先说结论：这个工具在这批图上的真实成绩单**（`evidence/mode-matrix.py`，
全部是工具自己找到的，没有人工喂给它任何框）：

| 配置（工具自主） | 掩码 px | **动到画面其它内容** | **水印还在吗** | 退出码 |
| --- | --- | --- | --- | --- |
| `single --search bottom-right` | 9688 | **71924 px** | **在**（相关度 0.84 / 原图 0.94） | 0 |
| `single --search all` | 75916 | **803914 px** | **在**（0.84） | 0 |
| **`multi --search bottom-right`** | **11091** | **0 px** | **没了**（0.01） | 0 |
| `multi --search all` | 26432 | 146361 px | 没了（0.01） | 0 |
| `multi --period auto` | 11091 | 0 px | 没了（0.01） | 0 |

"水印还在吗"不是靠眼看：10 张图的水印是同一个固定图案，把 10 张原图逐像素取中值就能保住
水印、抵消掉各不相同的背景；拿这个模板去相关每一张输出的水印区域，**原图 0.94 → 单图模式
0.84（图案还在）→ multi 模式 0.01（结构没了）**。

单图模式在这批真实照片上**彻底失败**，失败方式值得记录：掩码落在这 10 张照片**各自的内容**
上（花瓣阴影、猫爪、桌沿），铺满右下象限，水印基本没被碰到；把搜索框缩到水印大小也不行
——搜索区一小，局部"背景水平"中值窗口里大半就是水印自己，种子被自己压掉了。
这就是 §3 里"单张图、水印未知、画面纹理很重 ⚠️"那条边界的真实现场。

**但上表也暴露了 multi 模式的一个真 bug，这次修掉了。** 修之前 multi
（`--search bottom-right`）的掩码是 11674 px = 水印 11091 + **水印上方 100 px 处一块
583 px 的平背景**——这 10 张照片恰好都在那儿是同一片平墙。multi 路径的掩码原来是个**裸阈值，
完全没有组件级校验**（单图路径有，multi 路径没有），所以那块平背景被原样涂掉：10 张一共
**4995 px 的画面被改**，正是这个工具保证不做的事。

修法用的是"水印有、背景块没有"的那条性质：**画出来的边界**。在 10 张的逐像素中值图上量
组件边缘的平均梯度——那块平背景是 **3.6**（比全图中位数 8.7 还低），而每个字都是
**63–108**，差 20 倍，不是勉强分开：

| 候选组件 | 面积 | 边缘梯度 | 结论 |
| --- | --- | --- | --- |
| 平背景块 | 583 px | 3.6 | 丢弃（没有画出来的边界） |
| 「豆」等 9 个字 | 189–708 px | 63–108 | 保留 |

（先试过"组件与周围环的亮度差"，是错的统计量：这个水印的白字压在浅色背景上，亮度差只有
9–12 级，而背景块是 0——能分开这两者的阈值是碰运气，边界测试则是 20 倍。）

修完之后的正式交付（**工具自主，无人工框**）：

```
$ python -X utf8 src/remove_watermark.py 测试例 --strategy multi --search bottom-right \
      -o 结果 --suffix "" --out-ext png
    (multi: stacking 10 frames of size (1280, 1280))
>>> remove_watermark: 10 file(s) | strategy=multi search=bottom-right rects=-
    [OK]   微信图片_20260922215822_109_5.jpg inpainted 11091 px (0.677% of frame)
    ...
[SUMMARY] ok 10 / fail 0 / total 10
EXITCODE=0
```

掩码 11091 px，bbox x 1052–1263 / y 1205–1259，正好贴住字形。逐像素复核
（`evidence/verify_results.py`，可复跑）：

```
image                changed    inside   outside
215822_109_5           11054     11054         0
...（10 张全部如此）
total changed inside the mark box: 108797
total changed OUTSIDE the mark box: 0
[PASS] outside the watermark, the pictures are byte-for-byte the decoded originals
```

即**框外一个像素都没动**。视觉上：浅色背景那几张看不出处理痕迹；深色木纹那张在 2× 放大下
能看到笔画处极轻微的"过平滑"——填充区灰度标准差 18.2 → 9.1，已经低于旁边桌面的 11.7；
`--radius 10/15` 只再好一点点（8.7 / 8.4），所以默认 5 就够。

两条诚实的备注：
1. `multi --search all` 仍然会动到画面（146361 px）：全图范围内还有别的区域在 10 张里恰好
   相似又带边界。`--search` 缩窄窗口就是为此存在的，工具自己的帮助文本也这么写。
2. `--out-ext png` 是这次为此加的：输入是 JPEG，若按输入扩展名写回 JPEG，整幅图会被重新
   压缩一遍，输出就不再是"工具改了什么"的记录，框外也不可能逐位相同。


---

## 6. 目录结构

```
dsh-watermark/
├── package.json                    # dsh / dshhub 清单（市场收录用）
├── cordis.patch.yml                # 挂载行：insert dsh-watermark
├── README.md                       # 本文件（含实测数据与原始输出）
├── PUBLISHING.md                   # 发布清单（GitHub / npm / registry PR）
├── LICENSE                         # MIT
├── lib/
│   └── index.js                    # 宿主插件层：注册 remove_watermark 工具
├── src/
│   └── remove_watermark.py         # 核心算法 + CLI（策略 A/B/C + --learn + 周期估计）
├── test/
│   ├── test_remove_watermark.py    # 11 个合成用例（通用性，含负向对照）
│   ├── test_signature_and_lattice.py  # 签名反解 / 周期估计 / 回归用例
│   └── run.mjs                     # 宿主层：argv/解析/渲染/注册 + 真实端到端
└── evidence/                       # 施工期测量脚本与原始输出（README 引用的在这里）
```

---

## 7. 已知的坑（照抄 §3，别当成"全部通过"）

1. **`E_semi35_busy` 不过**：半透明 35% + 噪声忙碌画面，默认参数下一个像素都不动。标为 XFAIL，不计入通过。
2. **单图启发式不可能对所有水印都稳**：没有额外信息时，某些画面上的水印与内容在统计上不可区分。
   定位是"默认走最可靠的路（A/B），启发式兜底并把统计报出来"，所以请用 `--dry-run` + `--mask-out-dir` 先看。
3. **不透明水印无法"还原"**，只能 inpaint；`--restore` 会报出有多少像素属于这种。
4. **`--learn` 需要同一张图的带水印/无水印一对**（同一内容、同一尺寸）。平台不允许导出无水印版时，用 `--template` 或 `--strategy multi`。
5. **`--period auto` 估不出来会直接失败（退出 1）**，不会硬猜：错的网格会去改干净像素。

## 8. 参考实现（同类插件的真实格式，已在本机核对）

| 参考 | 学到什么 |
| --- | --- |
| `dsh-taskboard@0.7.2` | `exports["."]` = 宿主入口；`cordis.patch.yml` 用 `- insert:` 加一行；`inject = ["tools", ...]` |
| `dsh-cost-meter@1.7.29` | **`dshhub` 上架清单的完整字段**（categories / surfaces / capabilities / permissions） |
| `dsh-docx-report` | `link:` 安装的 host-only 插件：不 import `@deepseek-ai/*`（裸导入在 link 场景下解析不到），直接 `tools.register({...})` + `ctx.effect` 托管 disposer |
| `dshmarket` | 安装源 = curated registry `awesome-dsh-plugin`；收录方式 = 提 PR 加一条条目 |
