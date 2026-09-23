# 发布清单

**已完成的**：本地仓库（8 个提交、工作区干净）、推送
[GitHub 仓库](https://github.com/blackzhou3452-max/dsh-watermark)（公开，远端 `main` = `8658bf2`）、
`package.json` 的 `dsh.bundle` 与根目录 `cordis.patch.yml`、从 GitHub 源实际安装并跑通。

下面两步需要你在网页上操作。步骤已按
[awesome-dsh-plugin 的 contributing.md](https://github.com/awesome-dsh-plugin/awesome-dsh-plugin/blob/main/contributing.md)
核对（2026-09-22 读取）。

> ⚠️ 更正：本文件早期版本说收录是"往 registry 的 JSON 数组里插一条"——**那是错的**。
> 收录格式是**一个插件一个 YAML 文件**。`publish/registry-entry.json` 保留作参考，
> 那是构建后的成品快照格式，**不是投稿格式**；投稿用 `publish/awesome-dsh-plugin.yml`。

---

## 1. 加 topic（30 秒，网页操作）

仓库页 → 右侧 **About** 齿轮 → **Topics** → 填 **`dsh-plugin`**（可再加 `deepseek-harness`）。

contributing.md 把这条列为收录要求之一。

---

## 2. 提 PR 进市场

### ⚠️ 先等一天

CI 第 3 项是**仓库创建满 1 天**，自动检查。仓库建于 2026-09-22，**当天提 PR 会被 CI 拒**，
明天起再提即可（重提不会有任何影响）。

### 然后

1. fork [awesome-dsh-plugin](https://github.com/awesome-dsh-plugin/awesome-dsh-plugin)；
2. 新建**一个文件**：`data/plugins/blackzhou3452-max__dsh-watermark.yml`，
   内容照抄 `publish/awesome-dsh-plugin.yml`（YAML 已用 js-yaml 验证可解析、字段齐全、
   category 在合法取值内）：

```yaml
url: https://github.com/blackzhou3452-max/dsh-watermark
name: blackzhou3452-max/dsh-watermark
category: tools
description:
  en: 'Batch watermark removal for a set of images that share one mark: ...'
  zh: '批量去水印：需要同一尺寸的图片 ≥3 张且共用同一个水印，...'
```

3. 提 PR，**只加这一个文件**。

### 几条容易踩的

| 要求 | 我们的情况 |
| --- | --- |
| 文件名 = `<owner>__<repo>.yml` | `blackzhou3452-max__dsh-watermark.yml` |
| `url` 与仓库完全一致 | ✅ |
| `package.json` 声明 `dsh.bundle`（只声明 `dsh.client` 会被拒） | ✅ `dsh.bundle.patch` |
| 根目录有 `cordis.patch.yml` | ✅ |
| 仓库满 1 天 | ⏳ 明天 |
| 加 `dsh-plugin` topic | 第 1 步 |
| 描述里含 `: `（冒号+空格）必须加引号 | ✅ en 描述已加引号 |
| 描述必须与代码相符（维护者逐句核对） | ✅ 只写批量能力；**不写单张、不写平铺**——一条已删除、一条是已知限制（README §3） |
| 不要手工编辑两个 README（脚本生成的） | 只加那一个 yml |
| 不要碰别人的条目 | 只加自己那一条 |
| 一个 PR 最多 3 条 | 我们 1 条 |

---

## 3. npm 发布（**可选，与收录无关**）

registry 的 176 条条目**全部**用 `github:` 安装，没有一条依赖 npm；contributing.md 也写明
"listing is unaffected either way"。发 npm 的唯一好处是市场能显示下载量。

想发的话：

```powershell
cd D:\dsh-watermark
npm login          # 打开浏览器授权；或 npm login --auth-type=legacy 走账号密码
npm whoami         # 应打印你的 npm 用户名
npm publish
```

- 包名 `dsh-watermark` 在 npm 上**未被占用**（2026-09-22 查过）
- `prepack` 会自动清掉 `__pycache__`：实测 11 个文件 / 48.2 kB
- 发布后 npm 包会通过 `repository` 字段自动与这个仓库关联（他们自动采集，条目里不用加字段）

---

## 4. 可选：截图

在本仓库根放 `screenshots.json`，列 1-8 张图（路径相对该文件）：

```json
["assets/screenshot-1.png", "assets/screenshot-2.png"]
```

放自己仓库的好处：以后换图推自己的仓库即可，不用再提 PR、不用等维护者。

---

## 5. 发布前自检

```powershell
cd D:\dsh-watermark
python -X utf8 test/test_remove_watermark.py      # 批量通用性
python -X utf8 test/test_signature_and_lattice.py # 签名反解
node test/run.mjs                                 # 宿主插件层
python -X utf8 evidence/verify_readme_quotes.py   # README 引用的输出与 evidence 逐字一致
git status --short                                # 应无输出
```

`evidence/` 里施工期的测量脚本（`diag.py` / `probe*.py` 等）已在 `.gitignore` 里排除；
README 引用的三段原始输出（`evidence/final-*.txt`）用 `!evidence/final-*.txt` 例外**已提交**。
