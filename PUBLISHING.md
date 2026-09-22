# 发布清单（GitHub + npm + 插件市场）

本文件是**待执行**的发布步骤，不是已经完成的记录。仓库里能自动完成的都已完成
（git 仓库、提交、registry 条目 JSON、README 徽章文本）；剩下三步需要账号凭据，
本会话没有，所以没有代跑，也没有假装跑过。

## 0. 需要先替换的占位符

`package.json` 里有 4 处 `CHANGE_ME`（`repository` / `bugs` / `homepage` / `author`），
替换成你的 GitHub 账号后再发布：

```powershell
cd D:\dsh-watermark
(Get-Content package.json -Raw) -replace 'CHANGE_ME', '<你的 GitHub 用户名>' |
  Set-Content package.json -NoNewline
```

`git config user.name` / `user.email` 在本机是空的，提交前需要设置（仓库级即可）：

```powershell
git config user.name  "<你的名字>"
git config user.email "<你的邮箱>"
git commit --amend --reset-author --no-edit     # 重新署名已有提交
```

## 1. 上 GitHub

```powershell
cd D:\dsh-watermark
# 先在 github.com 上手动建一个空仓库 dsh-watermark（不要勾 README/gitignore）
git remote add origin https://github.com/<你的 GitHub 用户名>/dsh-watermark.git
git branch -M main
git push -u origin main
```

仓库建好后，给仓库加上 `dsh-plugin` topic（市场靠这个 topic 做 GitHub 搜索）：

> 仓库页 → About 齿轮 → Topics → 填 `dsh-plugin`、`deepseek-harness`、`watermark`

## 2. 上 npm（可选；纯 GitHub 安装也能用）

```powershell
npm login                       # 需要 npm 账号
npm publish --dry-run           # 先看会打包哪些文件
npm publish
```

如果不想上 npm，用户也可以直接装 GitHub 源：

```powershell
dsh plugin --profile web add github:<你的 GitHub 用户名>/dsh-watermark
```

## 3. 进插件市场（awesome-dsh-plugin registry）

市场（`dshmarket` 的安装源）是 curated registry
[awesome-dsh-plugin](https://github.com/awesome-dsh-plugin)，收录方式是**提 PR 加一条条目**。
要加的条目已经写好在 `publish/registry-entry.json`，与其它 176 条同一个 schema：

```json
{
  "name": "dsh-watermark",
  "owner": "<你的 GitHub 用户名>",
  "url": "https://github.com/<你的 GitHub 用户名>/dsh-watermark",
  "category": "tools",
  "description": { "en": "…", "zh": "…" },
  "install": "dsh plugin --profile web add github:<你的 GitHub 用户名>/dsh-watermark",
  "added": "<YYYY-MM-DD>"
}
```

步骤：

1. fork `awesome-dsh-plugin`；
2. 把上面这条（改好 owner/url/install/added）插进 registry 的 `plugins` 数组；
3. 提 PR，说明这是 DSH host 插件、提供 `remove_watermark` 工具、MIT；
4. PR 描述里贴上 `evidence/final-python-suite.txt`、`evidence/final-signature-suite.txt`、
   `evidence/final-node-suite.txt` 三段原始输出作为可复现证据。

`category` 取值参考 registry 现有条目（`ui` / `tools` / `integration` …）；
`description` 必须 en + zh 双语，这是该 registry 的既有约定。

## 4. 发布前自检

```powershell
cd D:\dsh-watermark
npm run test:all                 # 三个套件：10 pass/1 xfail、7 pass、16 pass
node -e "JSON.parse(require('fs').readFileSync('package.json'))"   # JSON 合法性
git status --short               # 确认没有把 evidence/ 里的临时脚本提交进去
```

`evidence/` 目录里**施工期的测量脚本**（`diag.py` / `probe*.py` / `tune.py` / `capture.py`）
是过程材料，已在 `.gitignore` 里排除；README 引用的三段原始输出
（`evidence/final-*.txt`）用 `!evidence/final-*.txt` 例外**已经提交**，无需额外操作。
它们可以用 `python -X utf8 evidence/capture.py` 重新生成（三个套件重跑一遍并写成 UTF-8）。
