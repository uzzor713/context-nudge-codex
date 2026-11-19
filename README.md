# context-nudge-codex

Codex Desktop 的 context 用量感知与会话交接提醒。

它通过 Codex hooks 读取当前 session 的 JSONL transcript，在自动压缩次数过多或压缩后基线过高时，把提醒注入下一条用户消息对应的 Agent 上下文。

## 触发规则

- compact 2 次：建议交接。
- compact 3 次：尽快交接。
- 压缩后基线达到 30% / 50%：进入对应档位。
- 每个档位每个 session 只提醒一次；升档时再次提醒。
- `/` 开头的 prompt 不插入提醒。

用户可见的 `systemMessage` 取决于 Codex 客户端渲染；`additionalContext` 会作为隐藏 developer context 提供给 Agent。

## 安装

```bash
mkdir -p ~/.codex/hooks
cp context-nudge.py ~/.codex/hooks/context-nudge.py
chmod +x ~/.codex/hooks/context-nudge.py
```

把 `hooks.json.example` 中的两个 hook 合并到 `~/.codex/hooks.json`。已有其它 hooks 时保留它们，不要覆盖整个文件。

状态写入 `~/.codex/context-nudge-state/`。每个 session 保存 transcript 增量扫描位置、compact 计数、基线、当前用量和最后提醒档位。

## 为什么使用增量扫描

Codex transcript 是追加写入的 JSONL。旧版本每次 prompt 都从头解析完整文件，大 session 会超过 hook 超时。现在首次通过内存映射快速定位 compact 标记，只解析最后一次 compact 之后的记录；后续只扫描上次保存的字节偏移之后的新内容。没有 compact 的 transcript 才会退回读取末尾 4 MiB。

compact 后 Codex 可能先写一条 `input_tokens=0` 的 bookkeeping 记录。该记录不作为基线，首条正数 token 用量才会被采用。

## 测试

```bash
python3 -m unittest discover -s tests -v
```

## 许可

MIT
