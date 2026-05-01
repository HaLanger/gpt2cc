# gpt2cc

`gpt2cc` 是一个给 Claude Code 使用的本地 Anthropic Messages API 兼容代理。它对 Claude Code 暴露 `/v1/messages`，再把请求转换到 OpenAI/Codex 中转站常见的 `/v1/chat/completions` 格式，并把响应转换回 Claude Code 需要的 Anthropic Messages/SSE 格式。

## 功能

- 支持 Claude Code 常用的 `POST /v1/messages`
- 支持非流式响应和 Anthropic SSE 流式响应
- 支持 Claude Code 工具调用：Anthropic `tool_use`/`tool_result` ↔ OpenAI `tool_calls`/`tool`
- 支持 `system`、`messages`、`max_tokens`、`temperature`、`top_p`、`stop_sequences`
- 支持模型映射：把 Claude Code 请求的 Claude 模型名映射到你的 Codex 中转站模型名
- 支持本地代理鉴权，避免同网段误用
- 支持 `/v1/messages/count_tokens` 近似估算，满足 Claude Code 预算检查
- 支持 `/v1/models`、`/healthz`、`/debug/config`
- 零第三方依赖，Python 3.10+ 可直接运行

## 工作原理

Claude Code 使用 Anthropic Messages API，请求体大致是：

```json
{
  "model": "claude-sonnet-4-5-20250929",
  "system": "You are Claude Code...",
  "messages": [{"role": "user", "content": "hello"}],
  "tools": [
    {
      "name": "Bash",
      "description": "Run shell command",
      "input_schema": {"type": "object", "properties": {}}
    }
  ],
  "stream": true
}
```

本代理会转换成 OpenAI/Codex 兼容的 Chat Completions：

```json
{
  "model": "gpt-4.1",
  "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "hello"}],
  "tools": [{"type": "function", "function": {"name": "Bash", "parameters": {}}}],
  "stream": true
}
```

上游返回的 `content` 会变成 Anthropic `text` 内容块；上游返回的 `tool_calls` 会变成 Claude Code 能执行的 `tool_use` 内容块。

## 快速开始

1. 复制配置文件：

```powershell
Copy-Item .env.example .env
```

2. 修改 `.env`：

```dotenv
GPT2CC_UPSTREAM_BASE_URL=https://你的中转站域名/v1
GPT2CC_UPSTREAM_API_KEY=你的中转站 API Key
GPT2CC_UPSTREAM_MODEL=你的中转站模型名
```

如果你的中转站模型名是 `gpt-4.1`，可以保持示例值。如果是其他名字，例如 `gpt-5.1-codex`，把 `GPT2CC_UPSTREAM_MODEL` 改成对应值。

3. 启动代理：

```powershell
python -m gpt2cc
```

默认监听：

```text
http://127.0.0.1:3456
```

4. 健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:3456/healthz
```

## 配置 Claude Code

Claude Code 支持通过环境变量设置 Anthropic API 地址。使用本代理时，把 Anthropic base URL 指到本地代理：

```powershell
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:3456"
$env:ANTHROPIC_AUTH_TOKEN = "local-test-key"
claude
```

如果你使用 `ccswitch`，把 provider/base URL 设置为：

```text
http://127.0.0.1:3456
```

API key 可以随便填一个本地值；如果你设置了 `GPT2CC_PROXY_API_KEY`，则 Claude Code 或 ccswitch 里填写的 key 必须与它一致。

环境变量使用 `GPT2CC_*` 作为新前缀；旧版 `CCPROXY_*` 仍作为兼容别名可用。如果两个前缀同时设置，`GPT2CC_*` 优先。

## 推荐配置

最常用的 `.env`：

```dotenv
GPT2CC_HOST=127.0.0.1
GPT2CC_PORT=3456
GPT2CC_UPSTREAM_BASE_URL=https://your-codex-relay.example.com/v1
GPT2CC_UPSTREAM_API_KEY=sk-your-upstream-key
GPT2CC_UPSTREAM_MODEL=gpt-4.1
GPT2CC_PROXY_API_KEY=local-claude-code-key
```

然后让 Claude Code 使用：

```powershell
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:3456"
$env:ANTHROPIC_AUTH_TOKEN = "local-claude-code-key"
claude
```

## 证书错误排查

如果 Claude Code 输出类似：

```text
API Error: 502 {"type":"error","error":{"type":"api_error","message":"[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate"}}
```

这表示 `Claude Code -> gpt2cc` 已经通了，失败发生在 `gpt2cc -> 中转站` 的 HTTPS 证书校验。

优先检查：

```dotenv
GPT2CC_UPSTREAM_BASE_URL=https://你的中转站域名/v1
```

确认域名没有写错，浏览器访问该域名证书有效。如果你的中转站或公司网络使用自签名证书、内网 CA、HTTPS 代理，需要把 CA 证书导出为 PEM 文件，然后配置：

```dotenv
GPT2CC_UPSTREAM_CA_BUNDLE=C:\path\to\root-or-company-ca.pem
GPT2CC_UPSTREAM_SSL_VERIFY=true
```

只为了临时确认问题是否确实来自证书链，可以短时间关闭上游证书校验：

```dotenv
GPT2CC_UPSTREAM_SSL_VERIFY=false
```

改完 `.env` 后必须重启 `gpt2cc`。关闭校验会让代理无法确认上游服务器身份，不建议长期使用。

另一个容易混淆的点：如果设置了 `GPT2CC_PROXY_API_KEY=local-claude-code-key`，那么 Claude Code 的 `ANTHROPIC_AUTH_TOKEN` 或 ccswitch 里填写的 API key 也必须是 `local-claude-code-key`。本地代理 key 不需要等于中转站 API key。

## 模型映射

如果你希望不同 Claude 模型映射到不同上游模型：

```dotenv
GPT2CC_MODEL_MAP={"claude-sonnet-4-5-20250929":"gpt-4.1","claude-3-5-sonnet-20241022":"gpt-4.1-mini"}
```

优先级：

1. `GPT2CC_MODEL_MAP` 命中时使用映射结果
2. 否则如果设置了 `GPT2CC_UPSTREAM_MODEL`，所有请求都强制使用该模型
3. 否则如果 `GPT2CC_PASS_THROUGH_MODEL=true`，透传 Claude Code 请求里的模型名
4. 最后回退到请求中的模型名

通常建议直接设置 `GPT2CC_UPSTREAM_MODEL`，最稳。

## 上游兼容性

大多数 OpenAI-compatible 中转站都接受：

```text
POST /v1/chat/completions
Authorization: Bearer <key>
```

如果你的中转站路径不同：

```dotenv
GPT2CC_UPSTREAM_BASE_URL=https://your-relay.example.com
GPT2CC_UPSTREAM_CHAT_PATH=/openai/v1/chat/completions
```

如果你的中转站不支持 `stream_options`，保持：

```dotenv
GPT2CC_RETRY_WITHOUT_STREAM_OPTIONS=true
```

代理会在上游返回 `400` 或 `422` 时自动去掉 `stream_options` 重试一次。

如果你的上游模型不接受 `max_tokens`，例如需要 `max_completion_tokens`：

```dotenv
GPT2CC_MAX_TOKENS_FIELD=max_completion_tokens
```

如果上游模型不接受采样参数：

```dotenv
GPT2CC_OMIT_TEMPERATURE=true
GPT2CC_OMIT_TOP_P=true
```

## 图像生成模型

如果你的中转站支持 `gpt-image-2` 这类图像模型，不要再让它走 `/v1/chat/completions`。这类模型需要 Images API，代理会在上游模型匹配 `GPT2CC_IMAGE_MODELS` 时自动改走：

```text
POST /v1/images/generations  # 无参考图
POST /v1/images/edits        # Claude Code 消息里带参考图
```

最简单配置：

```dotenv
GPT2CC_UPSTREAM_MODEL=gpt-image-2
GPT2CC_UPSTREAM_IMAGES_GENERATIONS_PATH=/images/generations
GPT2CC_UPSTREAM_IMAGES_EDITS_PATH=/images/edits
GPT2CC_IMAGE_OUTPUT_DIR=generated-images
GPT2CC_IMAGE_SIZE=auto
GPT2CC_IMAGE_QUALITY=auto
GPT2CC_IMAGE_BACKGROUND=auto
GPT2CC_IMAGE_OUTPUT_FORMAT=png
GPT2CC_IMAGE_N=1
GPT2CC_IMAGE_MAX_REFERENCE_IMAGES=16
```

然后在 Claude Code 里直接说：

```text
生成一张 16:9 的雨夜霓虹街道概念图，电影感，高细节
```

代理会提取最后一条用户文本作为图片 prompt，调用上游 Images API，把返回的 base64 图片保存到 `GPT2CC_IMAGE_OUTPUT_DIR`，并把本地文件路径返回给 Claude Code。上游响应体偶发提前结束时，代理会按 `GPT2CC_MAX_RETRIES` 重试。

如果你在 Claude Code 里 `@` 了参考图片，Claude Code 通常会把它作为 Anthropic `image` 内容块发送给代理。代理会从最后一条用户消息里提取这些 base64 图片，并以 multipart `image[]` 字段上传到 `/images/edits`。日志会显示：

```text
model route: requested=claude-sonnet-... upstream=gpt-image-2 endpoint=images/edits stream=True tools_ignored=... references=True
image edit complete: model=gpt-image-2 references=2 images=1
```

参考图相关配置：

```dotenv
GPT2CC_UPSTREAM_IMAGES_EDITS_PATH=/images/edits
GPT2CC_IMAGE_MAX_REFERENCE_IMAGES=16
# 可选；上游支持时再设置，例如 high/low。留空表示不发送。
GPT2CC_IMAGE_INPUT_FIDELITY=
```

控制台日志会显示：

```text
model route: requested=claude-sonnet-... upstream=gpt-image-2 endpoint=images/generations stream=True tools_ignored=...
image generation complete: model=gpt-image-2 images=1
```

重要限制：

- `gpt-image-2` 是图像生成模型，不是 Claude Code 的代码/工具调用模型；它不会执行 Bash、Read、Edit 等 Claude Code 工具
- Claude Code 终端通常不会直接渲染图片，只会看到生成后的本地文件路径
- 带参考图时代理只处理 Claude Code 发送的 base64 图片块；如果某个中转站把上传图片改成 URL 或 file_id，可能需要额外适配
- 如果你想同时写代码和生图，建议平时使用聊天模型，例如 `gpt-5.4`，只在专门生图时切换到 `gpt-image-2`

## 工具调用转换

Claude Code 的工具定义：

```json
{
  "name": "Bash",
  "description": "Run shell command",
  "input_schema": {
    "type": "object",
    "properties": {
      "command": {"type": "string"}
    },
    "required": ["command"]
  }
}
```

会被转换为 OpenAI function tool：

```json
{
  "type": "function",
  "function": {
    "name": "Bash",
    "description": "Run shell command",
    "parameters": {
      "type": "object",
      "properties": {
        "command": {"type": "string"}
      },
      "required": ["command"]
    }
  }
}
```

当上游返回：

```json
{
  "tool_calls": [
    {
      "id": "call_1",
      "type": "function",
      "function": {
        "name": "Bash",
        "arguments": "{\"command\":\"pwd\"}"
      }
    }
  ]
}
```

代理会返回 Claude Code 可执行的：

```json
{
  "type": "tool_use",
  "id": "call_1",
  "name": "Bash",
  "input": {"command": "pwd"}
}
```

## 流式响应

上游 OpenAI stream chunk 会被转换为 Anthropic SSE：

```text
event: message_start
data: {...}

event: content_block_start
data: {...}

event: content_block_delta
data: {"delta":{"type":"text_delta","text":"..."}}

event: message_delta
data: {"delta":{"stop_reason":"end_turn"}}

event: message_stop
data: {"type":"message_stop"}
```

工具调用流式参数会转换成 `input_json_delta`，Claude Code 可以边接收边组装工具输入。

## 安全建议

- 默认只监听 `127.0.0.1`，不要直接暴露到公网
- 如果需要局域网访问，务必设置 `GPT2CC_PROXY_API_KEY`
- 不要在生产环境开启 `GPT2CC_DEBUG_PAYLOADS=true`，它会记录完整 prompt 和工具参数
- 上游 API Key 只写在 `.env` 或系统环境变量里，不要提交到仓库

## 调试

查看当前配置，密钥会被遮蔽：

```powershell
Invoke-RestMethod http://127.0.0.1:3456/debug/config
```

每次 `/v1/messages` 请求都会在 `gpt2cc` 控制台打印一行模型路由日志：

```text
model route: requested=claude-sonnet-... upstream=gpt-5.4 stream=True tools=14
```

这里的 `requested` 是 Claude Code/ccswitch 发给本地代理的 Anthropic 模型名，`upstream` 才是代理实际发给中转站的模型名。

如果是流式请求，控制台还会打印：

```text
stream diagnostics: first text delta received from upstream
stream diagnostics: complete text_deltas=18 tool_deltas=0 finish_reason=end_turn
```

能看到第一行，说明中转站到 `gpt2cc` 的流式增量已经到了；如果 Claude Code 终端仍然等完整答案才显示，多半是 Claude Code 当前 UI/调用方式在缓冲展示。若 `model route` 里显示 `stream=False`，说明 Claude Code 这次请求本身不是流式请求。

测试非流式消息：

```powershell
$body = @{
  model = "claude-sonnet-4-5-20250929"
  max_tokens = 200
  stream = $false
  messages = @(@{ role = "user"; content = "用一句话介绍你自己" })
} | ConvertTo-Json -Depth 10

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:3456/v1/messages `
  -ContentType "application/json" `
  -Body $body
```

测试 token 估算：

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:3456/v1/messages/count_tokens `
  -ContentType "application/json" `
  -Body $body
```

## 已知边界

- `/v1/messages/count_tokens` 是本地近似估算，不等于上游模型真实 tokenizer
- Anthropic prompt caching 字段会被忽略
- Anthropic extended thinking 不会被上游 OpenAI-compatible Chat Completions 原生复现
- 图片会以 OpenAI `image_url` data URL 形式转发；你的上游模型必须支持视觉输入才会生效
- 上游模型需要具备可靠 function calling 能力，否则 Claude Code 的工具使用体验会明显下降

## 参考

- Claude Code 环境变量文档：<https://docs.anthropic.com/en/docs/claude-code/settings#environment-variables>
- Anthropic Messages API：<https://docs.anthropic.com/en/api/messages>
- Anthropic streaming Messages：<https://docs.anthropic.com/en/api/messages-streaming>
- OpenAI Chat Completions API：<https://platform.openai.com/docs/api-reference/chat>
