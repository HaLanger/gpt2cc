from __future__ import annotations

import argparse
import html
import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from . import __version__
from .config import Config, ConfigStore, load_config
from .image import (
    anthropic_message_from_image_result,
    build_image_edit_request,
    build_image_generation_payload,
    edit_image,
    generate_image,
    is_image_model,
    request_has_reference_images,
    stream_image_result_to_anthropic,
)
from .streaming import stream_openai_to_anthropic
from .tokens import estimate_tokens
from .transform import anthropic_message_from_openai, transform_anthropic_to_openai
from .upstream import UpstreamError, open_stream_with_retry, post_json


LOG = logging.getLogger(__name__)


def make_handler(config: Config) -> type[BaseHTTPRequestHandler]:
    store = ConfigStore(config)

    class Handler(BaseHTTPRequestHandler):
        server_version = f"gpt2cc/{__version__}"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            LOG.info("%s - %s", self.address_string(), fmt % args)

        def do_OPTIONS(self) -> None:
            self.send_response(HTTPStatus.NO_CONTENT)
            self._send_common_headers()
            self.end_headers()

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            try:
                if path in {"/", "/health", "/healthz"}:
                    current = store.snapshot()
                    self._send_json(
                        {
                            "ok": True,
                            "name": "gpt2cc",
                            "version": __version__,
                            "anthropic_compatible": True,
                            "upstream": current.upstream_base_url,
                        }
                    )
                    return
                if path == "/admin":
                    self._require_auth()
                    self._send_html(admin_html(store.state()))
                    return
                if path == "/admin/state":
                    self._require_auth()
                    self._send_json(store.state())
                    return
                if path == "/debug/config":
                    self._require_auth()
                    current = store.snapshot()
                    payload = current.redacted()
                    payload["admin_state"] = store.state()
                    self._send_json(payload)
                    return
                if path == "/v1/models":
                    models = self._known_anthropic_models()
                    self._send_json(
                        {
                            "data": [
                                {
                                    "type": "model",
                                    "id": model_id,
                                    "display_name": model_id,
                                    "created_at": "2025-01-01T00:00:00Z",
                                }
                                for model_id in models
                            ],
                            "has_more": False,
                            "first_id": models[0],
                            "last_id": models[-1],
                        }
                    )
                    return
                self._send_error(HTTPStatus.NOT_FOUND, "not_found_error", f"unknown endpoint: {path}")
            except PermissionError as exc:
                self._send_error(HTTPStatus.UNAUTHORIZED, "authentication_error", str(exc))
            except ValueError as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request_error", str(exc))

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            try:
                if path == "/admin/providers":
                    self._require_auth()
                    self._send_json(store.add_or_update_provider(self._read_json()))
                    return
                if path == "/admin/providers/delete":
                    self._require_auth()
                    payload = self._read_json()
                    self._send_json(store.delete_provider(str(payload.get("id") or "")))
                    return
                if path == "/admin/active":
                    self._require_auth()
                    payload = self._read_json()
                    self._send_json(store.set_active(str(payload.get("provider_id") or ""), str(payload.get("model") or "")))
                    return
                if path == "/v1/messages":
                    self._handle_messages()
                    return
                if path == "/v1/messages/count_tokens":
                    self._handle_count_tokens()
                    return
                self._send_error(HTTPStatus.NOT_FOUND, "not_found_error", f"unknown endpoint: {path}")
            except UpstreamError as exc:
                LOG.warning("upstream error %s: %s", exc.status, exc)
                self._send_error(status_from_upstream(exc.status), "api_error", str(exc), exc.body)
            except PermissionError as exc:
                self._send_error(HTTPStatus.UNAUTHORIZED, "authentication_error", str(exc))
            except json.JSONDecodeError as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request_error", f"invalid JSON: {exc}")
            except ValueError as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request_error", str(exc))
            except Exception as exc:
                LOG.exception("request failed")
                self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "api_error", str(exc))

        def _handle_messages(self) -> None:
            self._require_auth()
            request_config = store.snapshot()
            request = self._read_json(request_config)
            if request_config.debug_payloads:
                LOG.debug("anthropic request: %s", json.dumps(request, ensure_ascii=False))

            upstream_payload, ctx = transform_anthropic_to_openai(request, request_config)
            if is_image_model(ctx.upstream_model, request_config):
                has_references = request_has_reference_images(request, request_config)
                endpoint = "images/edits" if has_references else "images/generations"
                LOG.info(
                    "model route: requested=%s upstream=%s endpoint=%s stream=%s tools_ignored=%s references=%s provider=%s",
                    ctx.requested_model or "<empty>",
                    ctx.upstream_model,
                    endpoint,
                    upstream_payload.get("stream"),
                    len(upstream_payload.get("tools") or []),
                    has_references,
                    request_config.active_provider_label(),
                )

                if has_references:
                    edit_request = build_image_edit_request(request, request_config, ctx)
                    if request_config.debug_payloads:
                        safe_payload = dict(edit_request.fields)
                        safe_payload["prompt"] = f"<{len(edit_request.prompt)} chars>"
                        LOG.debug(
                            "upstream image edit request: %s files=%s",
                            json.dumps(safe_payload, ensure_ascii=False),
                            len(edit_request.files),
                        )

                    if upstream_payload.get("stream"):
                        self._send_stream_headers()
                        stream_image_result_to_anthropic(
                            lambda: edit_image(request_config, edit_request, ctx),
                            ctx,
                            self._write_stream,
                            f"Calling image edit model {ctx.upstream_model} with {edit_request.reference_count} reference image(s)...\n\n",
                        )
                        return

                    result = edit_image(request_config, edit_request, ctx)
                    self._send_json(anthropic_message_from_image_result(result, ctx))
                    return

                image_payload = build_image_generation_payload(request, request_config, ctx)
                if request_config.debug_payloads:
                    safe_payload = dict(image_payload)
                    safe_payload["prompt"] = f"<{len(str(image_payload.get('prompt') or ''))} chars>"
                    LOG.debug("upstream image generation request: %s", json.dumps(safe_payload, ensure_ascii=False))

                if upstream_payload.get("stream"):
                    self._send_stream_headers()
                    stream_image_result_to_anthropic(
                        lambda: generate_image(request_config, image_payload, ctx),
                        ctx,
                        self._write_stream,
                        f"Calling image generation model {ctx.upstream_model}...\n\n",
                    )
                    return

                result = generate_image(request_config, image_payload, ctx)
                self._send_json(anthropic_message_from_image_result(result, ctx))
                return

            LOG.info(
                "model route: requested=%s upstream=%s endpoint=chat/completions stream=%s tools=%s provider=%s",
                ctx.requested_model or "<empty>",
                ctx.upstream_model,
                upstream_payload.get("stream"),
                len(upstream_payload.get("tools") or []),
                request_config.active_provider_label(),
            )
            if request_config.debug_payloads:
                safe_payload = dict(upstream_payload)
                LOG.debug("upstream request: %s", json.dumps(safe_payload, ensure_ascii=False))

            if upstream_payload.get("stream"):
                upstream_stream = open_stream_with_retry(request_config, upstream_payload)
                self._send_stream_headers()
                with upstream_stream as response:
                    stream_openai_to_anthropic(response, ctx, self._write_stream)
                return

            upstream_response = post_json(request_config, upstream_payload)
            data = upstream_response.json()
            result = anthropic_message_from_openai(data, ctx)
            self._send_json(result)

        def _handle_count_tokens(self) -> None:
            self._require_auth()
            request = self._read_json(store.snapshot())
            self._send_json({"input_tokens": estimate_tokens(request)})

        def _read_json(self, request_config: Config | None = None) -> dict[str, Any]:
            limit = (request_config or store.snapshot()).max_body_bytes
            content_length = int(self.headers.get("Content-Length") or "0")
            if content_length > limit:
                raise ValueError(f"request body too large: {content_length} bytes")
            raw = self.rfile.read(content_length)
            if not raw:
                return {}
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("request JSON must be an object")
            return value

        def _require_auth(self) -> None:
            proxy_api_key = store.snapshot().proxy_api_key
            if not proxy_api_key:
                return
            x_api_key = self.headers.get("x-api-key") or self.headers.get("anthropic-api-key")
            auth = self.headers.get("authorization") or ""
            bearer = auth[7:] if auth.lower().startswith("bearer ") else ""
            if proxy_api_key not in {x_api_key, bearer}:
                raise PermissionError("invalid proxy API key")

        def _send_stream_headers(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self._send_common_headers()
            self.end_headers()
            self.close_connection = True

        def _write_stream(self, data: bytes) -> None:
            self.wfile.write(data)
            self.wfile.flush()

        def _send_html(self, body: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            raw = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self._send_common_headers()
            self.end_headers()
            self.wfile.write(raw)

        def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self._send_common_headers()
            self.end_headers()
            self.wfile.write(body)

        def _send_error(
            self,
            status: HTTPStatus,
            error_type: str,
            message: str,
            upstream_body: bytes | None = None,
        ) -> None:
            if upstream_body and store.snapshot().debug_payloads:
                LOG.debug("upstream error body: %s", upstream_body.decode("utf-8", errors="replace"))
            payload = {"type": "error", "error": {"type": error_type, "message": message}}
            self._send_json(payload, status)

        def _send_common_headers(self) -> None:
            if store.snapshot().cors:
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "content-type,x-api-key,authorization,anthropic-version")
                self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")

        @staticmethod
        def _known_anthropic_models() -> list[str]:
            return [
                "claude-sonnet-4-5-20250929",
                "claude-opus-4-1-20250805",
                "claude-3-5-sonnet-20241022",
            ]

    Handler.config_store = store  # type: ignore[attr-defined]
    return Handler


def admin_html(state: dict[str, Any]) -> str:
    auth_hint = "true" if state.get("auth_required") else "false"
    return f"""<!doctype html>
<html lang=\"zh-CN\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>gpt2cc relay console</title>
<style>
:root {{ color-scheme: light; --bg:#f6f7fb; --card:#ffffff; --text:#172033; --muted:#657086; --line:#e5e9f2; --brand:#2563eb; --brand2:#0f172a; --bad:#dc2626; --ok:#059669; }}
* {{ box-sizing:border-box; }} body {{ margin:0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:linear-gradient(135deg,#f8fbff,#f3f4f8); color:var(--text); }}
.shell {{ max-width:1180px; margin:0 auto; padding:32px 20px 48px; }}
.hero {{ display:flex; justify-content:space-between; gap:24px; align-items:flex-start; margin-bottom:24px; }}
h1 {{ margin:0; font-size:34px; letter-spacing:-.04em; }} .subtitle {{ color:var(--muted); margin-top:8px; }}
.status {{ min-width:280px; background:var(--brand2); color:white; padding:18px 20px; border-radius:22px; box-shadow:0 20px 60px rgba(15,23,42,.16); }}
.status small {{ color:#b9c4d8; display:block; margin-bottom:6px; }} .status strong {{ display:block; font-size:18px; word-break:break-all; }}
.toolbar {{ display:grid; grid-template-columns:minmax(220px,1fr) auto auto; gap:10px; margin:16px 0; }}
.panel,.drawer {{ background:rgba(255,255,255,.9); border:1px solid var(--line); border-radius:22px; box-shadow:0 12px 38px rgba(30,41,59,.08); }} .panel {{ padding:16px; }}
input,textarea,select {{ width:100%; border:1px solid var(--line); border-radius:13px; padding:10px 12px; font:inherit; background:white; color:var(--text); }} textarea {{ min-height:126px; resize:vertical; }}
button {{ border:0; border-radius:12px; padding:10px 13px; font-weight:750; cursor:pointer; background:#e8eefc; color:#1e3a8a; white-space:nowrap; }} button.primary {{ background:var(--brand); color:white; }} button.danger {{ background:#fee2e2; color:#991b1b; }} button.ghost {{ background:#f8fafc; color:#334155; border:1px solid var(--line); }} button:hover {{ filter:brightness(.97); }}
.banner {{ display:none; margin-bottom:14px; padding:11px 13px; border-radius:14px; font-weight:700; }} .banner.ok {{ display:block; background:#dcfce7; color:#166534; }} .banner.err {{ display:block; background:#fee2e2; color:#991b1b; }} .auth {{ display:none; margin-bottom:14px; }} .auth.show {{ display:block; }}
.list {{ display:grid; gap:8px; }} .row {{ display:grid; grid-template-columns:1.35fr 1.2fr .9fr auto; gap:12px; align-items:center; padding:12px; border:1px solid var(--line); background:white; border-radius:16px; }} .row.active {{ border-color:#93c5fd; box-shadow:0 10px 30px rgba(37,99,235,.10); }}
.name {{ font-weight:800; }} .sub {{ color:var(--muted); font-size:12px; margin-top:3px; word-break:break-all; }} .models {{ display:flex; flex-wrap:wrap; gap:5px; max-height:52px; overflow:hidden; }} .model {{ border:1px solid var(--line); background:#f8fafc; border-radius:999px; padding:4px 8px; font-size:12px; }} button.model {{ color:#334155; font-weight:750; }} button.model.active-model {{ background:#dbeafe; border-color:#93c5fd; color:#1d4ed8; }} .pill {{ border-radius:999px; padding:4px 9px; background:#eef2ff; color:#1d4ed8; font-size:12px; font-weight:800; display:inline-block; margin-left:6px; }} .actions {{ display:flex; gap:7px; justify-content:flex-end; flex-wrap:wrap; }} .empty {{ text-align:center; color:var(--muted); padding:32px; }}
.drawer-backdrop {{ position:fixed; inset:0; background:rgba(15,23,42,.28); opacity:0; pointer-events:none; transition:.18s; }} .drawer-backdrop.show {{ opacity:1; pointer-events:auto; }} .drawer {{ position:fixed; top:18px; right:18px; bottom:18px; width:min(460px,calc(100vw - 36px)); padding:20px; overflow:auto; transform:translateX(calc(100% + 28px)); transition:.2s; }} .drawer.show {{ transform:translateX(0); }} .drawer-head {{ display:flex; justify-content:space-between; align-items:center; gap:12px; }} .drawer h2 {{ margin:0; }} label {{ display:block; color:#334155; font-size:13px; font-weight:750; margin:13px 0 6px; }}
@media (max-width:900px) {{ .hero,.toolbar,.row {{ grid-template-columns:1fr; }} .actions {{ justify-content:flex-start; }} }}
</style>
</head>
<body>
<div class=\"shell\">
  <div class=\"hero\"><div><h1>gpt2cc relay console</h1><div class=\"subtitle\">管理中转站、API key 和模型，新请求会立即使用当前选择。</div></div><div class=\"status\"><small>当前激活</small><strong id=\"activeTitle\">Loading...</strong><small id=\"configPath\"></small></div></div>
  <div id=\"banner\" class=\"banner\"></div>
  <div id=\"authBox\" class=\"panel auth\"><b>代理密钥</b><div class=\"muted\">此服务启用了 GPT2CC_PROXY_API_KEY。密钥只保存在本次浏览器会话。</div><div class=\"toolbar\"><input id=\"proxyKey\" type=\"password\" autocomplete=\"off\" placeholder=\"Proxy API key\"><button class=\"primary\" onclick=\"saveProxyKey()\">保存并连接</button></div></div>
  <section class=\"panel\"><div class=\"toolbar\"><input id=\"searchBox\" placeholder=\"搜索中转站名称、ID、域名或模型...\" oninput=\"render()\"><button class=\"ghost\" onclick=\"clearSearch()\">清除搜索</button><button class=\"primary\" onclick=\"openForm()\">添加中转站</button></div><div id=\"providers\" class=\"list\"></div></section>
</div>
<div id=\"drawerBackdrop\" class=\"drawer-backdrop\" onclick=\"closeForm()\"></div>
<aside id=\"drawer\" class=\"drawer\"><div class=\"drawer-head\"><h2 id=\"formTitle\">添加中转站</h2><button class=\"ghost\" onclick=\"closeForm()\">关闭</button></div><label>ID</label><input id=\"providerId\" placeholder=\"my-relay\"><label>名称</label><input id=\"providerName\" placeholder=\"My Relay\"><label>Base URL</label><input id=\"baseUrl\" placeholder=\"https://relay.example.com/v1\"><label>API key</label><input id=\"apiKey\" type=\"password\" placeholder=\"编辑时留空表示保留原 key\"><label>模型（每行一个）</label><textarea id=\"models\" placeholder=\"gpt-4.1&#10;gpt-image-2\"></textarea><div class=\"actions\"><button class=\"primary\" onclick=\"saveProvider()\">保存中转站</button><button onclick=\"resetForm()\">清空</button></div><p class=\"muted\">关闭面板不会清空未保存内容；再次点“添加中转站”会继续显示。</p></aside>
<script>
const AUTH_REQUIRED = {auth_hint};
let state = null;
function key() {{ return sessionStorage.getItem('gpt2cc_proxy_key') || ''; }}
function headers() {{ const h={{'content-type':'application/json'}}; if(key()) h['x-api-key']=key(); return h; }}
function show(msg, cls='ok') {{ const b=document.getElementById('banner'); b.className='banner '+cls; b.textContent=msg; setTimeout(()=>{{b.className='banner';}},3500); }}
function esc(s) {{ return String(s ?? '').replace(/[&<>\"]/g, c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c])); }}
function saveProxyKey() {{ sessionStorage.setItem('gpt2cc_proxy_key', document.getElementById('proxyKey').value); load(); }}
async function api(path, opts={{}}) {{ const r=await fetch(path, {{...opts, headers:{{...headers(), ...(opts.headers||{{}})}}}}); if(r.status===401) {{ document.getElementById('authBox').classList.add('show'); throw new Error('需要代理密钥'); }} const data=await r.json(); if(!r.ok) throw new Error(data.error?.message || '请求失败'); return data; }}
async function load() {{ try {{ state=await api('/admin/state'); document.getElementById('authBox').classList.toggle('show', AUTH_REQUIRED && !key()); render(); }} catch(e) {{ show(e.message,'err'); }} }}
function providerText(p) {{ return [p.id,p.name,p.upstream_base_url,...(p.models||[])].join(' ').toLowerCase(); }}
function filteredProviders() {{ const q=(searchBox.value||'').trim().toLowerCase(); return !q ? state.providers : state.providers.filter(p=>providerText(p).includes(q)); }}
function clearSearch() {{ searchBox.value=''; render(); }}
function render() {{ const active=state.providers.find(p=>p.id===state.active_provider); activeTitle.textContent=(active?.name||state.active_provider)+' / '+state.active_model; configPath.textContent='配置文件：'+state.config_path; const items=filteredProviders(); providers.innerHTML=items.length ? items.map(p=>rowHtml(p)).join('') : '<div class="empty">没有匹配的中转站</div>'; }}
function rowHtml(p) {{ const active=p.id===state.active_provider; const models=p.models||[]; const visible=models.slice(0,8).map(m=>`<button class="model ${{active&&m===state.active_model?'active-model':''}}" onclick="activateModel('${{esc(p.id)}}',decodeURIComponent('${{encodeURIComponent(m)}}'))">${{esc(m)}}</button>`).join('') || '<span class="muted">未配置模型</span>'; return `<div class="row ${{active?'active':''}}"><div><div class="name">${{esc(p.name)}}${{active?'<span class="pill">Active</span>':''}}</div><div class="sub">${{esc(p.id)}} · ${{p.has_api_key?'key 已保存':'未设置 key'}}</div></div><div class="sub">${{esc(p.upstream_base_url)}}</div><div class="models">${{visible}}${{models.length>8?`<span class="model">+${{models.length-8}}</span>`:''}}</div><div class="actions"><select id="sel-${{esc(p.id)}}">${{models.map(m=>`<option ${{m===state.active_model?'selected':''}}>${{esc(m)}}</option>`).join('')}}</select><button class="primary" onclick="activate('${{esc(p.id)}}')">切换</button><button onclick="editProvider('${{esc(p.id)}}')">编辑</button><button class="danger" onclick="deleteProvider('${{esc(p.id)}}')">删除</button></div></div>`; }}
function openForm() {{ drawer.classList.add('show'); drawerBackdrop.classList.add('show'); }}
function closeForm() {{ drawer.classList.remove('show'); drawerBackdrop.classList.remove('show'); }}
function editProvider(id) {{ const p=state.providers.find(x=>x.id===id); if(!p) return; formTitle.textContent='编辑中转站'; providerId.value=p.id; providerName.value=p.name; baseUrl.value=p.upstream_base_url; apiKey.value=''; models.value=(p.models||[]).join('\\n'); openForm(); }}
function resetForm() {{ formTitle.textContent='添加中转站'; providerId.value=''; providerName.value=''; baseUrl.value=''; apiKey.value=''; models.value=''; }}
async function saveProvider() {{ try {{ await api('/admin/providers', {{method:'POST', body:JSON.stringify({{id:providerId.value,name:providerName.value,upstream_base_url:baseUrl.value,upstream_api_key:apiKey.value,models:models.value.split(/\\n+/).map(x=>x.trim()).filter(Boolean)}})}}); resetForm(); closeForm(); await load(); show('中转站已保存'); }} catch(e) {{ show(e.message,'err'); }} }}
async function activate(id) {{ try {{ const sel=document.getElementById('sel-'+id); await activateModel(id, sel?.value||''); }} catch(e) {{ show(e.message,'err'); }} }}
async function activateModel(id, model) {{ try {{ await api('/admin/active', {{method:'POST', body:JSON.stringify({{provider_id:id, model}})}}); await load(); show('已切换，新的请求会立即使用该配置'); }} catch(e) {{ show(e.message,'err'); }} }}
async function deleteProvider(id) {{ if(!confirm('删除这个中转站？')) return; try {{ await api('/admin/providers/delete', {{method:'POST', body:JSON.stringify({{id}})}}); await load(); show('中转站已删除'); }} catch(e) {{ show(e.message,'err'); }} }}
load();
</script>
</body>
</html>"""


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def status_from_upstream(status: int) -> HTTPStatus:
    if status in {400, 401, 403, 404, 408, 409, 422, 429, 500, 502, 503, 504}:
        return HTTPStatus(status)
    if 400 <= status < 500:
        return HTTPStatus.BAD_REQUEST
    if 500 <= status < 600:
        return HTTPStatus.BAD_GATEWAY
    return HTTPStatus.INTERNAL_SERVER_ERROR


def run(config: Config) -> None:
    server = ReusableThreadingHTTPServer((config.host, config.port), make_handler(config))
    LOG.info("gpt2cc %s listening on http://%s:%s", __version__, config.host, config.port)
    LOG.info("admin console: http://%s:%s/admin", config.host, config.port)
    LOG.info("upstream chat endpoint: %s", config.upstream_chat_url)
    LOG.info("upstream images endpoint: %s", config.upstream_images_generations_url)
    LOG.info("upstream image edits endpoint: %s", config.upstream_images_edits_url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("stopping")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Anthropic Messages API proxy for Claude Code.")
    parser.add_argument("--host", help="listen host; overrides GPT2CC_HOST")
    parser.add_argument("--port", type=int, help="listen port; overrides GPT2CC_PORT")
    args = parser.parse_args()

    config = load_config()
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port

    run(config)


if __name__ == "__main__":
    main()
