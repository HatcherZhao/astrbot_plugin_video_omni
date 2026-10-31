"""视频理解核心流程：拿到视频文件 → 压缩 → 上传 → 多模态推理。

不依赖 AstrBot，可单独调用，方便测试。
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
import uuid
from typing import Awaitable, Callable, Optional

import aiohttp

BV_RE = re.compile(r"(BV[0-9A-Za-z]{10})")
AV_RE = re.compile(r"/video/(av\d+)", re.I)

DEFAULT_PROMPT = (
    "这是一条视频。看完后给结论，不要写成播放流水账：\n"
    "1. 第一句就说清它到底在讲什么，结论先行；\n"
    "2. 再列 3～6 条最关键的信息，每条一句话，只留观点、数据和结论；\n"
    "3. 如果视频在教方法，把做法整理成能照着做的条目；\n"
    "4. 最后一行给你自己的判断，或者提醒哪里值得注意。\n"
    "禁止按时间顺序复述“第几分钟讲了什么”，禁止罗列画面细节，除非那是关键信息。\n"
    "用中文，不确定的地方直说不确定。"
)

# 元 / 百万 token，仅用于给用户报个大概花费
PRICE = {
    "qwen3.8-omni-flash": (0.8, 2.7),
    "qwen3.8-omni-plus": (1.6, 6.4),
}


class OmniError(Exception):
    pass


def extract_bv(text: str) -> Optional[str]:
    """从消息或链接里抠出 BV 号/av 号。"""
    if not text:
        return None
    m = BV_RE.search(text)
    if m:
        return m.group(1)
    m = AV_RE.search(text)
    return m.group(1) if m else None


async def _run(cmd: list[str], timeout: int = 900) -> str:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise OmniError(f"命令超时：{cmd[0]}")
    if proc.returncode != 0:
        raise OmniError(f"{cmd[0]} 失败：{err.decode('utf-8', 'ignore')[-300:]}")
    return out.decode("utf-8", "ignore")


def _probe_sync(path: str) -> dict:
    import subprocess
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration,size",
         "-show_entries", "stream=codec_type,width,height", "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    v = next((s for s in d.get("streams", []) if s["codec_type"] == "video"), {})
    return {
        "duration": float(d["format"]["duration"]),
        "size_mb": int(d["format"]["size"]) / 1048576,
        "width": v.get("width"),
        "height": v.get("height"),
        "has_audio": any(s["codec_type"] == "audio" for s in d.get("streams", [])),
    }


def probe(path: str) -> dict:
    try:
        return _probe_sync(path)
    except Exception as exc:
        raise OmniError(f"读不了这个视频文件（可能没写完）：{exc}") from exc


def write_cookie_file(cookie: str, path: str) -> Optional[str]:
    """把「SESSDATA=xx; bili_jct=yy」变成 yt-dlp 认识的 cookies.txt。"""
    if not cookie or not cookie.strip():
        return None
    expire = int(time.time()) + 86400 * 180
    lines = ["# Netscape HTTP Cookie File", ""]
    for part in cookie.replace("\n", ";").split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        lines.append(f".bilibili.com\tTRUE\t/\tFALSE\t{expire}\t{k.strip()}\t{v.strip()}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


class VideoOmni:
    def __init__(self, cfg: dict, log: Optional[Callable[[str], None]] = None):
        self.cfg = cfg
        self.log = log or (lambda m: None)

    # ---------- 配置快捷读取 ----------
    def g(self, key, default=None):
        v = self.cfg.get(key, default)
        return default if v is None or v == "" else v

    @property
    def work_dir(self) -> str:
        d = self.g("work_dir", "/tmp/astrbot_video_omni")
        os.makedirs(d, exist_ok=True)
        return d

    # ---------- 0. 清理 ----------
    def cleanup(self) -> list[str]:
        """按存活时长和条数上限清理临时文件。

        每次处理前跑一遍：超过 ``keep_cache_hours`` 小时的删掉，
        再按修改时间只留最近 ``keep_cache_items`` 个文件。
        正在用的文件 mtime 都是刚刚，不会被误删。
        """
        ttl_h = float(self.g("keep_cache_hours", 6))
        max_items = int(self.g("keep_cache_items", 12))
        now = time.time()
        removed: list[str] = []
        try:
            entries = sorted(
                ((os.path.getmtime(os.path.join(self.work_dir, f)),
                  os.path.join(self.work_dir, f))
                 for f in os.listdir(self.work_dir)
                 if os.path.isfile(os.path.join(self.work_dir, f))),
                reverse=True,
            )
        except FileNotFoundError:
            return removed

        for idx, (mtime, path) in enumerate(entries):
            name = os.path.basename(path)
            if name == "cookies.txt":
                continue
            if (now - mtime) > ttl_h * 3600 or idx >= max_items:
                try:
                    os.remove(path)
                    removed.append(name)
                except OSError:
                    pass
        if removed:
            self.log(f"清理临时文件 {len(removed)} 个")
        return removed

    # ---------- 1. 拿文件 ----------
    def _cache_hit(self, bv: str) -> Optional[str]:
        """若 parser_x 已经下过同一个视频，直接白捡。"""
        dirs = [self.g("parser_cache_dir", ""), "/AstrBot/data/plugin_data/astrbot_plugin_parser_x/cache_dir"]
        for d in dirs:
            if not d or not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if bv in f and f.lower().endswith((".mp4", ".m4v", ".flv")):
                    p = os.path.join(d, f)
                    if os.path.getsize(p) > 1024 * 1024:
                        return p
        return None

    async def fetch(self, bv: str) -> str:
        os.makedirs(self.work_dir, exist_ok=True)
        hit = self._cache_hit(bv)
        if hit:
            self.log(f"复用已有缓存：{os.path.basename(hit)}")
            return hit
        dst = os.path.join(self.work_dir, f"{bv}.mp4")
        if os.path.exists(dst) and os.path.getsize(dst) > 1024 * 1024:
            self.log("复用上次下载的文件")
            return dst
        if not shutil.which("yt-dlp"):
            raise OmniError("这台机器没装 yt-dlp，装一下：pip install yt-dlp")

        url = f"https://www.bilibili.com/video/{bv}"
        cmd = ["yt-dlp", "--no-warnings", "--no-playlist", "-f",
               "bv*[height<=1080]+ba/b[height<=1080]", "--merge-output-format", "mp4",
               "-o", dst]
        cookie_file = None
        cookie = self.g("bili_cookie", "")
        if cookie:
            cookie_file = write_cookie_file(cookie, os.path.join(self.work_dir, "cookies.txt"))
            if cookie_file:
                cmd[1:1] = ["--cookies", cookie_file]
        cmd.append(url)
        self.log("开始下载…")
        await _run(cmd, timeout=1200)
        if not os.path.exists(dst):
            raise OmniError("下载完了却没找到文件")
        return dst

    # ---------- 2. 压缩 ----------
    async def compress(self, src: str, height: int) -> str:
        stem = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join(self.work_dir, f"{stem}_{height}p.mp4")
        if os.path.exists(dst) and os.path.getsize(dst) > 1024 * 1024:
            return dst
        self.log(f"压缩到 {height}p…")
        # 先写 .part 再改名：中途被打断也不会留下一份看着像成品的半截文件
        tmp = dst + ".part"
        await _run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                    "-vf", f"scale=-2:{height}", "-c:v", "libx264", "-crf", "30",
                    "-preset", "veryfast", "-c:a", "aac", "-b:a", "64k",
                    "-f", "mp4", tmp])
        os.replace(tmp, dst)
        return dst

    def pick_height(self, duration: float) -> int:
        q = str(self.g("quality", "auto")).lower()
        if q in ("360", "480", "720"):
            return int(q)
        return 480 if duration <= 900 else 360

    # ---------- 3. 上传 ----------
    async def upload(self, path: str) -> str:
        """流式上传到 DashScope 的临时 OSS。

        body 用异步生成器边读边发，52MB 的文件只占几 MB 内存。
        OSS 不认 chunked，所以按文件大小显式算好 Content-Length；
        顺序也照老规矩：policy 这些字段在前，文件 part 在最后。
        """
        key = self.g("dashscope_api_key", "")
        if not key:
            raise OmniError("还没填百炼 API Key")
        base = self.g("base_url", "https://dashscope.aliyuncs.com").rstrip("/")
        model = self.g("model", "qwen3.8-omni-flash")
        fn = os.path.basename(path)
        timeout = aiohttp.ClientTimeout(total=1800)

        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.get(
                f"{base}/api/v1/uploads",
                params={"action": "getPolicy", "model": model},
                headers={"Authorization": f"Bearer {key}"},
            ) as r:
                d = await r.json(content_type=None)
                if r.status != 200 or "data" not in d:
                    raise OmniError(f"拿上传凭证失败：{r.status} {str(d)[:200]}")
                pol = d["data"]

            obj = f"{pol['upload_dir']}/{fn}"
            b = uuid.uuid4().hex
            head = ""
            for name, val in (
                ("OSSAccessKeyId", pol["oss_access_key_id"]),
                ("Signature", pol["signature"]),
                ("policy", pol["policy"]),
                ("key", obj),
                ("x-oss-object-acl", pol["x_oss_object_acl"]),
                ("x-oss-forbid-overwrite", str(pol["x_oss_forbid_overwrite"]).lower()),
                ("success_action_status", "200"),
            ):
                head += (f'--{b}\r\nContent-Disposition: form-data; name="{name}"'
                         f'\r\n\r\n{val}\r\n')
            head += (f'--{b}\r\nContent-Disposition: form-data; name="file"; '
                     f'filename="{fn}"\r\nContent-Type: video/mp4\r\n\r\n')
            head_b = head.encode()
            tail_b = f"\r\n--{b}--\r\n".encode()
            total = len(head_b) + os.path.getsize(path) + len(tail_b)

            async def body():
                yield head_b
                with open(path, "rb") as fh:
                    while True:
                        chunk = await asyncio.to_thread(fh.read, 262144)
                        if not chunk:
                            break
                        yield chunk
                yield tail_b

            headers = {
                "Content-Type": f"multipart/form-data; boundary={b}",
                "Content-Length": str(total),
            }
            async with s.post(pol["upload_host"], data=body(), headers=headers) as r2:
                if r2.status not in (200, 204):
                    detail = (await r2.text())[:300]
                    raise OmniError(f"上传失败：{r2.status} {detail}")

        return f"oss://{obj}"

    # ---------- 4. 推理 ----------
    async def ask(self, oss_url: str, question: str) -> dict:
        key = self.g("dashscope_api_key", "")
        base = self.g("base_url", "https://dashscope.aliyuncs.com").rstrip("/")
        model = self.g("model", "qwen3.8-omni-flash")
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": oss_url}},
                {"type": "text", "text": question},
            ]}],
            "max_tokens": int(self.g("max_tokens", 3000)),
            "enable_thinking": False,
        }
        headers = {"Authorization": f"Bearer {key}",
                   "Content-Type": "application/json",
                   "X-DashScope-OssResourceResolve": "enable"}
        timeout = aiohttp.ClientTimeout(total=2400)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.post(f"{base}/compatible-mode/v1/chat/completions",
                              data=json.dumps(payload), headers=headers) as r:
                d = await r.json(content_type=None)
                if r.status != 200 or "error" in d:
                    raise OmniError(f"模型调用失败：{d.get('error') or d}")
                return d

    # ---------- 串起来 ----------
    async def process(self, bv: str, question: Optional[str] = None,
                      on_progress: Optional[Callable[[str], Awaitable[None]]] = None) -> dict:
        async def step(msg):
            self.log(msg)
            if on_progress:
                await on_progress(msg)

        t0 = time.time()
        question = question or self.g("prompt", DEFAULT_PROMPT)
        model = self.g("model", "qwen3.8-omni-flash")
        max_minutes = float(self.g("max_minutes", 60))

        await step("拿到链接，开始取视频")
        await asyncio.to_thread(self.cleanup)
        src = await self.fetch(bv)
        info = await asyncio.to_thread(probe, src)
        if info["duration"] > max_minutes * 60:
            raise OmniError(
                f"这条视频 {info['duration']/60:.0f} 分钟，超过设定的 {max_minutes:.0f} 分钟上限")

        height = self.pick_height(info["duration"])
        small = await self.compress(src, height)
        s = await asyncio.to_thread(probe, small)
        await step(f"{info['duration']/60:.1f} 分钟，压到 {height}p / {s['size_mb']:.0f}MB")

        oss = await self.upload(small)
        await step("上传完成，开始逐帧听懂")

        d = await self.ask(oss, question)
        answer = d["choices"][0]["message"]["content"]
        if d["choices"][0].get("finish_reason") == "length":
            answer += "\n\n（写到这里撞上长度上限了，想看更细的可以点某一段问我）"
        u = d.get("usage", {})
        pd = u.get("prompt_tokens_details", {}) or {}
        pin, pout = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        pi, po = PRICE.get(model, (0.8, 2.7))
        cost = pin / 1e6 * pi + pout / 1e6 * po
        return {
            "bv": bv,
            "answer": d["choices"][0]["message"]["content"],
            "duration": info["duration"],
            "height": height,
            "out_mb": round(s["size_mb"], 1),
            "tokens_in": pin,
            "tokens_out": pout,
            "video_tokens": pd.get("video_tokens", 0),
            "audio_tokens": pd.get("audio_tokens", 0),
            "cost_cny": round(cost, 4),
            "seconds": round(time.time() - t0),
            "request_id": d.get("id", ""),
        }
