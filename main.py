"""astrbot_plugin_video_omni - 让 Bot 真正看到的视频，而不只是收到一条链接。

设计上跟「抽帧 + 转写」那一派不一样：视频整段交给多模态模型，
画面和声音一次读完，中间不插抽帧、不切音频、不等转写。

也刻意不阻塞对话：收到链接先应一声，后台跑完再把结果单独发出来。
不会再出现解析卡住、把用户消息吞掉的情况。

许可证：MIT
"""
from __future__ import annotations

import asyncio
import os
import random
import time

from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star

from .core.omni import DEFAULT_PROMPT, OmniError, VideoOmni, extract_bv

PLUGIN_NAME = "astrbot_plugin_video_omni"
PLUGIN_VERSION = "0.1.2"

NOTICES = [
    "收到，我先看一遍。",
    "这个得看完才好说话，稍等。",
    "在看了，看完就跟你讲。",
]


class Main(Star):
    def __init__(self, context: Context, config: AstrBotConfig = None):
        super().__init__(context)
        self.config = config if config is not None else {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._sem: asyncio.Semaphore | None = None

    # ---------------- 小工具 ----------------

    def _on(self) -> bool:
        v = self.config.get("enabled", True)
        return bool(v) if not isinstance(v, str) else v.lower() in ("1", "true", "yes", "on")

    def _cfg(self) -> dict:
        try:
            return dict(self.config)
        except Exception:
            return {}

    def _omni(self) -> VideoOmni:
        return VideoOmni(self._cfg(), log=lambda m: self.logger.debug(f"[video_omni] {m}"))

    def _semaphore(self) -> asyncio.Semaphore:
        """全局并发闸门：几个群同时发链接也不会一起压榨 CPU 和钱包。"""
        if self._sem is None:
            try:
                n = int(self.config.get("max_concurrent", 2))
            except (TypeError, ValueError):
                n = 2
            self._sem = asyncio.Semaphore(max(1, n))
        return self._sem

    @staticmethod
    def _split(text: str, size: int = 1200) -> list[str]:
        """按段落切，别把句子劈断。"""
        parts, buf = [], ""
        for line in text.split("\n"):
            if len(buf) + len(line) + 1 > size:
                if buf:
                    parts.append(buf)
                buf = line
            else:
                buf = f"{buf}\n{line}" if buf else line
        if buf:
            parts.append(buf)
        return parts

    async def _send(self, umo: str, text: str) -> None:
        for chunk in self._split(text):
            await self.context.send_message(umo, MessageChain().message(chunk))

    def _question(self, raw: str) -> str:
        """用户除了链接还说了别的话，就把它当成本次关注的额外问题。"""
        extra = raw or ""
        for token in ("https://", "http://"):
            while token in extra:
                i = extra.index(token)
                j = i
                while j < len(extra) and not extra[j].isspace():
                    j += 1
                extra = extra[:i] + extra[j:]
        extra = extra.strip(" \n\t，,。.！!？?")
        if len(extra) < 3:
            return ""
        return f"{DEFAULT_PROMPT}\n\n另外，用户特别想知道：{extra}"

    # ---------------- 钩子 ----------------

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        if not self._on():
            return
        raw = getattr(event, "message_str", "") or ""
        bv = extract_bv(raw)
        if not bv:
            return

        key = event.unified_msg_origin
        running = self._tasks.get(key)
        if running and not running.done():
            await self._send(key, "上一条还在看着呢，等它出来再排这条。")
            event.stop_event()
            return

        event.stop_event()
        if self.config.get("notice", True):
            await self._send(key, random.choice(NOTICES))
        self._tasks[key] = asyncio.create_task(
            self._work(key, bv, self._question(raw)))

    async def _work(self, key: str, bv: str, question: str) -> None:
        t0 = time.time()
        try:
            sem = self._semaphore()
            if sem.locked():
                await self._send(key, "前面还有视频在看，这条排队等着，轮到我就开工。")
            async with sem:
                res = await self._omni().process(bv, question or None)
            await self._send(key, self._format(res))
        except OmniError as exc:
            await self._send(key, f"这条没看成：{exc}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # 兜住所有异常，绝不让消息石沉大海
            self.logger.warning(f"[video_omni] 处理失败：{exc}")
            await self._send(key, f"看的时候出岔子了：{exc}")
        finally:
            self._tasks.pop(key, None)
            self.logger.info(f"[video_omni] {bv} 处理结束，耗时 {time.time()-t0:.0f}s")

    def _format(self, res: dict) -> str:
        head = f"看完了，{res['duration']/60:.1f} 分钟。\n\n"
        body = res["answer"]
        if not self.config.get("show_usage", True):
            return head + body
        tail = (f"\n\n————\n{res['height']}p / {res['out_mb']:.0f}MB｜"
                f"输入 {res['tokens_in']:,}（画面 {res['video_tokens']:,}"
                f"{' / 声音 ' + format(res['audio_tokens'], ',') if res['audio_tokens'] else ''}）"
                f"｜输出 {res['tokens_out']:,}｜约 ¥{res['cost_cny']:.3f}"
                f"｜总耗时 {res['seconds']}s")
        return head + body + tail

    async def terminate(self):
        for task in list(self._tasks.values()):
            task.cancel()
        self._tasks.clear()
        self.logger.info("[video_omni] 插件已停止。")
