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

from .core.omni import OmniError, VideoOmni, extract_bv

PLUGIN_NAME = "astrbot_plugin_video_omni"
PLUGIN_VERSION = "0.1.3"

NOTICES = [
    "收到，我先看一遍。",
    "这个得看完才好说话，稍等。",
    "在看了，看完就跟你讲。",
]

# 收到链接后等本人把话说完再开工，两个判据配合：
# 静默 COLLECT_IDLE 秒就当他讲完了，一直絮絮叨叨最多等到 COLLECT_WINDOW。
COLLECT_IDLE = 8.0
COLLECT_WINDOW = 30.0

# 有明确要求时：先概括，再按用户要的形式给结果，不套默认模板。
USER_PROMPT_TEMPLATE = (
    "这是一条视频，用户对这次总结有明确要求。请按这个顺序输出：\n"
    "1. 先用两三句话说清视频主要讲了什么，结论先行，不要按时间复述；\n"
    "2. 然后严格按用户的要求输出：\n"
    "{req}\n\n"
    "不要写成播放流水账，不要罗列画面细节，除非那是关键信息。"
    "以用户的要求为准，不要套固定模板。不确定的地方直说不确定。"
)


class Main(Star):
    def __init__(self, context: Context, config: AstrBotConfig = None):
        super().__init__(context)
        self.config = config if config is not None else {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._pending: dict[str, dict] = {}
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

    @staticmethod
    def _strip_urls(text: str) -> str:
        extra = text or ""
        for token in ("https://", "http://"):
            while token in extra:
                i = extra.index(token)
                j = i
                while j < len(extra) and not extra[j].isspace():
                    j += 1
                extra = extra[:i] + extra[j:]
        return extra.strip(" \n\t，,。.！!？?")

    def _compose(self, texts: list[str]) -> str:
        """把链接后面用户说的话收成一条明确要求；一句没说就用默认总结。"""
        req = "\n".join(t for t in texts if t).strip()
        if len(req) < 2:
            return ""
        return USER_PROMPT_TEMPLATE.format(req=req)

    # ---------------- 钩子 ----------------

    @staticmethod
    def _slot(event: AstrMessageEvent) -> tuple[str, str]:
        """会话 + 发言者各占一个坑，群里别人说话不会串进你的任务里。"""
        return (event.unified_msg_origin, str(event.get_sender_id()))

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        if not self._on():
            return
        raw = getattr(event, "message_str", "") or ""
        slot = self._slot(event)
        umo = slot[0]
        bv = extract_bv(raw)

        # 只有发链接的那个人在收集期补的话才并进任务；别人发言原样交给主模型。
        pending = self._pending.get(slot)
        if pending is not None:
            if bv:
                await self._send(umo, "一次先看一条，等这条出来再发另一条。")
            else:
                extra = self._strip_urls(raw)
                if extra:
                    pending["texts"].append(extra)
                    pending["last"] = time.monotonic()
            event.stop_event()
            return

        if not bv:
            return

        running = self._tasks.get(slot)
        if running and not running.done():
            await self._send(umo, "上一条还在看着呢，等它出来再排这条。")
            event.stop_event()
            return

        event.stop_event()
        if self.config.get("notice", True):
            await self._send(umo, random.choice(NOTICES))
        now = time.monotonic()
        self._pending[slot] = {
            "bv": bv,
            "texts": [self._strip_urls(raw)],
            "start": now,
            "last": now,
        }
        asyncio.create_task(self._collect(slot))

    async def _collect(self, slot: tuple[str, str]) -> None:
        """说完就开工：静默 COLLECT_IDLE 秒算说完，最长等到 COLLECT_WINDOW。"""
        while True:
            await asyncio.sleep(0.3)
            p = self._pending.get(slot)
            if not p:
                return
            now = time.monotonic()
            if now - p["last"] >= COLLECT_IDLE or now - p["start"] >= COLLECT_WINDOW:
                break
        p = self._pending.pop(slot, None)
        if not p:
            return
        self._tasks[slot] = asyncio.create_task(
            self._work(slot, p["bv"], self._compose(p["texts"])))

    async def _work(self, slot: tuple[str, str], bv: str, question: str) -> None:
        key = slot[0]
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
            self._tasks.pop(slot, None)
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
        self._pending.clear()
        self.logger.info("[video_omni] 插件已停止。")
