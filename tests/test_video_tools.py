"""验证原消息保留、长工具继续、错误与截图输入隔离。"""
import asyncio
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock, patch
import importlib.util
root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "video_tool_test_plugin", root / "__init__.py",
    submodule_search_locations=[str(root)],
)
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from video_tool_test_plugin import main
from video_tool_test_plugin.core.omni import OmniError

class VideoToolTests(unittest.IsolatedAsyncioTestCase):
    """测试工具返回行为，避免用模拟消息发送替代实际结果。"""

    def setUp(self):
        """准备不会发送消息的上下文。"""
        self.ctx = MagicMock()
        self.bot = main.Main(self.ctx, {"show_usage": False})
        self.event = MagicMock()
        extras = {}
        self.event.set_extra.side_effect = lambda k,v: extras.__setitem__(k,v)
        self.event.get_extra.side_effect = lambda k: extras.get(k)
        self.event.unified_msg_origin = "qq:GroupMessage:test-group"
        self.event.get_sender_id.return_value = "member-123"
        self.event.message_str = "https://www.bilibili.com/video/BV1kERuBYEyy 做成文档并发给指定的人"
        self.res = {"duration": 100, "answer": "00:30 配置画面"}
        self.bot._analyse = AsyncMock(return_value=self.res)

    def prepare_parser_event(self):
        """模拟 Parser X 已发送媒体、尚未进入主模型的真实事件状态。"""
        self.event.is_stopped.return_value = False
        self.event.is_at_or_wake_command = True
        self.event._has_send_oper = True
        self.event.call_llm = False
        self.event.get_result.return_value = None
        self.event.set_extra("activated_handlers", [
            SimpleNamespace(handler_module_path="data.plugins.astrbot_plugin_parser_x.main")])
        self.ctx.get_config.return_value = {"provider_settings": {"enable": True, "wake_prefix": ""}}
        self.conversation = SimpleNamespace(cid="original-cid", history="[]")
        self.ctx.conversation_manager.get_curr_conversation_id = AsyncMock(return_value="original-cid")
        self.ctx.conversation_manager.get_conversation = AsyncMock(return_value=self.conversation)
        from astrbot.core.provider.entities import ProviderRequest
        self.event.request_llm.side_effect = lambda **kw: ProviderRequest(**kw)

    async def test_native_pipeline_resumes_after_parser_once(self):
        """真实 ProcessStage 必须在媒体发送后调用模型且只调用一次。"""
        from astrbot.core.pipeline.process_stage.stage import ProcessStage
        from astrbot.core.db.po import Conversation
        self.prepare_parser_event()
        # ProviderRequest 的 conversation 是实际数据库对象，而非模拟对象。
        self.conversation = Conversation(cid="original-cid", user_id="test-user", platform_id="qq", history="[]")
        self.ctx.conversation_manager.get_conversation.return_value = self.conversation
        stage = ProcessStage()
        stage.ctx = SimpleNamespace(astrbot_config={"provider_settings": {"enable": True}})
        captured = []
        async def plugin_stage(event):
            async for value in self.bot.resume_after_parser(event):
                yield value
        async def agent_stage(event):
            captured.append(event.get_extra("provider_request"))
            yield None
        stage.star_request_sub_stage = SimpleNamespace(process=plugin_stage)
        stage.agent_sub_stage = SimpleNamespace(process=agent_stage)
        with patch("astrbot.core.pipeline.process_stage.stage.FILTERS", []):
            async for _ in stage.process(self.event):
                pass
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].prompt, self.event.message_str)
        self.assertIs(captured[0].conversation, self.conversation)
        self.assertEqual([x async for x in self.bot.resume_after_parser(self.event)], [])

    async def test_resume_respects_wake_and_disable(self):
        """未唤醒、主动禁用及已停止的消息不能被强制继续。"""
        for name, value in [("is_at_or_wake_command", False), ("call_llm", True), ("_has_send_oper", False)]:
            self.prepare_parser_event()
            setattr(self.event, name, value)
            self.assertEqual([x async for x in self.bot.resume_after_parser(self.event)], [])
        self.prepare_parser_event()
        self.event.is_stopped.return_value = True
        self.assertEqual([x async for x in self.bot.resume_after_parser(self.event)], [])
        self.prepare_parser_event()
        self.ctx.get_config.return_value = {"provider_settings": {"enable": False}}
        self.assertEqual([x async for x in self.bot.resume_after_parser(self.event)], [])
        self.ctx.get_config.return_value = {"provider_settings": {"enable": True, "wake_prefix": "ask "}}
        self.assertEqual([x async for x in self.bot.resume_after_parser(self.event)], [])

    async def test_resume_ignores_unrelated_plugin_and_existing_request(self):
        """不接管其他插件回复，也不重复已有模型请求。"""
        self.prepare_parser_event()
        self.event.set_extra("activated_handlers", [SimpleNamespace(handler_module_path="another.plugin")])
        self.assertEqual([x async for x in self.bot.resume_after_parser(self.event)], [])
        self.prepare_parser_event()
        self.event.set_extra("provider_request", object())
        self.assertEqual([x async for x in self.bot.resume_after_parser(self.event)], [])

    async def test_original_intent_untouched(self):
        """文档、对比、转发等要求原样留给主模型。"""
        for prompt in ["视频转文档", "比较这个和昨天的方案", "把链接发给小王", "看完后起草邮件"]:
            req = SimpleNamespace(prompt=prompt, system_prompt="persona", func_tool=MagicMock())
            await self.bot.route(self.event, req)
            self.assertEqual(req.prompt, prompt)
            self.assertTrue(req.system_prompt.startswith("persona"))
        self.event.stop_event.assert_not_called()
        self.ctx.send_message.assert_not_called()
        self.assertFalse(hasattr(self.bot, "on_message"))

    async def test_completed_is_tool_data(self):
        """分析结果不自行交付给聊天。"""
        result = json.loads(await self.bot.understand(self.event, "BV1kERuBYEyy", "配置步骤"))
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["delivery_completed"])
        self.assertEqual(result["evidence"], self.res["answer"])
        self.ctx.send_message.assert_not_called()

    async def test_long_call_keeps_original_event(self):
        """视频工具独立放宽超时，结果返回后恢复原限制。"""
        async def slow(*args):
            await asyncio.sleep(.08)
            return self.res
        self.bot._analyse = slow
        from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor
        from astrbot.core.agent.tool import FunctionTool
        tool = FunctionTool(name="video_omni_understand", description="test", parameters={})
        tool.handler = self.bot.understand
        run = SimpleNamespace(context=SimpleNamespace(event=self.event), tool_call_timeout=.04)
        await self.bot.bind_run(self.event, run)
        await self.bot.tool_start(self.event, tool, {})
        output = [x async for x in FunctionToolExecutor._execute_local(
            tool, run, video="BV1kERuBYEyy", question="制作文件")]
        self.assertEqual(len(output), 1)
        self.assertEqual(json.loads(output[-1].content[0].text)["status"], "completed")
        self.assertIs(run.context.event, self.event)
        await self.bot.tool_end(self.event, tool, {}, output[-1])
        self.assertEqual(run.tool_call_timeout, .04)
        await self.bot.tool_start(self.event, tool, {})
        await self.bot.tool_start(self.event, SimpleNamespace(name="other"), {})
        self.assertEqual(run.tool_call_timeout, .04)
        await self.bot.run_done(self.event, run, None)
        self.assertIsNone(self.event.get_extra("_video_omni_run"))

    async def test_failure_is_not_success(self):
        """失败返回错误，不能假称已经观看。"""
        self.bot._analyse.side_effect = OmniError("test failure")
        result = await self.bot.understand(self.event, "BV1kERuBYEyy", "问题")
        self.assertEqual(json.loads(result)["status"], "failed")

    async def test_cancel_reaps_analysis(self):
        """消费者结束后回收在途任务。"""
        async def slow(*args):
            await asyncio.sleep(60)
        self.bot._analyse = slow
        task = asyncio.create_task(self.bot.understand(self.event, "BV1kERuBYEyy", "问题"))
        await asyncio.sleep(.01)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.assertFalse(self.bot._tasks)

    async def test_disabled_and_invalid(self):
        """关闭时隐藏工具，非法输入不启动分析。"""
        self.bot.config["enabled"] = False
        req = SimpleNamespace(func_tool=MagicMock(), prompt="原话", system_prompt="")
        await self.bot.route(self.event, req)
        self.assertEqual(req.func_tool.remove_tool.call_count, 2)
        self.bot.config["enabled"] = True
        self.assertIn("编号", (await self.bot.understand(self.event, "bad", "x")))
        self.bot._analyse.assert_not_called()
        for times in [[-1], [float("nan")], [True], list(range(7))]:
            result = await self.bot.frames(self.event, "BV1kERuBYEyy", times)
            self.assertFalse(result.startswith("{"))

    async def test_frames_register_for_original_member(self):
        """截图登记使用原事件身份，不使用模型指定的用户。"""
        engine = MagicMock()
        engine._cache_hit.return_value = "/tmp/source.mp4"
        engine.work_dir = "/tmp"
        office = MagicMock()
        office._eligible.return_value = True
        office._owner.return_value = "scoped-owner"
        office._capture_one = AsyncMock(return_value={"file_id":"image123","input_name":"frame.jpg","filename":"frame.jpg"})
        self.ctx.get_registered_star.return_value = SimpleNamespace(activated=True, star_cls=office)
        with patch.object(self.bot, "_omni", return_value=engine), patch.object(main, "probe", return_value={"duration":100}), patch.object(main, "_run", new_callable=AsyncMock):
            result = json.loads(await self.bot.frames(self.event, "BV1kERuBYEyy", [30]))
        office._owner.assert_called_once_with(self.event)
        self.assertEqual(office._capture_one.call_args.args[0], "scoped-owner")
        self.assertEqual(result["frames"][0]["office_input"]["file_id"], "image123")
        self.ctx.send_message.assert_not_called()

    async def test_frames_respect_office_permissions(self):
        """群权限未满足时不偷偷登记文档输入。"""
        engine = MagicMock()
        engine._cache_hit.return_value = "/tmp/source.mp4"
        engine.work_dir = "/tmp"
        office = MagicMock()
        office._eligible.return_value = False
        self.ctx.get_registered_star.return_value = SimpleNamespace(activated=True, star_cls=office)
        with patch.object(self.bot, "_omni", return_value=engine), patch.object(main, "probe", return_value={"duration":100}), patch.object(main, "_run", new_callable=AsyncMock):
            result = json.loads(await self.bot.frames(self.event, "BV1kERuBYEyy", [30]))
        office._capture_one.assert_not_called()
        self.assertIn("office_note", result["frames"][0])

if __name__ == "__main__":
    unittest.main(verbosity=2)
