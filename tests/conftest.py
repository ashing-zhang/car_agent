# pytest 配置 - 确保无 __init__.py 时测试可发现 app 包
# 运行指南: pytest 自动加载此文件
#
# 测试桩说明:
#   IntentStubLLM:       单元测试专用 LLM 桩,按关键词映射生成工具调用,验证 Agent 图数据流。
#   HashEmbeddingStub:   单元/集成测试专用 Embedding 桩,基于文本哈希生成确定性向量,维度 1536。
#   两者都不是生产实现的 mock,而是用于在无真实 API Key 环境下验证除模型本身外的逻辑正确性。

import hashlib
import re
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class IntentStubLLM:
    """单元测试专用的 LLM 调用桩(Stub),不依赖真实 API Key。
    按关键词映射生成 tool_calls 或文本回复,用于验证 Agent 图、数据流、Policy、记忆等逻辑。
    """

    def __init__(self) -> None:
        """初始化调用计数器,用于构造唯一 tool_call id。"""
        self._call_count: int = 0
        self._tools: list[Any] = []

    def bind_tools(self, tools: list[Any]) -> "IntentStubLLM":
        """缓存可用工具列表,兼容 LLMProvider 接口。"""
        self._tools = tools
        return self

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        """根据最后一条用户消息的关键词内容匹配生成工具调用或文本回复。"""
        self._call_count += 1
        last = messages[-1] if messages else None
        last_type = getattr(last, "type", "") if last else ""
        if last_type == "tool":
            return AIMessage(content=self._summarize(last))
        user_text = ""
        for msg in reversed(messages):
            if getattr(msg, "type", "") == "human":
                user_text = msg.content if isinstance(msg.content, str) else str(msg.content)
                break
        full_text = " ".join(
            m.content if isinstance(m.content, str) else str(m.content) for m in messages
        )
        matched = self._match(user_text, full_text)
        if matched is not None:
            if isinstance(matched, list):
                return AIMessage(content="", tool_calls=matched)
            return AIMessage(content="", tool_calls=[matched])
        return AIMessage(content="我已了解您的需求,正在为您处理。")

    # ---------------- 内部辅助 ----------------

    def _summarize(self, tool_msg: BaseMessage) -> str:
        """根据工具执行结果生成面向用户的总结。"""
        content = tool_msg.content if isinstance(tool_msg.content, str) else str(tool_msg.content)
        if "超出安全范围" in content or "policy" in content:
            return "抱歉,该温度超出安全范围,我无法设置。"
        return content

    def _extract_dest(self, text: str) -> str | None:
        """从导航请求中提取目的地关键词。"""
        nav_keywords = ("导航到", "导航去", "去", "回", "前往", "出发去", "怎么去", "到")
        stop_words = (
            "顺便", "然后", "同时", "路上", "途中", "的路上", "的时候",
            "并", "且", "和", "跟", "还有", "再", "还", "，", ",", "。", ".",
        )
        best: str | None = None
        best_pos: int = -1
        for kw in nav_keywords:
            idx = text.find(kw)
            if idx < 0:
                continue
            if best_pos >= 0 and idx >= best_pos:
                continue
            rest = text[idx + len(kw):]
            earliest = len(rest)
            for stop in stop_words:
                pos = rest.find(stop)
                if 0 <= pos < earliest:
                    earliest = pos
            rest = rest[:earliest].strip()
            if rest and rest not in ("哪", "哪里", "哪儿", "什么"):
                best = rest
                best_pos = idx
        return best

    def _match(self, text: str, full_text: str) -> list[dict] | dict | None:
        """基于关键词匹配工具调用,支持多工具组合。"""
        calls: list[dict] = []
        base_id = self._call_count
        seq = 0

        def _mk(name: str, args: dict) -> dict:
            nonlocal seq
            seq += 1
            return {"name": name, "args": args, "id": f"stub_{base_id}_{seq}", "type": "tool_call"}

        dest = self._extract_dest(text)
        if dest is not None:
            calls.append(_mk("search_destination", {"query": dest}))
            calls.append(_mk("start_navigation", {"destination": f"{dest} 大厦"}))

        if any(k in text for k in ("天气", "下雨", "下雪", "外面下")) and "路况" not in text and "交通" not in text:
            calls.append(_mk("get_weather", {}))
        if any(k in text for k in ("路况", "交通状况", "前方交通", "交通密度", "交通情况")):
            calls.append(_mk("get_traffic", {}))
        if any(k in text for k in ("周围", "场景", "环境", "还适合", "路线", "前方")):
            calls.append(_mk("get_camera_scene", {}))

        pause_like = any(k in text for k in ("暂停音乐", "停止音乐", "暂停播放", "停止播放"))
        if pause_like:
            calls.append(_mk("pause_media", {}))
        elif any(k in text for k in ("播放", "放点", "放首", "音乐", "歌单", "听歌")):
            playlist = "默认歌单"
            if "回家" in text or "下班" in text:
                playlist = "下班放松歌单"
            elif "上班" in text or "公司" in text or "工作" in text:
                playlist = "上班通勤歌单"
            elif "轻音乐" in text:
                playlist = "轻音乐"
            calls.append(_mk("play_media", {"playlist": playlist}))

        if "音量" in text:
            vm = re.search(r"(\d+)", text)
            level = int(vm.group(1)) if vm else 15
            calls.append(_mk("set_volume", {"level": level}))

        if any(k in text for k in ("车速", "速度", "电量", "续航", "车辆状态")):
            calls.append(_mk("get_vehicle_status", {}))
        if any(k in text for k in ("车内温度", "多少度")) and not any(k in text for k in ("调", "设", "温度调到", "温度设为")):
            calls.append(_mk("get_cabin_temperature", {}))

        m = re.search(r"(\d+(?:\.\d+)?)\s*度", text)
        explicit_temp = any(k in text for k in ("调到", "设为", "调成", "保持", "习惯", "喜欢", "平时"))
        if explicit_temp and m:
            calls.append(_mk("set_temperature", {"temperature_c": float(m.group(1))}))
        elif any(k in text for k in ("冷", "热", "调温度", "温度调")) and m:
            calls.append(_mk("set_temperature", {"temperature_c": float(m.group(1))}))
        elif any(k in text for k in ("冷", "热")) and not m:
            pref = None
            match_pref = re.search(r"偏好[^0-9]*(\d+(?:\.\d+)?)", full_text)
            if match_pref:
                pref = float(match_pref.group(1))
            temp = pref if pref is not None else (24.0 if "冷" in text else 22.0)
            calls.append(_mk("set_temperature", {"temperature_c": temp}))
        elif explicit_temp and not m:
            for kw, default in [("24", 24.0), ("25", 25.0), ("23", 23.0), ("22", 22.0), ("26", 26.0)]:
                if kw in text:
                    calls.append(_mk("set_temperature", {"temperature_c": default}))
                    break

        if any(k in text for k in ("开空调", "开ac", "启动空调", "打开空调")):
            calls.append(_mk("set_ac", {"enabled": True}))
        if any(k in text for k in ("关空调", "关ac", "关闭空调", "关掉空调")):
            calls.append(_mk("set_ac", {"enabled": False}))

        if not calls:
            return None
        if len(calls) == 1:
            return calls[0]
        return calls


class HashEmbeddingStub:
    """单元/集成测试专用的 Embedding 调用桩,基于文本 SHA-256 哈希生成确定性向量。
    相同文本返回相同向量,保证测试可重复;向量维度与生产实现一致(1536),并做 L2 归一化。
    """

    dimension: int = 1536

    def __init__(self, seed: int = 42) -> None:
        """初始化随机种子,控制哈希前缀,避免不同测试间向量意外冲突。"""
        self._seed = seed

    def embed(self, text: str) -> list[float]:
        """将单条文本编码为向量。"""
        return self._hash_to_vec(text)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """批量编码文本为向量列表。"""
        return [self._hash_to_vec(t) for t in texts]

    def _hash_to_vec(self, text: str) -> list[float]:
        """对文本做 SHA-256,将字节流展开为 1536 维归一化向量。"""
        vec: list[float] = []
        digest = hashlib.sha256((f"{self._seed}:" + text).encode("utf-8")).digest()
        for i in range(self.dimension):
            byte_idx = i % len(digest)
            shift = (i // len(digest)) % 8
            val = ((digest[byte_idx] >> shift) & 0xFF) / 255.0
            vec.append((val - 0.5) * 2.0)
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]
