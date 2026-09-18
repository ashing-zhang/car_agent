# 意图分类器 Service 接口 - 定义 classify 契约
# 运行指南:
#   from app.agent.intent.classifier import IntentClassifier
#   继承并实现 classify(user_message) -> IntentClassification

import logging
from abc import ABC, abstractmethod

from app.agent.intent.models import IntentClassification

logger = logging.getLogger(__name__)


class IntentClassifier(ABC):
    """意图分类器抽象基类:对扩展开放(新增分类器),对修改关闭。"""

    @abstractmethod
    def classify(self, user_message: str, user_id: str = "demo-user") -> IntentClassification:
        """根据用户消息进行意图分类,返回 AgentType 及置信度。"""
        raise NotImplementedError
