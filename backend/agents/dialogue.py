"""
DialogueAgent - 对话智能体
支持多轮对话、上下文记忆、追问解释
"""

import uuid
import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    SYSTEM_REMARK = "system_remark"  # 系统备注


class MessageType(Enum):
    TEXT = "text"
    IMAGE = "image"
    ANALYSIS_RESULT = "analysis_result"
    RECOMMENDATION = "recommendation"
    ERROR = "error"
    STATUS = "status"


@dataclass
class Message:
    """对话消息"""
    id: str
    role: str
    content: str
    timestamp: str
    message_type: str = "text"
    metadata: Dict = field(default_factory=dict)
    
    @classmethod
    def create(cls, role: str, content: str, message_type: str = "text", metadata: Dict = None) -> 'Message':
        return cls(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            message_type=message_type,
            metadata=metadata or {}
        )


@dataclass
class Conversation:
    """对话会话"""
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[Dict] = field(default_factory=list)
    context: Dict = field(default_factory=dict)  # 附加上下文（分析结果等）
    is_active: bool = True
    
    @classmethod
    def create(cls, title: str = None) -> 'Conversation':
        now = datetime.now().isoformat()
        return cls(
            id=str(uuid.uuid4()),
            title=title or f"对话 {datetime.now().strftime('%H:%M')}",
            created_at=now,
            updated_at=now
        )


class DialogueAgent:
    """
    对话智能体
    功能:
    - 多轮对话管理
    - 上下文记忆（基于当前分析结果）
    - 支持追问、解释、建议
    - 对话历史持久化
    """

    # 医疗问答模板
    MEDICAL_QA_TEMPLATES = {
        'explain_finding': {
            'patterns': ['解释', '说明', '什么意思', '是什么'],
            'template': '根据当前分析结果，{finding}表示{finding_desc}。这种情况{finding_significance}。'
        },
        'recommend_action': {
            'patterns': ['建议', '怎么办', '怎么处理', '如何治疗'],
            'template': '针对当前情况，建议：{recommendations}'
        },
        'compare_results': {
            'patterns': ['对比', '比较', '和之前'],
            'template': '与历史记录对比：{comparison}'
        },
        'risk_assessment': {
            'patterns': ['风险', '严重吗', '危险吗'],
            'template': '风险评估：{risk_info}'
        }
    }

    def __init__(self, max_history: int = 50, session_timeout: int = 3600):
        """
        初始化对话智能体
        
        Args:
            max_history: 每个对话最大消息数
            session_timeout: 会话超时时间（秒）
        """
        self.max_history = max_history
        self.session_timeout = session_timeout
        
        # 对话管理
        self.conversations: Dict[str, Conversation] = {}
        self.active_conversation_id: Optional[str] = None
        self._lock = threading.RLock()
        
        # 分析上下文（关联到当前分析结果）
        self.current_analysis_context: Dict = {}
        
        # 通知回调
        self.notification_callbacks: List[Callable] = []
        
        # 创建默认对话
        self.create_conversation("默认会话")
        
        logger.info("DialogueAgent 初始化完成")

    def create_conversation(self, title: str = None) -> str:
        """创建新对话会话"""
        conv = Conversation.create(title)
        with self._lock:
            self.conversations[conv.id] = conv
            self.active_conversation_id = conv.id
            
        self._notify({
            'type': 'conversation_created',
            'conversation': asdict(conv)
        })
        
        logger.info(f"创建新对话: {conv.id}")
        return conv.id

    def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        """获取对话会话"""
        with self._lock:
            return self.conversations.get(conv_id)

    def set_active_conversation(self, conv_id: str) -> bool:
        """设置当前活跃对话"""
        with self._lock:
            if conv_id in self.conversations:
                self.active_conversation_id = conv_id
                return True
        return False

    def get_active_conversation(self) -> Optional[Conversation]:
        """获取当前活跃对话"""
        with self._lock:
            if self.active_conversation_id:
                return self.conversations.get(self.active_conversation_id)
            return None

    def add_message(
        self, 
        role: str, 
        content: str, 
        message_type: str = "text",
        metadata: Dict = None,
        conv_id: str = None
    ) -> Optional[Message]:
        """添加消息到对话"""
        if not conv_id:
            conv_id = self.active_conversation_id
            
        with self._lock:
            conv = self.conversations.get(conv_id)
            if not conv:
                return None
                
            msg = Message.create(role, content, message_type, metadata)
            conv.messages.append(asdict(msg))
            
            # 限制历史长度
            if len(conv.messages) > self.max_history:
                conv.messages = conv.messages[-self.max_history:]
                
            conv.updated_at = datetime.now().isoformat()
            
            self._notify({
                'type': 'message_added',
                'conversation_id': conv_id,
                'message': asdict(msg)
            })
            
            return msg

    def set_analysis_context(self, analysis_result: Dict):
        """
        设置当前分析上下文
        用于后续对话引用
        """
        self.current_analysis_context = {
            'task_id': analysis_result.get('task_id'),
            'filename': analysis_result.get('filename'),
            'body_part': analysis_result.get('body_part'),
            'detection': analysis_result.get('detection', {}),
            'ai_report': analysis_result.get('ai_report'),
            'risk_level': analysis_result.get('detection', {}).get('risk_level'),
            'findings': analysis_result.get('detection', {}).get('abnormal_findings', []),
            'recommendation': analysis_result.get('detection', {}).get('clinical_recommendation'),
            'set_at': datetime.now().isoformat()
        }
        
        # 同时添加到当前对话上下文
        conv = self.get_active_conversation()
        if conv:
            conv.context = self.current_analysis_context.copy()
            
        logger.info(f"分析上下文已设置: {analysis_result.get('filename')}")

    def process_user_message(
        self, 
        user_input: str, 
        conv_id: str = None,
        analysis_context: Dict = None
    ) -> Dict:
        """
        处理用户消息并生成回复
        
        Args:
            user_input: 用户输入
            conv_id: 对话ID（可选）
            analysis_context: 外部传入的分析上下文（可选）
            
        Returns:
            回复结果
        """
        # 确定上下文
        ctx = analysis_context or self.current_analysis_context
        if not ctx and conv_id:
            conv = self.get_conversation(conv_id)
            if conv:
                ctx = conv.context
                
        # 添加用户消息
        user_msg = self.add_message('user', user_input, conv_id=conv_id)
        if not user_msg:
            return {'success': False, 'error': '无法添加消息'}
            
        # 生成回复
        response = self._generate_response(user_input, ctx)
        
        # 添加助手回复
        assistant_msg = self.add_message(
            'assistant', 
            response['content'],
            message_type=response.get('type', 'text'),
            metadata=response.get('metadata', {}),
            conv_id=conv_id
        )
        
        return {
            'success': True,
            'user_message': asdict(user_msg),
            'assistant_message': asdict(assistant_msg),
            'suggestions': response.get('suggestions', [])
        }

    def _generate_response(self, user_input: str, context: Dict) -> Dict:
        """
        生成回复
        这里可以接入LLM进行更智能的回复
        """
        user_lower = user_input.lower()
        
        # 意图识别
        intent = self._recognize_intent(user_lower)
        
        # 基于意图生成回复
        if intent == 'explain':
            return self._handle_explain(user_input, context)
        elif intent == 'recommend':
            return self._handle_recommend(user_input, context)
        elif intent == 'risk':
            return self._handle_risk(user_input, context)
        elif intent == 'compare':
            return self._handle_compare(user_input, context)
        elif intent == 'greeting':
            return self._handle_greeting(context)
        else:
            return self._handle_general(user_input, context)

    def _recognize_intent(self, text: str) -> str:
        """识别用户意图"""
        intent_keywords = {
            'explain': ['解释', '说明', '什么意思', '是什么', '怎么看', '哪个'],
            'recommend': ['建议', '怎么办', '怎么处理', '如何', '要不要'],
            'risk': ['风险', '严重吗', '危险吗', '会不会'],
            'compare': ['对比', '比较', '和之前', '区别', '不同'],
            'greeting': ['你好', 'hi', 'hello', '嗨', '在吗']
        }
        
        for intent, keywords in intent_keywords.items():
            for kw in keywords:
                if kw in text:
                    return intent
        return 'general'

    def _handle_explain(self, user_input: str, context: Dict) -> Dict:
        """处理解释类问题"""
        detection = context.get('detection', {})
        findings = detection.get('abnormal_findings', [])
        overall = detection.get('overall_assessment', '')
        
        content = ""
        suggestions = []
        
        if findings:
            for f in findings:
                content += f"【{f.get('type', '异常发现')}】\n"
                content += f"• 位置：{f.get('location', '未明确')}\n"
                content += f"• 描述：{f.get('description', '暂无详细描述')}\n"
                content += f"• 严重程度：{f.get('severity', '未知')}\n\n"
            content += f"综合评估：{overall}\n" if overall else ""
            suggestions = ["这个情况严重吗？", "我该怎么办？", "需要做哪些进一步检查？"]
        else:
            content = "当前影像分析未发现明显异常，这是一个正常的检查结果。\n"
            content += "综合评估：{}\n".format(overall) if overall else ""
            suggestions = ["有什么需要注意的吗？", "需要定期复查吗？"]
            
        return {
            'content': content,
            'type': 'text',
            'metadata': {'intent': 'explain'},
            'suggestions': suggestions
        }

    def _handle_recommend(self, user_input: str, context: Dict) -> Dict:
        """处理建议类问题"""
        recommendation = context.get('detection', {}).get('clinical_recommendation', '')
        risk_level = context.get('detection', {}).get('risk_level', 'low')
        
        content = "【临床建议】\n\n"
        
        if recommendation:
            content += recommendation + "\n\n"
        
        # 补充风险相关的建议
        if risk_level == 'high':
            content += "⚠️ 鉴于当前较高的风险等级，建议您：\n"
            content += "1. 尽快就医，接受专业医生的进一步诊断\n"
            content += "2. 如有不适症状，请立即前往急诊\n"
            content += "3. 不要延误治疗时机\n"
        elif risk_level == 'medium':
            content += "📋 建议：\n"
            content += "1. 按照上述建议进行进一步检查\n"
            content += "2. 如有需要，可在1-2周内进行复查\n"
            content += "3. 保持关注身体状况变化\n"
        else:
            content += "✅ 目前风险较低，建议：\n"
            content += "1. 定期体检，关注健康\n"
            content += "2. 如有不适，及时就诊\n"
            
        suggestions = ["还有其他需要注意的吗？", "这个建议的原理是什么？"]
        
        return {
            'content': content,
            'type': 'recommendation',
            'metadata': {'risk_level': risk_level},
            'suggestions': suggestions
        }

    def _handle_risk(self, user_input: str, context: Dict) -> Dict:
        """处理风险评估问题"""
        risk_level = context.get('detection', {}).get('risk_level', 'low')
        confidence = context.get('detection', {}).get('confidence', 0.9)
        
        risk_descriptions = {
            'low': "当前影像学表现提示风险较低，未发现明显的危急情况。AI分析置信度为{:.0%}。".format(confidence),
            'medium': "当前影像学表现存在一定风险，建议结合临床症状进一步评估。AI分析置信度为{:.0%}。".format(confidence),
            'high': "⚠️ 当前影像学表现提示较高风险，请务必尽快就医进行专业诊断。AI分析置信度为{:.0%}。".format(confidence)
        }
        
        content = "【风险评估】\n\n"
        content += risk_descriptions.get(risk_level, risk_descriptions['low'])
        content += "\n\n⚠️ 重要提醒：AI分析仅供参考，最终诊断请以专业医师的判断为准。\n"
        
        suggestions = ["具体的临床建议是什么？", "我需要做哪些检查？"]
        
        return {
            'content': content,
            'type': 'text',
            'metadata': {'risk_level': risk_level, 'confidence': confidence},
            'suggestions': suggestions
        }

    def _handle_compare(self, user_input: str, context: Dict) -> Dict:
        """处理对比类问题"""
        # 获取历史记录进行对比（简化版）
        content = "【历史对比分析】\n\n"
        content += "要查看与历史记录的对比，请访问\"历史记录\"页面进行详细查看。\n"
        content += "通过对比不同时期的影像变化，可以更好地评估病情进展。\n\n"
        content += "💡 提示：您可以上传同一患者不同时期的影像进行纵向对比分析。\n"
        
        suggestions = ["我目前的状况如何？", "有什么建议吗？"]
        
        return {
            'content': content,
            'type': 'text',
            'metadata': {'requires_history': True},
            'suggestions': suggestions
        }

    def _handle_greeting(self, context: Dict) -> Dict:
        """处理问候"""
        content = "您好！我是医影智诊的AI助手。\n\n"
        
        if context and context.get('task_id'):
            content += f"我看到您刚完成了一次影像分析（{context.get('filename', '未知文件')}）。\n"
            content += "您可以针对分析结果进行追问，我会尽力为您解答。\n\n"
        else:
            content += "您可以：\n"
            content += "• 上传医学影像进行分析\n"
            content += "• 针对已有分析结果进行追问\n"
            content += "• 咨询相关医学问题\n\n"
            
        content += "请随时告诉我您的需求 😊"
        
        suggestions = ["帮我分析这张影像", "解释一下分析结果", "有什么建议吗？"]
        
        return {
            'content': content,
            'type': 'text',
            'metadata': {'greeting': True},
            'suggestions': suggestions
        }

    def _handle_general(self, user_input: str, context: Dict) -> Dict:
        """处理一般问题"""
        content = "感谢您的提问。\n\n"
        
        if context and context.get('detection'):
            content += "我正在分析您的问题... 基于当前的影像分析结果：\n\n"
            detection = context.get('detection', {})
            content += f"• 检查部位：{context.get('body_part', '未知')}\n"
            content += f"• 风险等级：{detection.get('risk_level', '未知')}\n"
            content += f"• 异常发现：{len(detection.get('abnormal_findings', []))}项\n\n"
            content += "您可以针对以下方面进行更具体的提问：\n"
            content += "• 解释某个发现的具体含义\n"
            content += "• 了解相关的临床建议\n"
            content += "• 评估当前风险情况\n"
        else:
            content += "目前没有关联的分析结果。您可以先上传一张医学影像，我会为您提供专业的分析。\n"
            
        suggestions = ["上传影像进行分析", "解释一下分析结果", "给我一些建议"]
        
        return {
            'content': content,
            'type': 'text',
            'metadata': {'general': True},
            'suggestions': suggestions
        }

    def get_conversation_history(self, conv_id: str, limit: int = 20) -> List[Dict]:
        """获取对话历史"""
        with self._lock:
            conv = self.conversations.get(conv_id)
            if not conv:
                return []
            messages = conv.messages[-limit:] if limit > 0 else conv.messages
            return messages

    def list_conversations(self) -> List[Dict]:
        """列出所有对话"""
        with self._lock:
            return [
                {
                    'id': c.id,
                    'title': c.title,
                    'created_at': c.created_at,
                    'updated_at': c.updated_at,
                    'message_count': len(c.messages),
                    'is_active': c.id == self.active_conversation_id
                }
                for c in self.conversations.values()
            ]

    def delete_conversation(self, conv_id: str) -> bool:
        """删除对话"""
        with self._lock:
            if conv_id in self.conversations:
                del self.conversations[conv_id]
                if self.active_conversation_id == conv_id:
                    # 切换到第一个可用对话
                    self.active_conversation_id = next(iter(self.conversations.keys())) if self.conversations else None
                return True
        return False

    def clear_context(self):
        """清空当前分析上下文"""
        self.current_analysis_context = {}
        logger.info("分析上下文已清空")

    def add_notification_callback(self, callback: Callable):
        """添加通知回调"""
        self.notification_callbacks.append(callback)

    def _notify(self, message: Dict):
        """发送通知"""
        for callback in self.notification_callbacks:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"通知发送失败: {e}")

    def get_status(self) -> Dict:
        """获取状态"""
        with self._lock:
            return {
                'total_conversations': len(self.conversations),
                'active_conversation_id': self.active_conversation_id,
                'has_analysis_context': bool(self.current_analysis_context),
                'context_filename': self.current_analysis_context.get('filename')
            }


# 便捷函数
def create_dialogue_agent(max_history: int = 50) -> DialogueAgent:
    """创建对话智能体实例"""
    return DialogueAgent(max_history=max_history)
