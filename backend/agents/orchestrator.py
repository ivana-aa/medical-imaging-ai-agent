"""
AgentOrchestrator - 智能体编排器
协调所有智能体工作，统一管理状态和通信
"""

import json
import logging
import asyncio
import threading
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum

from .file_watcher import FileWatcherAgent
from .dialogue import DialogueAgent
from .planner import PlannerAgent

logger = logging.getLogger(__name__)


class AgentType(Enum):
    FILE_WATCHER = "file_watcher"
    DIALOGUE = "dialogue"
    PLANNER = "planner"


@dataclass
class AgentMessage:
    """智能体消息"""
    from_agent: str
    to_agent: str  # 'broadcast' for all
    message_type: str
    content: Any
    timestamp: str
    metadata: Dict = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        from_agent: str,
        to_agent: str,
        msg_type: str,
        content: Any,
        metadata: Optional[Dict] = None,
    ) -> 'AgentMessage':
        return cls(
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=msg_type,
            content=content,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )


class AgentOrchestrator:
    """
    智能体编排器
    功能:
    - 统一初始化和管理所有智能体
    - 协调智能体之间的通信
    - 提供统一的API接口
    - 管理WebSocket推送
    - 统一状态管理
    """

    def __init__(self, base_dir: str):
        """
        初始化智能体编排器
        
        Args:
            base_dir: 基础目录（用于文件监控等）
        """
        self.base_dir = base_dir
        self._lock = threading.RLock()
        
        # 智能体实例
        self.agents: Dict[str, Any] = {}
        self._initialize_agents()
        
        # 消息队列
        self._message_queue: List[AgentMessage] = []
        self._message_handlers: Dict[str, List[Callable]] = {}
        
        # WebSocket连接管理
        self._websocket_connections: List[Callable] = []
        
        # 事件处理
        self._event_handlers: Dict[str, List[Callable]] = {}
        
        # 统计
        self.stats = {
            'total_events': 0,
            'total_messages': 0,
            'start_time': datetime.now().isoformat()
        }
        
        logger.info("AgentOrchestrator 初始化完成")

    def _initialize_agents(self):
        """初始化所有智能体"""
        # 文件监控智能体
        watch_dir = f"{self.base_dir}/watched_folders"
        self.agents['file_watcher'] = FileWatcherAgent(
            watch_dir=watch_dir,
            analyzer_callback=None  # 稍后设置
        )
        
        # 对话智能体
        self.agents['dialogue'] = DialogueAgent(
            max_history=50
        )
        
        # 任务规划智能体
        self.agents['planner'] = PlannerAgent()
        
        # 设置文件监控回调
        self.agents['file_watcher'].analyzer_callback = self._on_auto_analyze
        
        # 设置通知回调
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'add_notification_callback'):
                agent.add_notification_callback(
                    lambda msg, an=agent_name: self._handle_agent_notification(an, msg)
                )
                
        logger.info(f"已初始化 {len(self.agents)} 个智能体")

    def _handle_agent_notification(self, agent_name: str, message: Dict):
        """处理智能体通知"""
        # 转发给WebSocket
        self._broadcast({
            'source': agent_name,
            **message
        })
        
        # 处理特定事件
        if message.get('type') == 'processing_completed':
            # 自动更新对话上下文
            result = message.get('result', {})
            if result:
                self.agents['dialogue'].set_analysis_context(result)
                
        self.stats['total_events'] += 1

    # ==================== 文件监控智能体接口 ====================
    
    def start_file_watching(self, watch_dir: Optional[str] = None) -> Dict:
        """启动文件监控"""
        watcher = self.agents['file_watcher']
        
        if watch_dir:
            watcher.set_watch_directory(watch_dir)
            
        success = watcher.start()
        
        return {
            'success': success,
            'status': watcher.get_status()
        }

    def stop_file_watching(self) -> Dict:
        """停止文件监控"""
        self.agents['file_watcher'].stop()
        return {'success': True}

    def get_watcher_status(self) -> Dict:
        """获取文件监控状态"""
        return self.agents['file_watcher'].get_status()

    def get_pending_files(self) -> List[Dict]:
        """获取待处理文件"""
        return self.agents['file_watcher'].get_pending_files()

    def analyze_file(self, file_path: str) -> Dict:
        """手动分析文件"""
        return self.agents['file_watcher'].reprocess_file(file_path)

    def _on_auto_analyze(self, file_path: str, filename: str) -> Dict:
        """自动分析回调"""
        logger.info(f"自动分析文件: {filename}")
        
        # 这里需要调用主分析逻辑
        # 由于编排器不直接持有analyzer，需要通过事件或回调
        return {
            'status': 'analyzing',
            'filename': filename
        }

    # ==================== 对话智能体接口 ====================

    def send_message(self, user_input: str, conv_id: Optional[str] = None) -> Dict:
        """发送消息并获取回复"""
        result = self.agents['dialogue'].process_user_message(user_input, conv_id)
        return result

    def create_conversation(self, title: Optional[str] = None) -> str:
        """创建新对话"""
        return self.agents['dialogue'].create_conversation(title)

    def get_conversation_history(self, conv_id: Optional[str] = None, limit: int = 20) -> List[Dict]:
        """获取对话历史"""
        return self.agents['dialogue'].get_conversation_history(conv_id, limit)

    def list_conversations(self) -> List[Dict]:
        """列出所有对话"""
        return self.agents['dialogue'].list_conversations()

    def set_analysis_context(self, analysis_result: Dict):
        """设置分析上下文"""
        self.agents['dialogue'].set_analysis_context(analysis_result)

    def clear_context(self):
        """清空上下文"""
        self.agents['dialogue'].clear_context()

    # ==================== 任务规划智能体接口 ====================

    def create_analysis_plan(
        self, 
        body_part: Optional[str] = None,
        custom_steps: Optional[List[Dict]] = None
    ) -> str:
        """创建分析计划"""
        # 根据部位选择工作流
        workflow_map = {
            'chest': 'chest_analysis',
            'brain': 'brain_analysis',
            'abdomen': 'abdomen_analysis'
        }
        
        workflow_key = workflow_map.get(body_part) if body_part else None
        
        plan_id = self.agents['planner'].create_plan(
            name=f"{body_part or '综合'}影像分析计划",
            description=f"针对{body_part or '多部位'}影像的智能分析计划",
            workflow_key=workflow_key,
            custom_steps=custom_steps
        )
        
        return plan_id

    def execute_plan(self, plan_id: str, context: Optional[Dict] = None) -> Dict:
        """执行分析计划"""
        return self.agents['planner'].execute_plan(plan_id, context)

    def get_plan_status(self, plan_id: str) -> Optional[Dict]:
        """获取计划状态"""
        return self.agents['planner'].get_plan_status(plan_id)

    def list_plans(self) -> List[Dict]:
        """列出所有计划"""
        return self.agents['planner'].list_plans()

    def stop_plan(self):
        """停止当前计划"""
        self.agents['planner'].stop_execution()

    # ==================== 智能体间通信 ====================

    def send_message_to_agent(
        self, 
        from_agent: str, 
        to_agent: str, 
        msg_type: str, 
        content: Any
    ) -> bool:
        """向指定智能体发送消息"""
        msg = AgentMessage.create(from_agent, to_agent, msg_type, content)
        
        with self._lock:
            self._message_queue.append(msg)
            self.stats['total_messages'] += 1
            
        # 处理消息
        self._process_message(msg)
        
        return True

    def _process_message(self, message: AgentMessage):
        """处理智能体消息"""
        # 调用注册的处理器
        handlers = self._message_handlers.get(message.message_type, [])
        for handler in handlers:
            try:
                handler(message)
            except Exception as e:
                logger.error(f"消息处理失败: {e}")

    def register_message_handler(self, message_type: str, handler: Callable):
        """注册消息处理器"""
        if message_type not in self._message_handlers:
            self._message_handlers[message_type] = []
        self._message_handlers[message_type].append(handler)

    # ==================== WebSocket 推送 ====================

    def add_websocket_connection(self, send_callback: Callable):
        """添加WebSocket连接"""
        self._websocket_connections.append(send_callback)

    def remove_websocket_connection(self, send_callback: Callable):
        """移除WebSocket连接"""
        if send_callback in self._websocket_connections:
            self._websocket_connections.remove(send_callback)

    def _broadcast(self, message: Dict):
        """广播消息给所有WebSocket连接"""
        for callback in self._websocket_connections:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"WebSocket推送失败: {e}")

    def send_to_client(self, message: Dict):
        """发送消息给客户端"""
        self._broadcast(message)

    # ==================== 事件处理 ====================

    def register_event_handler(self, event_type: str, handler: Callable):
        """注册事件处理器"""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def trigger_event(self, event_type: str, data: Any = None):
        """触发事件"""
        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(data)
            except Exception as e:
                logger.error(f"事件处理失败: {e}")

    # ==================== 统一状态接口 ====================

    def get_all_status(self) -> Dict:
        """获取所有智能体状态"""
        return {
            'orchestrator': {
                'agent_count': len(self.agents),
                'websocket_connections': len(self._websocket_connections),
                'stats': self.stats
            },
            'file_watcher': self.agents['file_watcher'].get_status(),
            'dialogue': self.agents['dialogue'].get_status(),
            'planner': self.agents['planner'].get_status()
        }

    def shutdown(self):
        """关闭所有智能体"""
        # 停止文件监控
        if 'file_watcher' in self.agents:
            self.agents['file_watcher'].stop()
            
        logger.info("AgentOrchestrator 已关闭")

    # ==================== 组合操作 ====================

    def start_full_analysis(self, file_path: str, filename: str, body_part: Optional[str] = None) -> Dict:
        """
        完整的分析流程
        1. 创建分析计划
        2. 执行计划
        3. 更新对话上下文
        """
        # 1. 创建计划
        plan_id = self.create_analysis_plan(body_part)
        
        # 2. 获取分析结果作为上下文
        # 这里假设已有分析结果，实际使用时需要整合
        context = {
            'file_path': file_path,
            'filename': filename,
            'body_part': body_part
        }
        
        # 3. 执行计划
        result = self.execute_plan(plan_id, context)
        
        return {
            'plan_id': plan_id,
            'execution_result': result
        }

    def get_system_overview(self) -> Dict:
        """获取系统概览"""
        return {
            'timestamp': datetime.now().isoformat(),
            'agents': {
                'file_watcher': {
                    'name': '文件监控智能体',
                    'status': self.agents['file_watcher'].state.value,
                    'capabilities': ['文件夹监控', '自动分析', '实时推送']
                },
                'dialogue': {
                    'name': '对话智能体',
                    'status': 'running',
                    'capabilities': ['多轮对话', '上下文记忆', '智能问答']
                },
                'planner': {
                    'name': '任务规划智能体',
                    'status': 'running',
                    'capabilities': ['问题拆解', '步骤规划', '状态追踪']
                }
            },
            'stats': self.stats
        }


# 便捷函数
def create_orchestrator(base_dir: str) -> AgentOrchestrator:
    """创建智能体编排器"""
    return AgentOrchestrator(base_dir)
