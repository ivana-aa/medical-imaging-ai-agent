"""
WebSocketManager - WebSocket连接管理器
支持实时推送和双向通信
"""

import json
import logging
import threading
from typing import Dict, List, Callable, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        self.active_connections: List[Dict] = []
        self._lock = threading.RLock()

    async def connect(self, websocket, client_id: Optional[str] = None) -> str:
        """接受WebSocket连接"""
        await websocket.accept()
        
        conn_id = client_id or f"conn_{int(datetime.now().timestamp() * 1000)}"
        
        with self._lock:
            self.active_connections.append({
                'id': conn_id,
                'websocket': websocket,
                'connected_at': datetime.now().isoformat(),
                'client_info': websocket.client if websocket.client else {}
            })
            
        logger.info(f"WebSocket连接已建立: {conn_id}")
        
        # 发送欢迎消息
        await self.send_personal_message({
            'type': 'connected',
            'connection_id': conn_id,
            'timestamp': datetime.now().isoformat(),
            'message': '已连接到医影智诊智能体系统'
        }, websocket)
        
        return conn_id

    def disconnect(self, websocket):
        """断开WebSocket连接"""
        with self._lock:
            self.active_connections = [
                c for c in self.active_connections 
                if c['websocket'] != websocket
            ]
        logger.info("WebSocket连接已断开")

    async def send_personal_message(self, message: dict, websocket):
        """发送消息给单个连接"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"发送消息失败: {e}")

    async def broadcast(self, message: dict):
        """广播消息给所有连接"""
        with self._lock:
            connections = [c['websocket'] for c in self.active_connections]
            
        disconnected = []
        for websocket in connections:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"广播消息失败: {e}")
                disconnected.append(websocket)
                
        # 清理断开的连接
        for ws in disconnected:
            self.disconnect(ws)

    async def send_to_client(self, client_id: str, message: dict) -> bool:
        """发送消息给指定客户端"""
        with self._lock:
            conn = next(
                (c for c in self.active_connections if c['id'] == client_id),
                None
            )
            
        if conn:
            try:
                await conn['websocket'].send_json(message)
                return True
            except Exception as e:
                logger.error(f"发送消息给 {client_id} 失败: {e}")
                self.disconnect(conn['websocket'])
                
        return False

    def get_connection_count(self) -> int:
        """获取连接数"""
        with self._lock:
            return len(self.active_connections)

    def get_connections_info(self) -> List[Dict]:
        """获取所有连接信息"""
        with self._lock:
            return [
                {
                    'id': c['id'],
                    'connected_at': c['connected_at'],
                    'client': str(c['client_info'])
                }
                for c in self.active_connections
            ]


# 全局连接管理器实例
manager = ConnectionManager()


async def websocket_endpoint(websocket, client_id: Optional[str] = None):
    """
    WebSocket端点处理函数
    
    用于FastAPI路由:
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await ws_handler(websocket)
    """
    conn_id = await manager.connect(websocket, client_id)
    
    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                await handle_client_message(websocket, message)
            except json.JSONDecodeError:
                await manager.send_personal_message({
                    'type': 'error',
                    'message': '无效的JSON格式'
                }, websocket)
                
    except Exception as e:
        if 'WebSocketDisconnect' not in type(e).__name__:
            logger.error(f"WebSocket错误: {e}")
        manager.disconnect(websocket)
        logger.info(f"客户端断开连接: {conn_id}")


async def handle_client_message(websocket, message: dict):
    """
    处理客户端消息
    可以根据message['type']分发到不同的处理器
    """
    msg_type = message.get('type', 'unknown')
    
    # 这里会由主应用注入处理器
    handlers = _client_message_handlers.get(msg_type, [])
    
    for handler in handlers:
        try:
            await handler(websocket, message)
        except Exception as e:
            logger.error(f"消息处理器错误: {e}")
            await manager.send_personal_message({
                'type': 'error',
                'message': f'处理失败: {str(e)}'
            }, websocket)


# 消息处理器注册表
_client_message_handlers: Dict[str, List[Callable]] = {}


def register_client_message_handler(message_type: str, handler: Callable):
    """注册客户端消息处理器"""
    if message_type not in _client_message_handlers:
        _client_message_handlers[message_type] = []
    _client_message_handlers[message_type].append(handler)


# ==================== 便捷推送函数 ====================

async def push_notification(notification: dict):
    """推送通知给所有客户端"""
    await manager.broadcast({
        'type': 'notification',
        'data': notification,
        'timestamp': datetime.now().isoformat()
    })


async def push_file_event(event: dict):
    """推送文件监控事件"""
    await manager.broadcast({
        'type': 'file_event',
        'data': event,
        'timestamp': datetime.now().isoformat()
    })


async def push_analysis_result(result: dict):
    """推送分析结果"""
    await manager.broadcast({
        'type': 'analysis_result',
        'data': result,
        'timestamp': datetime.now().isoformat()
    })


async def push_agent_status(status: dict):
    """推送智能体状态"""
    await manager.broadcast({
        'type': 'agent_status',
        'data': status,
        'timestamp': datetime.now().isoformat()
    })


async def push_progress_update(task_id: str, progress: dict):
    """推送进度更新"""
    await manager.broadcast({
        'type': 'progress_update',
        'task_id': task_id,
        'data': progress,
        'timestamp': datetime.now().isoformat()
    })
