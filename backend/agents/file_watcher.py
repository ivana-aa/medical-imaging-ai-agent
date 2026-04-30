"""
FileWatcherAgent - 文件夹监控智能体
自动监控指定文件夹，发现新文件后触发分析
"""

import os
import json
import time
import logging
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class WatcherState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


class FileWatcherAgent:
    """
    文件夹监控智能体
    功能:
    - 监控指定文件夹的新文件
    - 自动过滤已处理的文件
    - 支持多种医学影像格式
    - 触发分析回调
    - 实时推送通知
    """

    SUPPORTED_EXTENSIONS = {'.dcm', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    def __init__(self, watch_dir: str, analyzer_callback: Optional[Callable] = None):
        """
        初始化文件监控器
        
        Args:
            watch_dir: 要监控的文件夹路径
            analyzer_callback: 发现新文件时的回调函数
        """
        self.watch_dir = Path(watch_dir)
        self.analyzer_callback = analyzer_callback
        self.state = WatcherState.STOPPED
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # 默认不暂停
        
        # 状态
        self.processed_files: set = set()  # 已处理文件记录
        self.pending_files: List[Dict] = []  # 待处理文件队列
        self.stats = {
            'total_detected': 0,
            'total_processed': 0,
            'total_errors': 0,
            'start_time': None,
            'last_processed': None
        }
        
        # 通知回调 (WebSocket推送等)
        self.notification_callbacks: List[Callable] = []
        
        # 确保目录存在
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"FileWatcherAgent 初始化完成，监控目录: {self.watch_dir}")

    def start(self) -> bool:
        """启动文件监控"""
        if self.state == WatcherState.RUNNING:
            logger.warning("FileWatcher 已经运行中")
            return True
            
        if self.state == WatcherState.STARTING:
            logger.warning("FileWatcher 正在启动中")
            return False
            
        try:
            self.state = WatcherState.STARTING
            self._stop_event.clear()
            self._pause_event.set()
            
            # 启动监控线程
            self._thread = threading.Thread(target=self._watch_loop, daemon=True)
            self._thread.start()
            
            self.stats['start_time'] = datetime.now().isoformat()
            self.state = WatcherState.RUNNING
            
            logger.info("FileWatcherAgent 启动成功")
            self._notify({
                'type': 'watcher_status',
                'state': 'running',
                'watch_dir': str(self.watch_dir)
            })
            return True
            
        except Exception as e:
            self.state = WatcherState.ERROR
            logger.error(f"FileWatcherAgent 启动失败: {e}")
            return False

    def stop(self):
        """停止文件监控"""
        if self.state != WatcherState.RUNNING:
            return
            
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
            
        self.state = WatcherState.STOPPED
        logger.info("FileWatcherAgent 已停止")
        self._notify({
            'type': 'watcher_status',
            'state': 'stopped'
        })

    def pause(self):
        """暂停监控"""
        if self.state == WatcherState.RUNNING:
            self._pause_event.clear()
            self.state = WatcherState.PAUSED
            logger.info("FileWatcherAgent 已暂停")
            self._notify({'type': 'watcher_status', 'state': 'paused'})

    def resume(self):
        """恢复监控"""
        if self.state == WatcherState.PAUSED:
            self._pause_event.set()
            self.state = WatcherState.RUNNING
            logger.info("FileWatcherAgent 已恢复")
            self._notify({'type': 'watcher_status', 'state': 'running'})

    def _watch_loop(self):
        """监控循环"""
        last_scan_time = {}
        
        while not self._stop_event.is_set():
            try:
                # 暂停检查
                self._pause_event.wait(timeout=1)
                if not self._pause_event.is_set():
                    continue
                    
                # 扫描目录
                current_files = self._scan_directory()
                
                for file_info in current_files:
                    file_path = file_info['path']
                    
                    # 跳过已处理的文件
                    if file_path in self.processed_files:
                        continue
                        
                    # 检查文件是否稳定（新文件写入可能需要时间）
                    if not self._is_file_stable(file_path):
                        continue
                    
                    # 发现新文件
                    self._on_new_file(file_info)
                    
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                self.stats['total_errors'] += 1
                
            time.sleep(2)  # 每2秒扫描一次

    def _scan_directory(self) -> List[Dict]:
        """扫描目录，返回新文件列表"""
        files = []
        try:
            for item in self.watch_dir.iterdir():
                if item.is_file() and self._is_supported_file(item):
                    stat = item.stat()
                    files.append({
                        'path': str(item),
                        'name': item.name,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'ext': item.suffix.lower()
                    })
        except Exception as e:
            logger.error(f"目录扫描失败: {e}")
        return files

    def _is_supported_file(self, path: Path) -> bool:
        """检查是否支持的文件格式"""
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def _is_file_stable(self, file_path: str, min_age: int = 2) -> bool:
        """
        检查文件是否稳定（已完全写入）
        文件在min_age秒内没有变化认为已稳定
        """
        try:
            stat = Path(file_path).stat()
            age = time.time() - stat.st_mtime
            return age > min_age
        except:
            return False

    def _on_new_file(self, file_info: Dict):
        """发现新文件时的处理"""
        file_path = file_info['path']
        
        logger.info(f"发现新文件: {file_info['name']}")
        self.stats['total_detected'] += 1
        
        # 添加入待处理队列
        pending_entry = {
            'id': f"file_{int(time.time() * 1000)}",
            'file_info': file_info,
            'detected_at': datetime.now().isoformat(),
            'status': 'pending'
        }
        self.pending_files.append(pending_entry)
        
        # 发送通知
        self._notify({
            'type': 'new_file_detected',
            'file': file_info,
            'pending_count': len(self.pending_files)
        })
        
        # 触发分析
        if self.analyzer_callback:
            self._process_file(pending_entry)

    def _process_file(self, pending_entry: Dict):
        """处理文件"""
        file_info = pending_entry['file_info']
        file_path = file_info['path']
        
        pending_entry['status'] = 'processing'
        pending_entry['process_started_at'] = datetime.now().isoformat()
        
        self._notify({
            'type': 'processing_started',
            'file_id': pending_entry['id'],
            'file': file_info
        })
        
        try:
            if self.analyzer_callback:
                result = self.analyzer_callback(file_path, file_info['name'])
                pending_entry['status'] = 'completed'
                pending_entry['completed_at'] = datetime.now().isoformat()
                pending_entry['result'] = result
                
                self.stats['total_processed'] += 1
                self.stats['last_processed'] = datetime.now().isoformat()
                
                self._notify({
                    'type': 'processing_completed',
                    'file_id': pending_entry['id'],
                    'file': file_info,
                    'result': result
                })
            else:
                pending_entry['status'] = 'no_callback'
                
        except Exception as e:
            pending_entry['status'] = 'error'
            pending_entry['error'] = str(e)
            self.stats['total_errors'] += 1
            
            logger.error(f"文件处理失败: {file_info['name']}, 错误: {e}")
            self._notify({
                'type': 'processing_error',
                'file_id': pending_entry['id'],
                'file': file_info,
                'error': str(e)
            })
            
        finally:
            # 标记为已处理
            self.processed_files.add(file_info['path'])
            
            # 从待处理队列移除
            if pending_entry in self.pending_files:
                self.pending_files.remove(pending_entry)

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
        """获取监控状态"""
        return {
            'state': self.state.value,
            'watch_dir': str(self.watch_dir),
            'stats': self.stats,
            'pending_count': len(self.pending_files),
            'processed_count': len(self.processed_files)
        }

    def get_pending_files(self) -> List[Dict]:
        """获取待处理文件列表"""
        return self.pending_files.copy()

    def reprocess_file(self, file_path: str) -> bool:
        """重新处理指定文件"""
        if file_path in self.processed_files:
            self.processed_files.discard(file_path)
            
        file_info = {
            'path': file_path,
            'name': Path(file_path).name,
            'size': Path(file_path).stat().st_size if Path(file_path).exists() else 0,
            'modified': datetime.now().isoformat(),
            'ext': Path(file_path).suffix.lower()
        }
        
        pending_entry = {
            'id': f"file_{int(time.time() * 1000)}",
            'file_info': file_info,
            'detected_at': datetime.now().isoformat(),
            'status': 'pending',
            'reprocessed': True
        }
        
        self._process_file(pending_entry)
        return True

    def clear_processed_history(self):
        """清空已处理记录（重新监控所有文件）"""
        self.processed_files.clear()
        logger.info("已清空处理历史")

    def set_watch_directory(self, new_dir: str) -> bool:
        """
        更改监控目录（需要重启监控）
        """
        if self.state == WatcherState.RUNNING:
            self.stop()
            
        self.watch_dir = Path(new_dir)
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self.processed_files.clear()
        
        logger.info(f"监控目录已更改: {self.watch_dir}")
        return True


# 便捷函数：创建默认实例
def create_file_watcher(watch_dir: str, analyzer_callback: Optional[Callable] = None) -> FileWatcherAgent:
    """创建文件监控器实例"""
    return FileWatcherAgent(watch_dir, analyzer_callback)
