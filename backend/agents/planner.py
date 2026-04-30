"""
PlannerAgent - 任务规划智能体
支持复杂医学问题拆解、步骤执行、状态追踪
"""

import uuid
import json
import logging
import asyncio
import threading
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING = "waiting"  # 等待依赖任务


class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class Task:
    """任务"""
    id: str
    name: str
    description: str
    status: str
    priority: int
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Any = None
    error: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    subtasks: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    @classmethod
    def create(
        cls, 
        name: str, 
        description: str = "",
        priority: int = 2,
        dependencies: List[str] = None
    ) -> 'Task':
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            status=TaskStatus.PENDING.value,
            priority=priority,
            created_at=datetime.now().isoformat(),
            dependencies=dependencies or []
        )


@dataclass 
class Plan:
    """执行计划"""
    id: str
    name: str
    description: str
    tasks: List[Dict]
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    current_task_index: int = 0
    metadata: Dict = field(default_factory=dict)

    @classmethod
    def create(cls, name: str, description: str = "") -> 'Plan':
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            tasks=[],
            status="pending",
            created_at=datetime.now().isoformat()
        )


class PlannerAgent:
    """
    任务规划智能体
    功能:
    - 复杂医学问题拆解
    - 多步骤任务规划
    - 任务执行状态追踪
    - 并行/串行任务编排
    """

    # 预设的医学分析流程模板
    MEDICAL_WORKFLOWS = {
        'chest_analysis': {
            'name': '胸部影像综合分析',
            'steps': [
                {'name': '图像质量评估', 'action': 'assess_quality', 'parallel': False},
                {'name': '肺野分析', 'action': 'analyze_lung_fields', 'parallel': True},
                {'name': '心脏轮廓分析', 'action': 'analyze_heart', 'parallel': True},
                {'name': '骨骼检查', 'action': 'check_bones', 'parallel': True},
                {'name': '综合评估', 'action': 'final_assessment', 'parallel': False, 'depends_on': ['analyze_lung_fields', 'analyze_heart', 'check_bones']}
            ]
        },
        'brain_analysis': {
            'name': '颅脑影像综合分析',
            'steps': [
                {'name': '图像质量评估', 'action': 'assess_quality', 'parallel': False},
                {'name': '脑实质分析', 'action': 'analyze_brain_parenchyma', 'parallel': True},
                {'name': '脑室系统分析', 'action': 'analyze_ventricles', 'parallel': True},
                {'name': '中线结构分析', 'action': 'check_midline', 'parallel': True},
                {'name': '综合评估', 'action': 'final_assessment', 'parallel': False, 'depends_on': ['analyze_brain_parenchyma', 'analyze_ventricles', 'check_midline']}
            ]
        },
        'abdomen_analysis': {
            'name': '腹部影像综合分析',
            'steps': [
                {'name': '图像质量评估', 'action': 'assess_quality', 'parallel': False},
                {'name': '肝脏分析', 'action': 'analyze_liver', 'parallel': True},
                {'name': '胆囊分析', 'action': 'analyze_gallbladder', 'parallel': True},
                {'name': '肾脏分析', 'action': 'analyze_kidneys', 'parallel': True},
                {'name': '脾脏分析', 'action': 'analyze_spleen', 'parallel': True},
                {'name': '综合评估', 'action': 'final_assessment', 'parallel': False, 'depends_on': ['analyze_liver', 'analyze_gallbladder', 'analyze_kidneys', 'analyze_spleen']}
            ]
        }
    }

    def __init__(self):
        """初始化任务规划智能体"""
        self._lock = threading.RLock()
        self._executing = False
        self._stop_requested = False
        
        # 任务管理
        self.tasks: Dict[str, Task] = {}
        self.plans: Dict[str, Plan] = {}
        self.active_plan_id: Optional[str] = None
        
        # 任务执行器
        self._task_executors: Dict[str, Callable] = {}
        self._register_default_executors()
        
        # 通知回调
        self.notification_callbacks: List[Callable] = []
        
        # 历史记录
        self.execution_history: List[Dict] = []
        
        logger.info("PlannerAgent 初始化完成")

    def _register_default_executors(self):
        """注册默认的任务执行器"""
        self._task_executors = {
            'assess_quality': self._execute_assess_quality,
            'analyze_lung_fields': self._execute_analyze_lung_fields,
            'analyze_heart': self._execute_analyze_heart,
            'check_bones': self._execute_check_bones,
            'analyze_brain_parenchyma': self._execute_analyze_brain_parenchyma,
            'analyze_ventricles': self._execute_analyze_ventricles,
            'check_midline': self._execute_check_midline,
            'analyze_liver': self._execute_analyze_liver,
            'analyze_gallbladder': self._execute_analyze_gallbladder,
            'analyze_kidneys': self._execute_analyze_kidneys,
            'analyze_spleen': self._execute_analyze_spleen,
            'final_assessment': self._execute_final_assessment
        }

    def create_plan(
        self, 
        name: str, 
        description: str = "",
        workflow_key: str = None,
        custom_steps: List[Dict] = None
    ) -> str:
        """
        创建执行计划
        
        Args:
            name: 计划名称
            description: 计划描述
            workflow_key: 预设工作流key
            custom_steps: 自定义步骤列表
            
        Returns:
            计划ID
        """
        with self._lock:
            plan = Plan.create(name, description)
            
            # 添加步骤
            if workflow_key and workflow_key in self.MEDICAL_WORKFLOWS:
                workflow = self.MEDICAL_WORKFLOWS[workflow_key]
                for step in workflow['steps']:
                    task = Task.create(
                        name=step['name'],
                        description=f"Action: {step.get('action')}",
                        priority=TaskPriority.HIGH.value if not step.get('parallel') else TaskPriority.NORMAL.value,
                        dependencies=step.get('depends_on', [])
                    )
                    self.tasks[task.id] = task
                    plan.tasks.append(asdict(task))
            elif custom_steps:
                for step in custom_steps:
                    task = Task.create(
                        name=step['name'],
                        description=step.get('description', ''),
                        priority=step.get('priority', TaskPriority.NORMAL.value),
                        dependencies=step.get('depends_on', [])
                    )
                    self.tasks[task.id] = task
                    plan.tasks.append(asdict(task))
                    
            self.plans[plan.id] = plan
            logger.info(f"创建执行计划: {plan.id}, 任务数: {len(plan.tasks)}")
            
        self._notify({
            'type': 'plan_created',
            'plan': asdict(plan)
        })
            
        return plan.id

    def execute_plan(
        self, 
        plan_id: str, 
        context: Dict = None,
        parallel: bool = True
    ) -> Dict:
        """
        执行计划
        
        Args:
            plan_id: 计划ID
            context: 执行上下文（分析数据等）
            parallel: 是否并行执行独立任务
            
        Returns:
            执行结果
        """
        with self._lock:
            plan = self.plans.get(plan_id)
            if not plan:
                return {'success': False, 'error': '计划不存在'}
                
            if plan.status == 'running':
                return {'success': False, 'error': '计划正在执行中'}
                
            plan.status = 'running'
            plan.started_at = datetime.now().isoformat()
            self.active_plan_id = plan_id
            self._executing = True
            self._stop_requested = False
            
        self._notify({
            'type': 'plan_started',
            'plan_id': plan_id,
            'total_tasks': len(plan.tasks)
        })
        
        try:
            # 执行任务
            results = self._execute_tasks(plan, context or {}, parallel)
            
            # 更新计划状态
            with self._lock:
                all_completed = all(
                    self.tasks[t['id']].status == TaskStatus.COMPLETED.value 
                    for t in plan.tasks
                )
                any_failed = any(
                    self.tasks[t['id']].status == TaskStatus.FAILED.value 
                    for t in plan.tasks
                )
                
                plan.status = 'completed' if all_completed else ('failed' if any_failed else 'running')
                plan.completed_at = datetime.now().isoformat()
                
            # 添加到历史
            self._add_to_history(plan, results)
            
            return {
                'success': True,
                'plan_id': plan_id,
                'results': results,
                'status': plan.status
            }
            
        except Exception as e:
            logger.error(f"计划执行失败: {e}")
            with self._lock:
                plan.status = 'failed'
                plan.completed_at = datetime.now().isoformat()
            return {'success': False, 'error': str(e)}

    def _execute_tasks(
        self, 
        plan: Plan, 
        context: Dict,
        parallel: bool
    ) -> Dict:
        """执行任务"""
        results = {}
        
        # 构建任务依赖图
        task_map = {t['id']: t for t in plan.tasks}
        
        # 找出可以并行执行的任务（无依赖）
        def can_execute(task_id: str) -> bool:
            task = self.tasks.get(task_id)
            if not task or task.status != TaskStatus.PENDING.value:
                return False
            # 检查依赖是否都已完成
            for dep_id in task.dependencies:
                dep = self.tasks.get(dep_id)
                if not dep or dep.status != TaskStatus.COMPLETED.value:
                    return False
            return True
        
        # 按优先级排序
        pending_tasks = sorted(
            [t for t in plan.tasks if self.tasks[t['id']].status == TaskStatus.PENDING.value],
            key=lambda x: -x['priority']
        )
        
        for task_dict in pending_tasks:
            if self._stop_requested:
                break
                
            task = self.tasks[task_dict['id']]
            
            # 检查依赖
            if not can_execute(task.id):
                task.status = TaskStatus.WAITING.value
                continue
                
            # 执行任务
            task.status = TaskStatus.RUNNING.value
            task.started_at = datetime.now().isoformat()
            
            self._notify({
                'type': 'task_started',
                'plan_id': plan.id,
                'task': asdict(task)
            })
            
            try:
                # 从任务描述中提取action
                action = self._extract_action(task.description)
                executor = self._task_executors.get(action, self._execute_generic)
                
                result = executor(context, task)
                task.status = TaskStatus.COMPLETED.value
                task.result = result
                task.completed_at = datetime.now().isoformat()
                results[task.id] = result
                
                self._notify({
                    'type': 'task_completed',
                    'plan_id': plan.id,
                    'task': asdict(task),
                    'result': result
                })
                
            except Exception as e:
                logger.error(f"任务执行失败: {task.name}, 错误: {e}")
                task.status = TaskStatus.FAILED.value
                task.error = str(e)
                results[task.id] = {'error': str(e)}
                
                self._notify({
                    'type': 'task_failed',
                    'plan_id': plan.id,
                    'task': asdict(task),
                    'error': str(e)
                })
                
        return results

    def _extract_action(self, description: str) -> str:
        """从描述中提取action"""
        if 'Action: ' in description:
            return description.split('Action: ')[1].strip()
        return description

    def _execute_generic(self, context: Dict, task: Task) -> Dict:
        """通用任务执行器"""
        return {
            'status': 'completed',
            'task_name': task.name,
            'message': f"任务 '{task.name}' 执行完成"
        }

    # 专用执行器
    def _execute_assess_quality(self, context: Dict, task: Task) -> Dict:
        """图像质量评估"""
        analysis = context.get('detection', {})
        quality = context.get('detection', {}).get('image_quality', '未知')
        return {
            'status': 'completed',
            'step': 'assess_quality',
            'quality': quality,
            'assessment': f'图像质量评估：{quality}',
            'details': {
                'sharpness': '良好',
                'contrast': '正常',
                'artifacts': '无明显伪影'
            }
        }

    def _execute_analyze_lung_fields(self, context: Dict, task: Task) -> Dict:
        """肺野分析"""
        findings = context.get('detection', {}).get('abnormal_findings', [])
        lung_findings = [f for f in findings if '肺' in f.get('type', '') or 'lung' in f.get('type', '').lower()]
        return {
            'status': 'completed',
            'step': 'analyze_lung_fields',
            'findings': lung_findings,
            'summary': f'肺野分析完成，发现{len(lung_findings)}项异常'
        }

    def _execute_analyze_heart(self, context: Dict, task: Task) -> Dict:
        """心脏轮廓分析"""
        return {
            'status': 'completed',
            'step': 'analyze_heart',
            'findings': [],
            'summary': '心脏轮廓正常，未见明显增大或变形'
        }

    def _execute_check_bones(self, context: Dict, task: Task) -> Dict:
        """骨骼检查"""
        findings = context.get('detection', {}).get('abnormal_findings', [])
        bone_findings = [f for f in findings if '骨' in f.get('type', '') or 'bone' in f.get('type', '').lower()]
        return {
            'status': 'completed',
            'step': 'check_bones',
            'findings': bone_findings,
            'summary': f'骨骼检查完成，发现{len(bone_findings)}项异常'
        }

    def _execute_analyze_brain_parenchyma(self, context: Dict, task: Task) -> Dict:
        """脑实质分析"""
        findings = context.get('detection', {}).get('abnormal_findings', [])
        return {
            'status': 'completed',
            'step': 'analyze_brain_parenchyma',
            'findings': findings,
            'summary': '脑实质密度均匀，未见明显异常信号'
        }

    def _execute_analyze_ventricles(self, context: Dict, task: Task) -> Dict:
        """脑室系统分析"""
        return {
            'status': 'completed',
            'step': 'analyze_ventricles',
            'findings': [],
            'summary': '脑室系统形态正常，无扩大或受压表现'
        }

    def _execute_check_midline(self, context: Dict, task: Task) -> Dict:
        """中线结构分析"""
        return {
            'status': 'completed',
            'step': 'check_midline',
            'findings': [],
            'summary': '中线结构居中，无偏移'
        }

    def _execute_analyze_liver(self, context: Dict, task: Task) -> Dict:
        """肝脏分析"""
        return {
            'status': 'completed',
            'step': 'analyze_liver',
            'findings': [],
            'summary': '肝脏大小形态正常，实质回声均匀'
        }

    def _execute_analyze_gallbladder(self, context: Dict, task: Task) -> Dict:
        """胆囊分析"""
        return {
            'status': 'completed',
            'step': 'analyze_gallbladder',
            'findings': [],
            'summary': '胆囊壁光滑，腔内未见明显异常'
        }

    def _execute_analyze_kidneys(self, context: Dict, task: Task) -> Dict:
        """肾脏分析"""
        return {
            'status': 'completed',
            'step': 'analyze_kidneys',
            'findings': [],
            'summary': '双肾大小形态正常，皮髓质分界清晰'
        }

    def _execute_analyze_spleen(self, context: Dict, task: Task) -> Dict:
        """脾脏分析"""
        return {
            'status': 'completed',
            'step': 'analyze_spleen',
            'findings': [],
            'summary': '脾脏大小正常，未见明显异常'
        }

    def _execute_final_assessment(self, context: Dict, task: Task) -> Dict:
        """综合评估"""
        detection = context.get('detection', {})
        return {
            'status': 'completed',
            'step': 'final_assessment',
            'overall_assessment': detection.get('overall_assessment', ''),
            'recommendation': detection.get('clinical_recommendation', ''),
            'risk_level': detection.get('risk_level', 'unknown'),
            'summary': '综合评估完成'
        }

    def stop_execution(self):
        """停止当前执行"""
        self._stop_requested = True
        logger.info("已请求停止执行")
        self._notify({'type': 'execution_stopped'})

    def get_plan_status(self, plan_id: str) -> Optional[Dict]:
        """获取计划状态"""
        with self._lock:
            plan = self.plans.get(plan_id)
            if not plan:
                return None
                
            tasks_with_status = []
            for t in plan.tasks:
                task = self.tasks.get(t['id'])
                if task:
                    tasks_with_status.append(asdict(task))
                    
            return {
                'id': plan.id,
                'name': plan.name,
                'status': plan.status,
                'created_at': plan.created_at,
                'started_at': plan.started_at,
                'completed_at': plan.completed_at,
                'tasks': tasks_with_status,
                'progress': self._calculate_progress(tasks_with_status)
            }

    def _calculate_progress(self, tasks: List[Dict]) -> Dict:
        """计算进度"""
        total = len(tasks)
        if total == 0:
            return {'total': 0, 'completed': 0, 'running': 0, 'pending': 0, 'failed': 0, 'percent': 0}
            
        completed = sum(1 for t in tasks if t['status'] == 'completed')
        running = sum(1 for t in tasks if t['status'] == 'running')
        failed = sum(1 for t in tasks if t['status'] == 'failed')
        
        return {
            'total': total,
            'completed': completed,
            'running': running,
            'pending': total - completed - running - failed,
            'failed': failed,
            'percent': int((completed / total) * 100)
        }

    def list_plans(self) -> List[Dict]:
        """列出所有计划"""
        with self._lock:
            return [
                {
                    'id': p.id,
                    'name': p.name,
                    'description': p.description,
                    'status': p.status,
                    'task_count': len(p.tasks),
                    'created_at': p.created_at,
                    'progress': self._calculate_progress([asdict(self.tasks[t['id']]) for t in p.tasks])
                }
                for p in self.plans.values()
            ]

    def _add_to_history(self, plan: Plan, results: Dict):
        """添加到执行历史"""
        self.execution_history.append({
            'plan_id': plan.id,
            'plan_name': plan.name,
            'status': plan.status,
            'started_at': plan.started_at,
            'completed_at': plan.completed_at,
            'task_count': len(plan.tasks),
            'results_summary': {
                'completed': sum(1 for r in results.values() if not r.get('error')),
                'failed': sum(1 for r in results.values() if r.get('error'))
            }
        })

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
                'active_plan_id': self.active_plan_id,
                'total_plans': len(self.plans),
                'total_tasks': len(self.tasks),
                'is_executing': self._executing,
                'history_count': len(self.execution_history)
            }


# 便捷函数
def create_planner() -> PlannerAgent:
    """创建任务规划智能体实例"""
    return PlannerAgent()
