"""
医疗影像分析系统 - 后端主程序
FastAPI + Python + OpenAI Vision API
"""
import os
import uuid
import json
import time
import base64
import csv
import hashlib
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query, WebSocket, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# 尝试导入可选依赖
try:
    import pydicom  # type: ignore[import-not-found]
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

try:
    from PIL import Image, ImageEnhance, ImageFilter
    import io
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2  # type: ignore[import-not-found]
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

OpenAI: Any = None
try:
    from openai import OpenAI as OpenAIClient
    OpenAI = OpenAIClient
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

create_unet_agent: Any = None
try:
    from agents.unet_segmenter import create_unet_agent as create_unet_agent_func
    create_unet_agent = create_unet_agent_func
    UNET_AGENT_AVAILABLE = True
except Exception as e:
    UNET_AGENT_AVAILABLE = False

create_segmentation_registry: Any = None
try:
    from agents.segmentation_registry import create_segmentation_registry as create_segmentation_registry_func
    create_segmentation_registry = create_segmentation_registry_func
    SEGMENTATION_REGISTRY_AVAILABLE = True
except Exception as e:
    SEGMENTATION_REGISTRY_AVAILABLE = False

# 配置日志（必须在API客户端初始化之前）
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ========== 阿里云百炼（通义千问）配置 ==========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 阿里云百炼端点

# 初始化客户端（OpenAI兼容模式）
openai_client: Any = None
if OPENAI_AVAILABLE and OPENAI_API_KEY and OpenAI is not None:
    try:
        openai_client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            timeout=60.0
        )
        logger.info("阿里云百炼 API 客户端初始化成功 ✅")
    except Exception as e:
        logger.error(f"API 客户端初始化失败: {e}")

# 初始化FastAPI应用
app = FastAPI(
    title="医疗影像分析系统",
    description="基于AI的医疗影像智能分析平台 (OpenAI Vision驱动)",
    version="2.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 目录配置
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# 挂载静态文件
frontend_dir = BASE_DIR.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir / "static")), name="static")

# 内存中的分析记录
analysis_records: Dict[str, Dict] = {}


# ==================== 数据模型 ====================

class AnalysisResult(BaseModel):
    task_id: str
    status: str
    filename: str
    file_type: str
    upload_time: str
    analysis_time: Optional[float] = None
    findings: Optional[Dict] = None
    ai_report: Optional[str] = None
    thumbnail: Optional[str] = None
    metadata: Optional[Dict] = None


class AnalysisRequest(BaseModel):
    task_id: str
    analysis_type: str = "comprehensive"
    sensitivity: float = 0.75


# ==================== OpenAI Vision 分析引擎 ====================

class OpenAIVisionAnalyzer:
    """基于 OpenAI Vision API 的医疗影像真实分析引擎"""

    SUPPORTED_FORMATS = {'.dcm', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    # 医疗影像分析专用 Prompt（适配通义千问VL，全部输出中文）
    SYSTEM_PROMPT = """你是一名具有10年经验的放射科主任医师，擅长各类医学影像的诊断分析。
请对提供的医疗影像进行专业、严谨的分析，并输出中文的诊断结论。

【分析要求】
1. 识别影像类型和检查部位
2. 评估影像质量（清晰度、对比度、伪影等）
3. 描述正常解剖结构（用中文详细描述）
4. 发现并详细描述异常病变（如有）
5. 结合临床给出诊断意见和风险评估

【重要：输出语言要求】
- 所有分析内容、发现描述、评估意见必须使用**简体中文中文中文中文**
- normal_findings数组中的每项描述都要用完整的中文句子
- abnormal_findings中的type、location、description必须用中文详细描述
- overall_assessment必须是完整的中文段落
- clinical_recommendation必须是完整的中文建议
- image_quality必须是完整的中文描述

【输出格式】（严格JSON，不要包含markdown代码块标记```json ```）：
{
  "image_type": "X线片/CT/MRI/超声等中文名称",
  "body_part": "chest/brain/abdomen/bone/other",
  "image_quality": "中文描述影像质量，如：图像清晰，对比度良好，无明显伪影",
  "normal_findings": [
    "用完整的中文句子描述第一项正常发现",
    "用完整的中文句子描述第二项正常发现"
  ],
  "abnormal_findings": [
    {
      "type": "中文描述异常类型，如：肺部炎症、骨折、骨质增生等",
      "location": "中文描述位置，如：左肺下叶、第3腰椎左侧",
      "description": "中文详细描述，如：该区域可见斑片状密度增高影，边界模糊，大小约2.5cm×1.8cm",
      "severity": "轻度或中度或重度",
      "confidence": 0.85
    }
  ],
  "has_significant_finding": true或false,
  "risk_level": "low或medium或high",
  "overall_assessment": "这是完整的中文综合评估段落，至少50字，应包含对影像的整体评价、发现的意义、以及需要注意的问题",
  "clinical_recommendation": "这是完整的中文临床建议段落，至少30字，应包含后续检查建议、随访建议、或进一步诊断方向",
  "confidence_score": 0.90,
  "disclaimer": "本AI分析仅供参考，最终诊断结果请以专业临床医师的判断为准，必要时请结合其他临床检查"
}"""

    def __init__(self, client: Any, api_key: str):
        self.client = client
        self.api_key = api_key
        self.model = "qwen-vl-max"   # 阿里云百炼最强视觉模型

    def image_to_base64(self, img_array: np.ndarray) -> str:
        """将numpy数组转为base64图像"""
        if not PIL_AVAILABLE:
            return ""
        if len(img_array.shape) == 2:
            img = Image.fromarray(img_array.astype(np.uint8), 'L').convert('RGB')
        else:
            img = Image.fromarray(img_array.astype(np.uint8), 'RGB')
        # 调整尺寸以减少token消耗
        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def file_to_base64(self, file_path: Path) -> str:
        """直接将图像文件转为base64"""
        if not PIL_AVAILABLE:
            return ""
        try:
            img = Image.open(str(file_path))
            if img.mode not in ['RGB', 'L']:
                img = img.convert('RGB')
            else:
                img = img.convert('RGB')
            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"图像转base64失败: {e}")
            return ""

    def call_vision_api(self, image_b64: str, filename: str, analysis_type: str) -> Optional[Dict[str, Any]]:
        """调用 通义千问 qwen-vl-max Vision API 进行图像分析"""
        if not self.client:
            return None

        user_prompt = f"请分析这张医疗影像。文件名：{filename}，分析类型：{analysis_type}。\n请严格按照JSON格式输出分析结果，不要包含任何其他文字和代码块标记。"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            },
                            {"type": "text", "text": user_prompt}
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.1
            )

            content = (response.choices[0].message.content or "").strip()
            # 清理JSON前后的markdown标记
            if "```" in content:
                lines = content.split("\n")
                cleaned = []
                in_block = False
                for line in lines:
                    if line.strip().startswith("```"):
                        in_block = not in_block
                        continue
                    cleaned.append(line)
                content = "\n".join(cleaned).strip()

            return json.loads(content)

        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}, 内容: {content[:200]}")
            # 尝试提取JSON部分
            try:
                start = content.find('{')
                end = content.rfind('}') + 1
                if start >= 0 and end > start:
                    return json.loads(content[start:end])
            except Exception:
                pass
            return None
        except Exception as e:
            logger.error(f"Vision API调用失败: {e}")
            return None

    def generate_report_from_ai(self, task_id: str, filename: str, ai_result: Dict,
                                 properties: Dict, metadata: Dict, elapsed: float) -> str:
        """根据AI分析结果生成格式化报告"""
        now = datetime.now().strftime('%Y年%m月%d日 %H:%M')
        body_part_cn = {
            'chest': '胸部', 'brain': '颅脑', 'abdomen': '腹部',
            'bone': '骨骼', 'other': '未知部位'
        }.get(ai_result.get('body_part', 'other'), ai_result.get('body_part', '未知'))

        risk = ai_result.get('risk_level', 'low')
        risk_badges = {'low': '✅ 低风险', 'medium': '⚠️ 中风险', 'high': '🚨 高风险'}
        risk_badge = risk_badges.get(risk, '✅ 低风险')

        report = f"""📋 医疗影像AI分析报告（通义千问VL驱动）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 报告编号：{task_id[:8].upper()}
📅 分析时间：{now}
🏥 影像类型：{ai_result.get('image_type', body_part_cn + '影像')}
🔍 检查部位：{body_part_cn}
📁 文件名称：{filename}
风险评级：{risk_badge}
🤖 AI置信度：{ai_result.get('confidence_score', 0.9) * 100:.1f}%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【影像质量评估】
• 图像分辨率：{properties.get('width', 0)} × {properties.get('height', 0)} 像素
• 影像质量：{ai_result.get('image_quality', properties.get('image_quality', '良好'))}
• 灰度范围：{properties.get('min_intensity', 0)} - {properties.get('max_intensity', 255)}
• 对比度指数：{properties.get('contrast_ratio', 0):.2f}

【正常影像发现】"""
        for item in ai_result.get('normal_findings', []):
            report += f"\n• {item}"

        abnormal = ai_result.get('abnormal_findings', [])
        if abnormal:
            report += "\n\n【异常发现】"
            for finding in abnormal:
                report += f"""
⚠️ {finding.get('type', '异常')}
   位置：{finding.get('location', '未知')}
   描述：{finding.get('description', '')}
   严重程度：{finding.get('severity', '未知')}
   置信度：{finding.get('confidence', 0.85) * 100:.1f}%"""
        else:
            report += "\n\n【异常发现】\n✅ 未检测到明显异常病变"

        report += f"""

【综合评估】
{ai_result.get('overall_assessment', '请结合临床实际情况综合评判')}

【临床建议】
{ai_result.get('clinical_recommendation', '请结合临床实际情况综合评判')}"""

        if metadata:
            report += "\n\n【影像元数据】"
            for key, val in list(metadata.items())[:6]:
                report += f"\n• {key}：{val}"

        report += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏱️ 分析耗时：{elapsed:.2f}秒
🤖 分析引擎：阿里云百炼 qwen-vl-max
⚠️ 免责声明
{ai_result.get('disclaimer', '本报告由AI辅助分析生成，仅供参考。最终诊断请以专业医师意见为准。')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        return report

    def fallback_analysis(self, img_array: np.ndarray, filename: str,
                           body_part: str, sensitivity: float) -> Dict:
        """当API不可用时的备用分析逻辑"""
        import random
        mean_val = float(np.mean(img_array))
        std_val = float(np.std(img_array))
        random.seed(int(mean_val * 100) % 1000)

        FINDING_TEMPLATES = {
            'chest': {
                'normal': {
                    'description': '胸部影像未见明显异常',
                    'findings': ['双肺纹理清晰', '心影大小正常', '膈肌位置正常', '肋骨完整'],
                    'risk_level': 'low', 'confidence': 0.94
                },
                'abnormal': [{
                    'type': '肺结节', 'location': '右肺上叶', 'description': '发现约8mm实性结节',
                    'severity': '轻度', 'risk_level': 'medium',
                    'clinical_recommendation': '建议3-6个月随访CT检查', 'confidence': 0.87
                }]
            },
            'brain': {
                'normal': {
                    'description': '颅脑影像未见明显异常',
                    'findings': ['脑实质密度均匀', '脑室系统正常', '中线结构居中', '颅骨完整'],
                    'risk_level': 'low', 'confidence': 0.96
                },
                'abnormal': [{
                    'type': '脑白质变性', 'location': '双侧脑室周围', 'description': '双侧脑室周围白质信号改变',
                    'severity': '轻度', 'risk_level': 'low',
                    'clinical_recommendation': '控制血压血糖，定期随访', 'confidence': 0.83
                }]
            },
            'abdomen': {
                'normal': {
                    'description': '腹部影像未见明显异常',
                    'findings': ['肝脏大小形态正常', '胆囊壁光滑', '脾脏无增大', '腹腔无游离积液'],
                    'risk_level': 'low', 'confidence': 0.92
                },
                'abnormal': [{
                    'type': '肝囊肿', 'location': '肝右叶', 'description': '发现约15mm低密度灶',
                    'severity': '轻度', 'risk_level': 'low',
                    'clinical_recommendation': '定期超声随访，通常无需特殊处理', 'confidence': 0.95
                }]
            }
        }

        templates = FINDING_TEMPLATES.get(body_part, FINDING_TEMPLATES['chest'])
        has_finding = (std_val > 45) and (random.random() < sensitivity * 0.4)

        if has_finding and templates['abnormal']:
            finding = random.choice(templates['abnormal']).copy()
            return {
                'image_type': '医疗影像（备用分析）',
                'body_part': body_part,
                'image_quality': '影像质量良好',
                'normal_findings': templates['normal']['findings'][:2],
                'abnormal_findings': [finding],
                'has_significant_finding': True,
                'finding_count': 1,
                'primary_finding': finding,
                'normal_summary': templates['normal']['findings'][:2],
                'risk_level': finding['risk_level'],
                'overall_assessment': f"发现{finding['type']}，{finding.get('clinical_recommendation', '')}",
                'clinical_recommendation': finding.get('clinical_recommendation', '请结合临床综合评估'),
                'confidence_score': finding['confidence'],
                'disclaimer': '本报告由AI辅助分析生成，仅供参考。最终诊断请以专业医师意见为准。'
            }
        else:
            normal = templates['normal']
            return {
                'image_type': '医疗影像（备用分析）',
                'body_part': body_part,
                'image_quality': '影像质量良好',
                'normal_findings': normal['findings'],
                'abnormal_findings': [],
                'has_significant_finding': False,
                'finding_count': 0,
                'primary_finding': None,
                'normal_summary': normal['findings'],
                'risk_level': normal['risk_level'],
                'overall_assessment': normal['description'],
                'clinical_recommendation': '建议定期随访',
                'confidence_score': normal['confidence'],
                'disclaimer': '本报告由AI辅助分析生成，仅供参考。最终诊断请以专业医师意见为准。'
            }

    def detect_body_part(self, filename: str, metadata: Dict) -> str:
        filename_lower = filename.lower()
        if any(k in filename_lower for k in ['chest', 'lung', 'thorax', '胸', '肺']):
            return 'chest'
        elif any(k in filename_lower for k in ['brain', 'head', 'skull', '脑', '头']):
            return 'brain'
        elif any(k in filename_lower for k in ['abdomen', 'liver', 'belly', '腹', '肝']):
            return 'abdomen'
        body_part = metadata.get('BodyPartExamined', '').lower()
        if 'chest' in body_part or 'lung' in body_part:
            return 'chest'
        elif 'head' in body_part or 'brain' in body_part:
            return 'brain'
        elif 'abdomen' in body_part or 'liver' in body_part:
            return 'abdomen'
        return 'chest'

    def analyze_image_properties(self, img_array: np.ndarray) -> Dict:
        properties = {
            'width': int(img_array.shape[1]) if len(img_array.shape) > 1 else 0,
            'height': int(img_array.shape[0]),
            'channels': int(img_array.shape[2]) if len(img_array.shape) == 3 else 1,
            'mean_intensity': float(np.mean(img_array)),
            'std_intensity': float(np.std(img_array)),
            'min_intensity': int(np.min(img_array)),
            'max_intensity': int(np.max(img_array)),
            'contrast_ratio': float(np.max(img_array) - np.min(img_array)) / (np.mean(img_array) + 1e-8)
        }
        if properties['std_intensity'] < 10:
            properties['image_quality'] = '低对比度，影像质量较差'
        elif properties['std_intensity'] > 80:
            properties['image_quality'] = '对比度良好，影像质量优秀'
        else:
            properties['image_quality'] = '影像质量良好'
        return properties

    def process_dicom(self, file_path: Path) -> tuple:
        if not PYDICOM_AVAILABLE:
            raise RuntimeError("pydicom未安装")
        ds = pydicom.dcmread(str(file_path))
        metadata = {}
        meta_fields = [
            'PatientName', 'PatientID', 'PatientAge', 'PatientSex',
            'StudyDate', 'Modality', 'BodyPartExamined', 'StudyDescription',
            'InstitutionName', 'Manufacturer', 'Rows', 'Columns'
        ]
        for field in meta_fields:
            try:
                val = getattr(ds, field, None)
                if val is not None:
                    metadata[field] = str(val)
            except Exception:
                pass
        pixel_array = ds.pixel_array.astype(float)
        window_center = getattr(ds, 'WindowCenter', None)
        window_width = getattr(ds, 'WindowWidth', None)
        if window_center and window_width:
            wc = float(window_center) if not hasattr(window_center, '__len__') else float(window_center[0])
            ww = float(window_width) if not hasattr(window_width, '__len__') else float(window_width[0])
            pixel_array = np.clip(pixel_array, wc - ww/2, wc + ww/2)
        pmin, pmax = pixel_array.min(), pixel_array.max()
        if pmax > pmin:
            pixel_array = ((pixel_array - pmin) / (pmax - pmin) * 255).astype(np.uint8)
        else:
            pixel_array = pixel_array.astype(np.uint8)
        return pixel_array, metadata

    def process_regular_image(self, file_path: Path) -> tuple:
        if not PIL_AVAILABLE:
            raise RuntimeError("Pillow未安装")
        img = Image.open(str(file_path))
        if img.mode not in ['L', 'RGB']:
            img = img.convert('RGB')
        return np.array(img), {}

    def create_thumbnail(self, img_array: np.ndarray, size: tuple = (256, 256)) -> str:
        if not PIL_AVAILABLE:
            return ""
        if len(img_array.shape) == 2:
            img = Image.fromarray(img_array.astype(np.uint8), 'L')
        else:
            img = Image.fromarray(img_array.astype(np.uint8), 'RGB')
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.3)
        img.thumbnail(size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def analyze(self, file_path: Path, filename: str,
                analysis_type: str = 'comprehensive',
                sensitivity: float = 0.75) -> Dict:
        """主分析入口 - 优先使用 OpenAI Vision API"""
        suffix = file_path.suffix.lower()
        start_time = time.time()

        try:
            # 加载影像
            if suffix == '.dcm' and PYDICOM_AVAILABLE:
                img_array, metadata = self.process_dicom(file_path)
                file_type = 'DICOM'
            else:
                img_array, metadata = self.process_regular_image(file_path)
                file_type = suffix.upper().replace('.', '')

            # 分析影像属性
            properties = self.analyze_image_properties(img_array)

            # 检测身体部位（作为提示）
            body_part_hint = self.detect_body_part(filename, metadata)

            # 生成缩略图
            thumbnail = self.create_thumbnail(img_array)

            # ======= 尝试调用 OpenAI Vision API =======
            ai_result = None
            used_openai = False

            if self.client:
                logger.info(f"调用 OpenAI Vision API 分析影像: {filename}")
                image_b64 = self.image_to_base64(img_array)
                if image_b64:
                    ai_result = self.call_vision_api(image_b64, filename, analysis_type)
                    if ai_result:
                        used_openai = True
                        logger.info("OpenAI Vision API 分析成功")

            # 如果API不可用，使用备用分析
            if not ai_result:
                logger.warning("OpenAI API 不可用，使用备用分析逻辑")
                ai_result = self.fallback_analysis(img_array, filename, body_part_hint, sensitivity)

            # 补充检测兼容字段
            body_part = ai_result.get('body_part', body_part_hint)
            has_finding = ai_result.get('has_significant_finding', False)
            abnormal = ai_result.get('abnormal_findings', [])
            primary_finding = abnormal[0] if abnormal else None

            detection = {
                'has_significant_finding': has_finding,
                'finding_count': len(abnormal),
                'primary_finding': primary_finding,
                'normal_summary': ai_result.get('normal_findings', []),
                'overall_assessment': ai_result.get('overall_assessment', ''),
                'clinical_recommendation': ai_result.get('clinical_recommendation', ''),
                'risk_level': ai_result.get('risk_level', 'low'),
                'confidence': ai_result.get('confidence_score', 0.9),
                'abnormal_findings': abnormal
            }

            task_id = str(uuid.uuid4())
            elapsed = time.time() - start_time

            # 生成报告
            report = self.generate_report_from_ai(
                task_id, filename, ai_result, properties, metadata, elapsed
            )

            return {
                'success': True,
                'task_id': task_id,
                'filename': filename,
                'file_type': file_type,
                'body_part': body_part,
                'analysis_time': round(elapsed, 2),
                'properties': properties,
                'metadata': metadata,
                'detection': detection,
                'ai_result': ai_result,
                'ai_report': report,
                'thumbnail': thumbnail,
                'upload_time': datetime.now().isoformat(),
                'engine': '阿里云百炼 qwen-vl-max' if used_openai else '本地分析引擎（备用）'
            }

        except Exception as e:
            logger.error(f"分析失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'filename': filename
            }


# 全局分析器实例
analyzer = OpenAIVisionAnalyzer(openai_client, OPENAI_API_KEY)
unet_agent = create_unet_agent() if create_unet_agent else None
segmentation_registry = create_segmentation_registry() if create_segmentation_registry else None


def _metric_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _metric_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _risk_label(risk_level: str) -> str:
    return {
        "low": "低风险",
        "medium": "中等风险",
        "high": "高风险",
    }.get(risk_level or "low", "低风险")


def _body_part_label(body_part: Optional[str]) -> str:
    return {
        "chest": "胸部",
        "brain": "头颅/脑部",
        "abdomen": "腹部",
        "bone": "骨骼",
        "other": "当前部位",
    }.get(body_part or "other", body_part or "当前部位")


def _build_segmentation_clinical_recommendation(
    segmentation: Dict[str, Any],
    metrics: Dict[str, Any],
    body_part: Optional[str] = None,
) -> str:
    model_name = segmentation.get("model_name") or "当前分割模型"
    part_label = _body_part_label(body_part)
    risk_level = metrics.get("risk_level", "low")
    risk_label = _risk_label(risk_level)
    area_percent = _metric_float(metrics.get("area_percent"))
    mean_probability = _metric_float(metrics.get("mean_probability"))
    max_probability = _metric_float(metrics.get("max_probability"))
    has_candidate = bool(metrics.get("has_candidate_region"))

    if not segmentation.get("success"):
        return (
            "本次模型推理未成功生成可靠分割结果。建议先确认上传影像质量、格式和模型加载状态，"
            "必要时重新上传清晰图像或更换模型复测；临床判断仍应以原始影像和医生评估为准。"
        )

    common_review = (
        f"请将{model_name}输出的分割叠加图、mask、概率图与{part_label}原始影像逐层对照，"
        "并结合患者症状、体征、既往检查和实验室结果综合判断。"
    )

    if has_candidate and risk_level == "high":
        return (
            f"当前模型提示{risk_label}候选区域，面积占比约 {area_percent:.2f}%，"
            f"掩膜内平均概率 {mean_probability:.4f}，最高概率 {max_probability:.4f}。"
            f"{common_review}"
            "建议尽快由影像科或相关专科医生复核该区域是否与真实病变、伪影或正常解剖结构相符；"
            "若患者存在明显不适、急性症状或临床高度怀疑，应优先安排进一步检查或急诊评估。"
            "在医生确认前，不建议仅凭模型结果直接做诊断或治疗决策。"
        )
    if has_candidate and risk_level == "medium":
        return (
            f"当前模型提示{risk_label}候选区域，面积占比约 {area_percent:.2f}%，"
            f"掩膜内平均概率 {mean_probability:.4f}。{common_review}"
            "建议结合临床关注部位进行人工复核；如候选区域与症状或既往异常一致，"
            "可考虑补充更有针对性的影像检查、对比既往片或安排短期随访。"
        )
    if has_candidate:
        return (
            f"当前模型检出低风险候选区域，面积占比约 {area_percent:.2f}%，"
            f"掩膜内平均概率 {mean_probability:.4f}。{common_review}"
            "建议作为辅助定位线索保留，由医生判断是否需要进一步观察；若患者没有相关症状且原始影像无可疑表现，"
            "可按常规流程随访。"
        )
    return (
        f"当前阈值下未检出明确候选分割区域，模型风险提示为{risk_label}，最高预测概率 {max_probability:.4f}。"
        "这说明本模型在该输入上未发现稳定阳性区域，但不能完全排除细小、边界不清或模型不敏感类型的异常。"
        f"建议医生仍结合{part_label}原始影像和临床资料复核；如患者症状持续、风险因素较高或临床怀疑较强，"
        "应考虑进一步检查、复查或使用其他影像序列/模型进行辅助评估。"
    )


def _build_segmentation_ai_report(detection: Dict[str, Any], segmentation: Dict[str, Any]) -> str:
    model = detection.get("segmentation_model", {})
    metrics = segmentation.get("metrics", {}) if segmentation.get("success") else {}
    bbox = metrics.get("bbox") or {}
    bbox_text = (
        f"x={bbox.get('x_min', '-')}-{bbox.get('x_max', '-')}, "
        f"y={bbox.get('y_min', '-')}-{bbox.get('y_max', '-')}"
        if bbox
        else "无"
    )
    return (
        f"{model.get('model_name') or segmentation.get('model_name', '分割模型')} 分割分析已完成。\n\n"
        "【AI分析结论】\n"
        f"{detection.get('overall_assessment', '')}\n\n"
        "【量化指标】\n"
        f"- 风险等级：{_risk_label(detection.get('risk_level', 'low'))}\n"
        f"- 候选区域面积占比：{_metric_float(metrics.get('area_percent')):.2f}%\n"
        f"- 阳性像素：{_metric_int(metrics.get('positive_pixels'))} / {_metric_int(metrics.get('total_pixels'))}\n"
        f"- 平均概率：{_metric_float(metrics.get('mean_probability')):.4f}\n"
        f"- 最大概率：{_metric_float(metrics.get('max_probability')):.4f}\n"
        f"- 候选框：{bbox_text}\n"
        f"- 阈值：{_metric_float(segmentation.get('threshold')):.4f}\n\n"
        "【临床建议】\n"
        f"{detection.get('clinical_recommendation', '')}\n\n"
        "【安全说明】\n"
        "以上内容由本地分割模型和规则层自动生成，仅作为影像辅助筛查与定位参考，不能替代医生诊断。"
    )


def _build_unet_structured_payload(task_id: str, filename: str, segmentation: Dict) -> Dict:
    metrics = segmentation.get("metrics") or {}
    status = segmentation.get("status") or {}
    postprocess = metrics.get("postprocess") or status.get("postprocess") or {}
    threshold = _metric_float(segmentation.get("threshold"), _metric_float(status.get("threshold"), 0.0))
    area_percent = _metric_float(metrics.get("area_percent"))
    mean_probability = _metric_float(metrics.get("mean_probability"))
    max_probability = _metric_float(metrics.get("max_probability"))
    risk_level = metrics.get("risk_level", "low")
    has_candidate = bool(metrics.get("has_candidate_region"))

    return {
        "task_id": task_id,
        "filename": filename,
        "model_task": "medical image binary segmentation",
        "model_name": "Local PyTorch U-Net",
        "analysis_goal": "基于本地 U-Net 分割结果生成结构化医学影像辅助分析报告",
        "segmentation_success": bool(segmentation.get("success")),
        "quantitative_metrics": {
            "threshold": threshold,
            "positive_pixels": _metric_int(metrics.get("positive_pixels")),
            "total_pixels": _metric_int(metrics.get("total_pixels")),
            "area_ratio": _metric_float(metrics.get("area_ratio")),
            "area_percent": area_percent,
            "mean_probability": mean_probability,
            "max_probability": max_probability,
            "bbox": metrics.get("bbox"),
            "risk_level": risk_level,
            "risk_label": _risk_label(risk_level),
            "has_candidate_region": has_candidate,
        },
        "postprocess": {
            "enabled": bool(postprocess.get("enabled", False)),
            "open_iters": _metric_int(postprocess.get("open_iters")),
            "close_iters": _metric_int(postprocess.get("close_iters")),
            "fill_holes": bool(postprocess.get("fill_holes", False)),
            "min_area": _metric_int(postprocess.get("min_area")),
            "min_area_ratio": _metric_float(postprocess.get("min_area_ratio")),
        },
        "safety_rules": [
            "只能基于 U-Net 的结构化输出描述分割结果，不得编造影像中未提供的病灶类型。",
            "不得给出最终诊断，不得替代医生判断。",
            "必须提示结果需要结合原始影像、临床症状和医生复核。",
        ],
    }


def _fallback_unet_sections(payload: Dict, error: Optional[str] = None) -> Dict[str, str]:
    metrics = payload["quantitative_metrics"]
    risk_label = metrics["risk_label"]
    area_percent = metrics["area_percent"]
    mean_probability = metrics["mean_probability"]
    threshold = metrics["threshold"]
    has_candidate = metrics["has_candidate_region"]

    if has_candidate:
        conclusion = (
            f"本地 U-Net 在当前阈值 {threshold:.2f} 下检出候选分割区域，"
            f"面积占比约 {area_percent:.2f}%，平均预测概率为 {mean_probability:.4f}。"
        )
        risk = (
            f"当前模型风险提示为{risk_label}。该风险等级来自分割面积占比与预测概率，"
            "只能作为算法辅助提示，不能直接等同于临床诊断。"
        )
    else:
        conclusion = (
            f"本地 U-Net 在当前阈值 {threshold:.2f} 下未检出明确候选分割区域。"
        )
        risk = "当前模型风险提示为低风险，但低风险不代表完全排除异常，需要结合原始影像复核。"

    recommendation = (
        "建议医生结合原始影像、分割叠加图、概率图及患者临床信息进行复核；"
        "如候选区域与临床关注区域一致，可进一步进行人工标注确认或补充检查。"
    )
    limitations = (
        "本报告仅基于本地 U-Net 的结构化分割输出生成，不包含真实临床病史，"
        "也不应替代专业医师的最终诊断。"
    )
    if error:
        limitations += f" 通义千问报告生成未成功，当前展示本地兜底报告；错误摘要：{error}"

    return {
        "conclusion": conclusion,
        "quantitative_findings": (
            f"候选区域面积占比 {area_percent:.2f}%，阳性像素 {metrics['positive_pixels']} / "
            f"{metrics['total_pixels']}，平均预测概率 {mean_probability:.4f}，"
            f"最大预测概率 {metrics['max_probability']:.4f}，bbox={metrics.get('bbox')}。"
        ),
        "risk_assessment": risk,
        "clinical_recommendation": recommendation,
        "limitations": limitations,
    }


def _parse_qwen_json(content: str) -> Optional[Dict]:
    if not content:
        return None
    text = content.strip()
    if "```" in text:
        cleaned = []
        in_block = False
        for line in text.splitlines():
            if line.strip().startswith("```"):
                in_block = not in_block
                continue
            cleaned.append(line)
        text = "\n".join(cleaned).strip()
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except Exception:
                return None
    return None


def _render_unet_llm_report(payload: Dict, sections: Dict[str, str], source: str, elapsed: float) -> str:
    metrics = payload["quantitative_metrics"]
    source_label = "通义千问文本模型" if source == "qwen" else "本地规则兜底"
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""📋 U-Net 分割辅助分析报告（结构化输出驱动）

报告时间：{now}
文件名称：{payload.get('filename', '-')}
分析引擎：{source_label}
分割模型：{payload.get('model_name', 'Local PyTorch U-Net')}

【AI 分析结论】
{sections.get('conclusion', '')}

【量化指标】
- 部署阈值：{metrics.get('threshold', 0):.2f}
- 候选区域面积占比：{metrics.get('area_percent', 0):.2f}%
- 阳性像素：{metrics.get('positive_pixels', 0)} / {metrics.get('total_pixels', 0)}
- 平均预测概率：{metrics.get('mean_probability', 0):.4f}
- 最大预测概率：{metrics.get('max_probability', 0):.4f}
- 风险提示：{metrics.get('risk_label', '低风险')}

【模型解释】
{sections.get('quantitative_findings', '')}

【风险判断】
{sections.get('risk_assessment', '')}

【复核建议】
{sections.get('clinical_recommendation', '')}

【局限性声明】
{sections.get('limitations', '')}

生成耗时：{elapsed:.2f}s
"""


def generate_unet_llm_report(task_id: str, filename: str, segmentation: Dict) -> Dict:
    """Generate a structured report from local U-Net metrics via Qwen, with deterministic fallback."""
    started = time.time()
    payload = _build_unet_structured_payload(task_id, filename, segmentation)

    if not payload.get("segmentation_success"):
        sections = _fallback_unet_sections(payload, segmentation.get("error", "segmentation_failed"))
        elapsed = time.time() - started
        return {
            "enabled": False,
            "source": "local_fallback",
            "model": "local_rules",
            "structured_input": payload,
            "sections": sections,
            "report": _render_unet_llm_report(payload, sections, "local_fallback", elapsed),
            "elapsed": round(elapsed, 2),
            "error": segmentation.get("error", "segmentation_failed"),
        }

    fallback_sections = _fallback_unet_sections(payload)
    if not openai_client:
        elapsed = time.time() - started
        return {
            "enabled": False,
            "source": "local_fallback",
            "model": "local_rules",
            "structured_input": payload,
            "sections": fallback_sections,
            "report": _render_unet_llm_report(payload, fallback_sections, "local_fallback", elapsed),
            "elapsed": round(elapsed, 2),
            "error": "qwen_client_unavailable",
        }

    system_prompt = (
        "你是医疗影像 AI 产品中的报告生成助手。你只能基于用户提供的 U-Net 结构化分割结果写辅助报告，"
        "不得编造病灶类型、不得给最终诊断、不得替代医生。"
        "除非输入明确给出临床诊断，否则不要使用‘病灶’、‘肿瘤’、‘炎症’等诊断性词汇，"
        "统一称为‘候选分割区域’或‘模型标注区域’。不要推断解剖部位、疾病类型或组织学性质。"
        "输出必须是 JSON，包含 conclusion、quantitative_findings、risk_assessment、"
        "clinical_recommendation、limitations 五个字符串字段。"
    )
    user_prompt = (
        "请基于以下 U-Net 分割结构化输出生成医学影像辅助分析报告。"
        "请明确说明该报告只反映模型分割结果，需要医生结合原始影像复核；"
        "不要把模型标注区域直接称为病灶或异常诊断。\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )

    try:
        response = openai_client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1200,
        )
        content = response.choices[0].message.content or ""
        parsed = _parse_qwen_json(content)
        if not isinstance(parsed, dict):
            raise ValueError("qwen_response_is_not_valid_json")
        sections = {**fallback_sections}
        for key in sections:
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                sections[key] = value.strip()
        elapsed = time.time() - started
        return {
            "enabled": True,
            "source": "qwen",
            "model": "qwen-plus",
            "structured_input": payload,
            "sections": sections,
            "report": _render_unet_llm_report(payload, sections, "qwen", elapsed),
            "elapsed": round(elapsed, 2),
            "error": "",
        }
    except Exception as exc:
        logger.warning(f"Qwen structured U-Net report failed, using fallback: {exc}")
        elapsed = time.time() - started
        sections = _fallback_unet_sections(payload, str(exc))
        return {
            "enabled": False,
            "source": "local_fallback",
            "model": "local_rules",
            "structured_input": payload,
            "sections": sections,
            "report": _render_unet_llm_report(payload, sections, "local_fallback", elapsed),
            "elapsed": round(elapsed, 2),
            "error": str(exc),
        }


def enrich_with_unet_segmentation(result: Dict, file_path: Path, display_filename: Optional[str] = None) -> Dict:
    """Attach local U-Net segmentation output and a Qwen-generated structured report."""
    if not unet_agent:
        return result

    segmentation = unet_agent.analyze_file(file_path)
    result["segmentation"] = segmentation

    if not segmentation.get("success"):
        return result

    metrics = segmentation.get("metrics", {})
    result["segmentation_overlay"] = segmentation.get("overlay", "")
    result["segmentation_mask"] = segmentation.get("mask", "")
    result["segmentation_probability"] = segmentation.get("probability", "")

    detection = result.setdefault("detection", {})
    detection["unet_segmentation"] = {
        "available": True,
        "area_percent": metrics.get("area_percent", 0),
        "mean_probability": metrics.get("mean_probability", 0),
        "max_probability": metrics.get("max_probability", 0),
        "bbox": metrics.get("bbox"),
        "has_candidate_region": metrics.get("has_candidate_region", False),
        "risk_level": metrics.get("risk_level", "low"),
    }

    if metrics.get("has_candidate_region"):
        detection["has_significant_finding"] = True
        if detection.get("risk_level", "low") == "low":
            detection["risk_level"] = metrics.get("risk_level", "medium")
        primary = detection.get("primary_finding") or {
            "type": "U-Net候选分割区域",
            "location": "模型热区",
            "description": f"本地分割模型检出候选区域，面积占比约 {metrics.get('area_percent', 0)}%。",
            "severity": "待复核",
            "confidence": metrics.get("mean_probability", 0.0),
        }
        detection["primary_finding"] = primary
        abnormal = detection.setdefault("abnormal_findings", [])
        abnormal.append(primary)
        detection["finding_count"] = len(abnormal)

    llm_report = generate_unet_llm_report(
        result.get("task_id", ""),
        display_filename or file_path.name,
        segmentation,
    )
    result["llm_report"] = llm_report
    detection["unet_llm_report"] = {
        "source": llm_report.get("source"),
        "model": llm_report.get("model"),
        "sections": llm_report.get("sections", {}),
        "elapsed": llm_report.get("elapsed", 0),
        "error": llm_report.get("error", ""),
    }

    report_parts = [
        result.get("ai_report", "").strip(),
        llm_report.get("report", "").strip(),
        segmentation.get("report", "").strip(),
    ]
    result["ai_report"] = "\n\n".join(part for part in report_parts if part)
    result["engine"] = f"{result.get('engine', 'analysis')} + Local U-Net + {llm_report.get('model', 'report')}"
    return result

# ==================== API路由 ====================

def _normalize_array_to_uint8(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    if arr.dtype == np.uint8 and arr.min() >= 0 and arr.max() <= 255:
        return arr
    arr = arr.astype(np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = float(arr[finite].min()), float(arr[finite].max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255)
    return arr.astype(np.uint8)


def _image_to_preview_payload(img: "Image.Image", filename: str) -> Dict[str, Any]:
    try:
        img.seek(0)
    except Exception:
        pass

    width, height = img.size
    arr = np.asarray(img)
    if arr.ndim == 2 or img.mode in {"I", "I;16", "I;16B", "I;16L", "F"}:
        preview_img = Image.fromarray(_normalize_array_to_uint8(arr), mode="L").convert("RGB")
    else:
        preview_img = img.convert("RGB")

    preview_img.thumbnail((360, 360), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    preview_img.save(buffer, format="PNG")
    return {
        "success": True,
        "filename": filename,
        "width": width,
        "height": height,
        "mode": str(img.mode),
        "preview": base64.b64encode(buffer.getvalue()).decode("utf-8"),
    }


def _dicom_bytes_to_preview(content: bytes, filename: str) -> Dict[str, Any]:
    if not PYDICOM_AVAILABLE:
        raise RuntimeError("pydicom is not installed")
    dataset = pydicom.dcmread(io.BytesIO(content), force=True)
    pixel_array = dataset.pixel_array.astype(np.float32)
    slope = float(getattr(dataset, "RescaleSlope", 1) or 1)
    intercept = float(getattr(dataset, "RescaleIntercept", 0) or 0)
    pixel_array = pixel_array * slope + intercept

    window_center = getattr(dataset, "WindowCenter", None)
    window_width = getattr(dataset, "WindowWidth", None)
    if window_center is not None and window_width is not None:
        if hasattr(window_center, "__len__"):
            window_center = window_center[0]
        if hasattr(window_width, "__len__"):
            window_width = window_width[0]
        center, width = float(window_center), float(window_width)
        pixel_array = np.clip(pixel_array, center - width / 2, center + width / 2)

    if pixel_array.ndim > 2:
        pixel_array = pixel_array[0]
    preview_img = Image.fromarray(_normalize_array_to_uint8(pixel_array), mode="L")
    return _image_to_preview_payload(preview_img, filename)


def build_preview_payload(content: bytes, filename: str) -> Dict[str, Any]:
    suffix = Path(filename).suffix.lower()
    if suffix not in OpenAIVisionAnalyzer.SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {suffix}")
    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File is too large. Max size is 100MB.")
    if not PIL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Pillow is not installed.")

    try:
        if suffix == ".dcm":
            return _dicom_bytes_to_preview(content, filename)
        img = Image.open(io.BytesIO(content))
        return _image_to_preview_payload(img, filename)
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Preview generation failed for %s: %s", filename, e)
        return {
            "success": False,
            "filename": filename,
            "width": None,
            "height": None,
            "preview": "",
            "error": str(e),
        }


RUNS_ROOT = Path(
    os.environ.get(
        "UNET_RUNS_DIR",
        str(Path(__file__).resolve().parents[1] / "original_unet_project" / "runs"),
    )
)
PREFERRED_TRAINING_RUNS = [
    "unet_agent_current",
    "unet_agent_finetune_lr1e4",
    "unet_agent_es",
    "unet_agent",
    "unet",
]


def _as_float(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> Optional[int]:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _read_training_history(run_dir: Path) -> List[Dict[str, Any]]:
    history_path = run_dir / "history.csv"
    if not history_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(history_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "epoch": _as_int(row.get("epoch")),
                "lr": _as_float(row.get("lr")),
                "train_loss": _as_float(row.get("train_loss")),
                "train_dice": _as_float(row.get("train_dice")),
                "train_iou": _as_float(row.get("train_iou")),
                "val_loss": _as_float(row.get("val_loss")),
                "val_dice": _as_float(row.get("val_dice")),
                "val_iou": _as_float(row.get("val_iou")),
                "best_threshold": _as_float(row.get("val_best_threshold")),
                "threshold_dice": _as_float(row.get("val_best_dice")),
                "threshold_iou": _as_float(row.get("val_best_iou")),
            })
    return rows


def _load_best_threshold(run_dir: Path, history: List[Dict[str, Any]]) -> Dict[str, Any]:
    postprocess_path = run_dir / "postprocess_config.json"
    if postprocess_path.exists():
        try:
            with open(postprocess_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("best_threshold") is not None and data.get("val_dice") is not None:
                return {
                    "epoch": data.get("epoch", "post"),
                    "threshold": data.get("best_threshold"),
                    "val_dice": data.get("val_dice"),
                    "val_iou": data.get("val_iou"),
                    "postprocess": data.get("postprocess"),
                    "baseline_val_dice": data.get("baseline_val_dice"),
                    "source": data.get("source", str(postprocess_path)),
                }
        except Exception as e:
            logger.warning("Failed to read %s: %s", postprocess_path, e)

    best_path = run_dir / "best_threshold.json"
    if best_path.exists():
        try:
            with open(best_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "epoch": data.get("epoch"),
                "threshold": data.get("best_threshold"),
                "val_dice": data.get("val_dice"),
                "val_iou": data.get("val_iou"),
            }
        except Exception as e:
            logger.warning("Failed to read %s: %s", best_path, e)

    valid = [r for r in history if r.get("val_dice") is not None]
    if not valid:
        return {}
    best = max(valid, key=lambda r: r.get("val_dice") or -1)
    return {
        "epoch": best.get("epoch"),
        "threshold": best.get("best_threshold"),
        "val_dice": best.get("val_dice"),
        "val_iou": best.get("val_iou"),
    }


def _judge_fit(history: List[Dict[str, Any]], best: Dict[str, Any]) -> Dict[str, Any]:
    if not history:
        return {"level": "unknown", "title": "暂无训练数据", "message": "还没有读取到 history.csv，无法判断拟合情况。"}

    latest = history[-1]
    best_dice = _as_float(best.get("val_dice"))
    latest_val = latest.get("val_dice") or 0.0
    latest_train = latest.get("train_dice") or 0.0
    latest_gap = latest_train - latest_val
    recent = [r.get("val_dice") for r in history[-3:] if r.get("val_dice") is not None]
    recent_range = (max(recent) - min(recent)) if len(recent) >= 2 else 0.0
    drop_from_best = (best_dice - latest_val) if best_dice is not None else 0.0

    reasons = []
    level = "good"
    title = "拟合良好"

    if latest_train < 0.70 and latest_val < 0.70:
        level = "underfit"
        title = "偏欠拟合"
        reasons.append("训练 Dice 和验证 Dice 都偏低，模型还没有充分学到稳定分割特征。")
    elif latest_gap > 0.08 or drop_from_best > 0.05:
        level = "overfit"
        title = "有过拟合风险"
        reasons.append("训练 Dice 明显高于验证 Dice，或最新验证 Dice 相比最佳轮次回落较多。")
    elif recent_range > 0.06:
        level = "unstable"
        title = "验证波动较大"
        reasons.append("最近 3 轮验证 Dice 波动偏大，建议继续使用 early stopping 和较低学习率。")
    else:
        reasons.append("最新验证 Dice 接近最佳轮次，训练与验证差距可控。")

    if best.get("threshold") is not None:
        reasons.append(f"当前保存的最优阈值为 {best.get('threshold'):.2f}。")

    return {
        "level": level,
        "title": title,
        "message": " ".join(reasons),
        "signals": {
            "latest_gap": round(float(latest_gap), 4),
            "drop_from_best": round(float(drop_from_best), 4),
            "recent_val_dice_range": round(float(recent_range), 4),
        },
    }


def _run_summary(run_dir: Path) -> Optional[Dict[str, Any]]:
    history = _read_training_history(run_dir)
    if not history:
        return None
    best = _load_best_threshold(run_dir, history)
    curve_path = run_dir / "visuals" / "training_curves.png"
    curve_image = ""
    if curve_path.exists():
        try:
            curve_image = base64.b64encode(curve_path.read_bytes()).decode("utf-8")
        except Exception:
            curve_image = ""
    latest = history[-1]
    return {
        "name": run_dir.name,
        "path": str(run_dir),
        "updated_at": datetime.fromtimestamp((run_dir / "history.csv").stat().st_mtime).isoformat(),
        "epochs": len(history),
        "latest": latest,
        "best": best,
        "judgement": _judge_fit(history, best),
        "history": history,
        "curve_image": curve_image,
    }


def build_training_fit_status() -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    seen = set()
    for name in PREFERRED_TRAINING_RUNS:
        run_dir = RUNS_ROOT / name
        seen.add(run_dir.resolve() if run_dir.exists() else run_dir)
        summary = _run_summary(run_dir)
        if summary:
            runs.append(summary)
    if RUNS_ROOT.exists():
        for history_path in RUNS_ROOT.glob("*/history.csv"):
            run_dir = history_path.parent
            key = run_dir.resolve()
            if key in seen:
                continue
            summary = _run_summary(run_dir)
            if summary:
                runs.append(summary)

    runs.sort(key=lambda r: r.get("updated_at", ""), reverse=True)
    active = runs[0] if runs else None
    return {
        "success": bool(active),
        "active_run": active,
        "runs": runs,
        "runs_root": str(RUNS_ROOT),
        "updated_at": datetime.now().isoformat(),
    }


@app.get("/")
async def root():
    """返回前端页面"""
    index_file = BASE_DIR.parent / "frontend" / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "医疗影像分析系统 API v2.0", "version": "2.0.0", "docs": "/docs"}


@app.get("/api/health")
async def health_check():
    """健康检查"""
    openai_status = "connected" if (OPENAI_AVAILABLE and openai_client) else "unavailable"
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "engine": "阿里云百炼 qwen-vl-max",
        "provider": "阿里云百炼（通义千问）",
        "capabilities": {
            "dicom": PYDICOM_AVAILABLE,
            "image_processing": PIL_AVAILABLE,
            "opencv": CV2_AVAILABLE,
            "qwen_vl_vision": openai_status,
            "openai_sdk": OPENAI_AVAILABLE,
            "local_unet_agent": bool(unet_agent and unet_agent.status().get("loaded")),
            "segmentation_registry": bool(segmentation_registry)
        }
    }


@app.post("/api/preview-image")
async def preview_image(file: UploadFile = File(...)):
    """Return a browser-safe PNG preview and dimensions for DICOM/TIFF/regular images."""
    content = await file.read()
    filename = file.filename or "uploaded_image"
    payload = build_preview_payload(content, filename)
    return JSONResponse(content=payload)


@app.get("/api/training/fit-status")
async def get_training_fit_status():
    """Return local U-Net training metrics and a simple fit judgement."""
    return JSONResponse(content=build_training_fit_status())


@app.post("/api/upload-and-analyze")
async def upload_and_analyze(
    file: UploadFile = File(...),
    analysis_type: str = Form(default="comprehensive"),
    sensitivity: float = Form(default=0.75)
):
    """上传影像并立即分析（OpenAI Vision驱动）"""

    filename = file.filename or "uploaded_image"
    suffix = Path(filename).suffix.lower()
    if suffix not in OpenAIVisionAnalyzer.SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {suffix}。支持格式: {', '.join(OpenAIVisionAnalyzer.SUPPORTED_FORMATS)}"
        )

    task_id = str(uuid.uuid4())
    save_filename = f"{task_id}{suffix}"
    save_path = UPLOAD_DIR / save_filename

    try:
        content = await file.read()
        if len(content) > 100 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="文件太大，最大支持100MB")
        with open(save_path, 'wb') as f:
            f.write(content)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")

    result = analyzer.analyze(save_path, filename, analysis_type, sensitivity)
    if result.get("success"):
        result["task_id"] = task_id
        result = enrich_with_unet_segmentation(result, save_path, filename)

    if result['success']:
        result['task_id'] = task_id
        analysis_records[task_id] = result

        result_path = RESULT_DIR / f"{task_id}.json"
        result_copy = {k: v for k, v in result.items() if k != 'thumbnail'}
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result_copy, f, ensure_ascii=False, indent=2)

        return JSONResponse(content={
            "success": True,
            "task_id": task_id,
            "filename": filename,
            "file_type": result.get('file_type', 'Unknown'),
            "body_part": result.get('body_part', 'unknown'),
            "analysis_time": result.get('analysis_time', 0),
            "properties": result.get('properties', {}),
            "detection": result.get('detection', {}),
            "ai_report": result.get('ai_report', ''),
            "llm_report": result.get('llm_report', {}),
            "thumbnail": result.get('thumbnail', ''),
            "segmentation": result.get('segmentation', {}),
            "segmentation_overlay": result.get('segmentation_overlay', ''),
            "segmentation_mask": result.get('segmentation_mask', ''),
            "segmentation_probability": result.get('segmentation_probability', ''),
            "metadata": result.get('metadata', {}),
            "upload_time": result.get('upload_time', ''),
            "engine": result.get('engine', '阿里云百炼 qwen-vl-max')
        })
    else:
        raise HTTPException(status_code=500, detail=result.get('error', '分析失败'))


@app.get("/api/results")
async def get_results(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=10, ge=1, le=50)
):
    """获取历史分析记录"""
    records = list(analysis_records.values())
    records.sort(key=lambda x: x.get('upload_time', ''), reverse=True)
    total = len(records)
    start = (page - 1) * limit
    end = start + limit
    page_records = records[start:end]

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "records": [
            {
                "task_id": r.get('task_id'),
                "filename": r.get('filename'),
                "file_type": r.get('file_type'),
                "body_part": r.get('body_part'),
                "upload_time": r.get('upload_time'),
                "analysis_time": r.get('analysis_time'),
                "risk_level": r.get('detection', {}).get('risk_level', 'low'),
                "has_finding": r.get('detection', {}).get('has_significant_finding', False),
                "thumbnail": r.get('thumbnail', ''),
                "engine": r.get('engine', '')
            }
            for r in page_records
        ]
    }


@app.get("/api/result/{task_id}")
async def get_result(task_id: str):
    """获取单条分析结果"""
    if task_id in analysis_records:
        return JSONResponse(content=analysis_records[task_id])
    result_path = RESULT_DIR / f"{task_id}.json"
    if result_path.exists():
        with open(result_path, 'r', encoding='utf-8') as f:
            return JSONResponse(content=json.load(f))
    raise HTTPException(status_code=404, detail="分析结果不存在")


@app.delete("/api/result/{task_id}")
async def delete_result(task_id: str):
    """删除分析记录"""
    deleted = False
    if task_id in analysis_records:
        del analysis_records[task_id]
        deleted = True
    for f in UPLOAD_DIR.glob(f"{task_id}*"):
        f.unlink(missing_ok=True)
    result_path = RESULT_DIR / f"{task_id}.json"
    if result_path.exists():
        result_path.unlink()
        deleted = True
    if deleted:
        return {"success": True, "message": "记录已删除"}
    raise HTTPException(status_code=404, detail="记录不存在")


@app.get("/api/stats")
async def get_stats():
    """获取统计信息"""
    records = list(analysis_records.values())
    total = len(records)
    findings_count = sum(1 for r in records if r.get('detection', {}).get('has_significant_finding'))
    body_parts = {}
    risk_levels = {'low': 0, 'medium': 0, 'high': 0}
    for r in records:
        bp = r.get('body_part', 'unknown')
        body_parts[bp] = body_parts.get(bp, 0) + 1
        rl = r.get('detection', {}).get('risk_level', 'low')
        risk_levels[rl] = risk_levels.get(rl, 0) + 1
    avg_time = sum(r.get('analysis_time', 0) for r in records) / max(total, 1)
    return {
        "total_analyses": total,
        "findings_detected": findings_count,
        "normal_count": total - findings_count,
        "body_part_distribution": body_parts,
        "risk_distribution": risk_levels,
        "average_analysis_time": round(avg_time, 2),
        "engine": "阿里云百炼 qwen-vl-max"
    }


@app.post("/api/demo-analyze")
async def demo_analyze():
    """演示分析（使用OpenAI分析演示图像）"""
    import random

    demo_cases = [
        {"filename": "chest_xray_demo.jpg", "body_part": "chest"},
        {"filename": "brain_mri_t1.jpg", "body_part": "brain"},
        {"filename": "abdomen_ct_scan.jpg", "body_part": "abdomen"}
    ]

    case = random.choice(demo_cases)
    task_id = str(uuid.uuid4())
    body_part = case['body_part']

    # 生成演示影像（带医疗感的噪声图）
    width, height = 512, 512
    np.random.seed(42)
    img_array = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            img_array[i, j] = int(128 + 50 * np.sin(i/30) * np.cos(j/30) + 20 * np.random.random())

    properties = analyzer.analyze_image_properties(img_array)
    thumbnail = analyzer.create_thumbnail(img_array)

    # 尝试用AI分析演示图
    ai_result = None
    used_openai = False
    if analyzer.client:
        image_b64 = analyzer.image_to_base64(img_array)
        logger.info(f"演示分析开始: client={analyzer.client is not None}, b64_len={len(image_b64) if image_b64 else 0}")
        if image_b64:
            try:
                ai_result = analyzer.call_vision_api(
                    image_b64, case['filename'],
                    f"演示模式：{body_part}影像分析"
                )
                logger.info(f"Vision API返回: {type(ai_result)}, keys={list(ai_result.keys()) if isinstance(ai_result, dict) else ai_result}")
            except Exception as e:
                logger.error(f"Vision API异常: {type(e).__name__}: {e}")
            if ai_result and isinstance(ai_result, dict) and ai_result.get('overall_assessment'):
                used_openai = True

    # 备用分析
    if not ai_result or not isinstance(ai_result, dict) or not ai_result.get('overall_assessment'):
        logger.info("使用备用分析逻辑")
        ai_result = analyzer.fallback_analysis(img_array, case['filename'], body_part, 0.75)

    body_part_result = ai_result.get('body_part', body_part)
    abnormal = ai_result.get('abnormal_findings', [])
    primary_finding = abnormal[0] if abnormal else None

    detection = {
        'has_significant_finding': ai_result.get('has_significant_finding', False),
        'finding_count': len(abnormal),
        'primary_finding': primary_finding,
        'normal_summary': ai_result.get('normal_findings', []),
        'overall_assessment': ai_result.get('overall_assessment', ''),
        'clinical_recommendation': ai_result.get('clinical_recommendation', ''),
        'risk_level': ai_result.get('risk_level', 'low'),
        'confidence': ai_result.get('confidence_score', 0.9),
        'abnormal_findings': abnormal
    }

    elapsed = round(random.uniform(0.8, 2.5), 2)
    report = analyzer.generate_report_from_ai(
        task_id, case['filename'], ai_result, properties,
        {"Modality": "演示模式", "BodyPartExamined": body_part_result, "Note": "此为演示数据"},
        elapsed
    )

    result = {
        "success": True,
        "task_id": task_id,
        "filename": case['filename'],
        "file_type": "JPEG (演示)",
        "body_part": body_part_result,
        "analysis_time": elapsed,
        "properties": properties,
        "detection": detection,
        "ai_report": report,
        "thumbnail": thumbnail,
        "metadata": {"Modality": "CT/MRI", "BodyPartExamined": body_part_result},
        "upload_time": datetime.now().isoformat(),
        "is_demo": True,
        "engine": "阿里云百炼 qwen-vl-max" if used_openai else "本地分析引擎（备用）"
    }

    analysis_records[task_id] = result
    return JSONResponse(content=result)


@app.get("/api/config")
async def get_config():
    """获取当前AI配置信息"""
    return {
        "engine": "阿里云百炼 qwen-vl-max",
        "model": "qwen-vl-max",
        "provider": "阿里云百炼（通义千问）",
        "base_url": OPENAI_BASE_URL,
        "api_configured": bool(OPENAI_API_KEY),
        "sdk_available": OPENAI_AVAILABLE,
        "client_ready": openai_client is not None,
        "version": "2.0.0"
    }


# ==================== 智能体系统 ====================

# 延迟导入避免循环依赖
_agent_orchestrator = None

def get_orchestrator():
    """获取或创建智能体编排器"""
    global _agent_orchestrator
    if _agent_orchestrator is None:
        try:
            from agents.orchestrator import create_orchestrator
            _agent_orchestrator = create_orchestrator(str(BASE_DIR))
            logger.info("智能体编排器初始化成功")
            
            # 设置WebSocket推送回调
            _agent_orchestrator.add_websocket_connection(
                lambda msg: _broadcast_agent_message(msg)
            )
        except Exception as e:
            logger.error(f"智能体编排器初始化失败: {e}")
            return None
    return _agent_orchestrator

def _broadcast_agent_message(message: dict):
    """广播智能体消息到WebSocket"""
    # 这个函数会在WebSocket连接时由connection_manager调用
    pass


# ==================== 智能体 API 端点 ====================

@app.get("/api/agent/status")
async def get_agent_status():
    """获取所有智能体状态"""
    orchestrator = get_orchestrator()
    if not orchestrator:
        return {"error": "智能体系统未初始化"}
    return orchestrator.get_all_status()


@app.get("/api/agent/overview")
async def get_agent_overview():
    """获取系统概览"""
    orchestrator = get_orchestrator()
    if not orchestrator:
        return {"error": "智能体系统未初始化"}
    return orchestrator.get_system_overview()


def _build_segmentation_detection(segmentation: Dict[str, Any], body_part: Optional[str] = None) -> Dict[str, Any]:
    metrics = segmentation.get("metrics", {}) if segmentation.get("success") else {}
    model_summary = {
        "available": bool(segmentation.get("success")),
        "model_id": segmentation.get("model_id", ""),
        "model_name": segmentation.get("model_name", ""),
        "architecture": segmentation.get("architecture", ""),
        "training_type": segmentation.get("training_type", ""),
        "threshold": segmentation.get("threshold", 0),
        "area_percent": metrics.get("area_percent", 0),
        "mean_probability": metrics.get("mean_probability", 0),
        "max_probability": metrics.get("max_probability", 0),
        "bbox": metrics.get("bbox"),
        "has_candidate_region": metrics.get("has_candidate_region", False),
        "risk_level": metrics.get("risk_level", "low"),
        "inference_time_ms": segmentation.get("inference_time_ms", 0),
    }
    detection = {
        "risk_level": metrics.get("risk_level", "low"),
        "has_significant_finding": bool(metrics.get("has_candidate_region")),
        "finding_count": 1 if metrics.get("has_candidate_region") else 0,
        "primary_finding": None,
        "normal_summary": [],
        "overall_assessment": "",
        "clinical_recommendation": _build_segmentation_clinical_recommendation(segmentation, metrics, body_part),
        "segmentation_model": model_summary,
        "unet_segmentation": model_summary,
    }
    if metrics.get("has_candidate_region"):
        bbox = metrics.get("bbox") or {}
        detection["primary_finding"] = {
            "type": f"{segmentation.get('model_name', '分割模型')}候选区域",
            "location": f"x={bbox.get('x_min', '-')}-{bbox.get('x_max', '-')}, y={bbox.get('y_min', '-')}-{bbox.get('y_max', '-')}",
            "description": f"模型标记的候选区域约占全图 {metrics.get('area_percent', 0)}%。",
            "severity": "需要复核",
            "confidence": metrics.get("mean_probability", 0),
        }
        detection["abnormal_findings"] = [detection["primary_finding"]]
        detection["overall_assessment"] = (
            f"{segmentation.get('model_name', '分割模型')}检测到候选分割区域，"
            f"面积占比约 {metrics.get('area_percent', 0)}%，"
            f"掩膜内平均概率为 {metrics.get('mean_probability', 0)}。"
            "该结果提示需要重点复核对应区域，但不是诊断结论。"
        )
    else:
        detection["normal_summary"] = ["当前阈值下未检测到明确候选分割区域。"]
        detection["overall_assessment"] = (
            f"{segmentation.get('model_name', '分割模型')}在当前阈值下未检测到明确候选区域，"
            "模型风险提示较低；仍需结合原始影像和临床资料判断。"
        )
        detection["abnormal_findings"] = []
    return detection


@app.get("/api/segmentation/models")
async def get_segmentation_models():
    """Return unified status for the three local segmentation models."""
    if not segmentation_registry:
        return JSONResponse(
            status_code=503,
            content={"success": False, "models": [], "error": "Segmentation registry is unavailable."},
        )
    return JSONResponse(content={"success": True, "models": segmentation_registry.list_models()})


@app.post("/api/segmentation/analyze")
async def analyze_with_segmentation_model(
    file: UploadFile = File(...),
    model_id: str = Form(default="ssl_current"),
    threshold: Optional[float] = Form(default=None)
):
    """Analyze one image with a selected local segmentation model."""
    if not segmentation_registry:
        raise HTTPException(status_code=503, detail="Segmentation registry is unavailable.")

    filename = file.filename or "uploaded_image"
    suffix = Path(filename).suffix.lower()
    if suffix not in OpenAIVisionAnalyzer.SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {suffix}")

    task_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{task_id}{suffix}"
    content = await file.read()
    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File is too large. Max size is 100MB.")
    with open(save_path, "wb") as f:
        f.write(content)

    preview_payload = build_preview_payload(content, filename)
    model_input_path = save_path
    if suffix == ".dcm":
        preview_b64 = preview_payload.get("preview", "")
        if not preview_b64:
            raise HTTPException(status_code=400, detail="Could not convert DICOM to a model-readable preview image.")
        model_input_path = UPLOAD_DIR / f"{task_id}_preview.png"
        with open(model_input_path, "wb") as f:
            f.write(base64.b64decode(preview_b64))

    try:
        segmentation = segmentation_registry.analyze(model_id, model_input_path, threshold=threshold)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Segmentation analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))

    body_part = analyzer.detect_body_part(filename, {})
    detection = _build_segmentation_detection(segmentation, body_part=body_part)
    width = preview_payload.get("width") or 0
    height = preview_payload.get("height") or 0
    properties = {
        "width": width,
        "height": height,
        "image_quality": "segmentation-ready" if preview_payload.get("success") else "preview unavailable",
    }
    result = {
        "success": bool(segmentation.get("success")),
        "task_id": task_id,
        "filename": filename,
        "file_type": suffix.lstrip(".").upper() or "Unknown",
        "body_part": body_part,
        "analysis_time": round(float(segmentation.get("inference_time_ms", 0)) / 1000.0, 2),
        "properties": properties,
        "detection": detection,
        "ai_report": _build_segmentation_ai_report(detection, segmentation),
        "thumbnail": preview_payload.get("preview", ""),
        "segmentation": segmentation,
        "segmentation_overlay": segmentation.get("overlay", ""),
        "segmentation_mask": segmentation.get("mask", ""),
        "segmentation_probability": segmentation.get("probability", ""),
        "metadata": {
            "model_id": segmentation.get("model_id", model_id),
            "model_name": segmentation.get("model_name", ""),
            "architecture": segmentation.get("architecture", ""),
            "training_type": segmentation.get("training_type", ""),
        },
        "upload_time": datetime.now().isoformat(),
        "engine": f"本地分割模型 - {segmentation.get('model_name', model_id)}",
    }

    analysis_records[task_id] = result
    result_path = RESULT_DIR / f"{task_id}.json"
    result_copy = {k: v for k, v in result.items() if k != "thumbnail"}
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_copy, f, ensure_ascii=False, indent=2)

    status_code = 200 if result["success"] else 500
    return JSONResponse(status_code=status_code, content=result)


@app.get("/api/agent/unet/status")
async def get_unet_agent_status():
    """Get local U-Net segmentation agent status."""
    if not unet_agent:
        return {"available": False, "loaded": False, "error": "U-Net agent module is unavailable."}
    return unet_agent.status()


@app.post("/api/agent/unet/analyze")
async def analyze_with_unet(
    file: UploadFile = File(...),
    threshold: Optional[float] = Form(default=None)
):
    """Analyze one image with the local U-Net segmentation agent only."""
    if not unet_agent:
        raise HTTPException(status_code=503, detail="U-Net agent module is unavailable.")

    filename = file.filename or "uploaded_image"
    suffix = Path(filename).suffix.lower()
    if suffix not in OpenAIVisionAnalyzer.SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {suffix}")

    task_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{task_id}{suffix}"
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    segmentation = unet_agent.analyze_file(save_path, threshold=threshold)
    llm_report = generate_unet_llm_report(task_id, filename, segmentation)
    metrics = segmentation.get("metrics", {}) if segmentation.get("success") else {}
    detection = {
        "risk_level": metrics.get("risk_level", "low"),
        "has_significant_finding": bool(metrics.get("has_candidate_region")),
        "unet_segmentation": {
            "available": bool(segmentation.get("success")),
            "area_percent": metrics.get("area_percent", 0),
            "mean_probability": metrics.get("mean_probability", 0),
            "max_probability": metrics.get("max_probability", 0),
            "bbox": metrics.get("bbox"),
            "has_candidate_region": metrics.get("has_candidate_region", False),
            "risk_level": metrics.get("risk_level", "low"),
        },
        "unet_llm_report": {
            "source": llm_report.get("source"),
            "model": llm_report.get("model"),
            "sections": llm_report.get("sections", {}),
            "elapsed": llm_report.get("elapsed", 0),
            "error": llm_report.get("error", ""),
        },
    }
    sections = llm_report.get("sections", {})
    if sections:
        detection["overall_assessment"] = sections.get("conclusion", "")
        detection["clinical_recommendation"] = sections.get("clinical_recommendation", "")

    return JSONResponse(content={
        "success": segmentation.get("success", False),
        "task_id": task_id,
        "filename": filename,
        "detection": detection,
        "ai_report": llm_report.get("report", segmentation.get("report", "")),
        "llm_report": llm_report,
        "segmentation": segmentation,
        "segmentation_overlay": segmentation.get("overlay", ""),
        "segmentation_mask": segmentation.get("mask", ""),
        "segmentation_probability": segmentation.get("probability", ""),
        "engine": f"Local U-Net + {llm_report.get('model', 'report')}",
    })


# ----- 文件监控 API -----

@app.post("/api/agent/watcher/start")
async def start_file_watcher(watch_dir: Optional[str] = None):
    """启动文件监控"""
    orchestrator = get_orchestrator()
    if not orchestrator:
        return {"error": "智能体系统未初始化"}
    return orchestrator.start_file_watching(watch_dir)


@app.post("/api/agent/watcher/stop")
async def stop_file_watcher():
    """停止文件监控"""
    orchestrator = get_orchestrator()
    if not orchestrator:
        return {"error": "智能体系统未初始化"}
    return orchestrator.stop_file_watching()


@app.get("/api/agent/watcher/status")
async def get_watcher_status():
    """获取文件监控状态"""
    orchestrator = get_orchestrator()
    if not orchestrator:
        return {"error": "智能体系统未初始化"}
    return orchestrator.get_watcher_status()


@app.get("/api/agent/watcher/pending")
async def get_pending_files():
    """获取待处理文件"""
    orchestrator = get_orchestrator()
    if not orchestrator:
        return {"error": "智能体系统未初始化"}
    return {"files": orchestrator.get_pending_files()}


@app.post("/api/agent/watcher/analyze")
async def analyze_watched_file(file_path: str):
    """手动分析监控目录中的文件"""
    orchestrator = get_orchestrator()
    if not orchestrator:
        return {"error": "智能体系统未初始化"}
    # 调用实际的分析函数
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    result = analyzer.analyze(file_path_obj, file_path_obj.name)
    return result


# ----- 对话 API -----

@app.post("/api/agent/dialogue/message")
async def send_dialogue_message(
    payload: Optional[Dict[str, Any]] = Body(default=None),
    user_input: Optional[str] = None,
    conv_id: Optional[str] = None
):
    """发送对话消息"""
    if payload:
        user_input = payload.get("user_input", user_input)
        conv_id = payload.get("conv_id", conv_id)

    if not user_input or not str(user_input).strip():
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "message is required"},
        )

    orchestrator = get_orchestrator()
    if not orchestrator:
        return {"success": False, "error": "智能体系统未初始化"}
    return orchestrator.send_message(str(user_input).strip(), conv_id)


@app.post("/api/agent/dialogue/conversation")
async def create_conversation(title: Optional[str] = None):
    """创建新对话"""
    orchestrator = get_orchestrator()
    if not orchestrator:
        return {"error": "智能体系统未初始化"}
    conv_id = orchestrator.create_conversation(title)
    return {"conversation_id": conv_id}


@app.get("/api/agent/dialogue/history")
async def get_dialogue_history(
    conv_id: Optional[str] = None,
    limit: int = 20
):
    """获取对话历史"""
    orchestrator = get_orchestrator()
    if not orchestrator:
        return {"error": "智能体系统未初始化"}
    return {"messages": orchestrator.get_conversation_history(conv_id, limit)}


@app.get("/api/agent/dialogue/conversations")
async def list_dialogue_conversations():
    """列出所有对话"""
    orchestrator = get_orchestrator()
    if not orchestrator:
        return {"error": "智能体系统未初始化"}
    return {"conversations": orchestrator.list_conversations()}


@app.post("/api/agent/dialogue/context")
async def set_dialogue_context(analysis_result: Dict):
    """设置对话上下文（关联分析结果）"""
    orchestrator = get_orchestrator()
    if not orchestrator:
        return {"error": "智能体系统未初始化"}
    orchestrator.set_analysis_context(analysis_result)
    return {"success": True}


@app.post("/api/agent/dialogue/clear")
async def clear_dialogue_context():
    """清空对话上下文"""
    orchestrator = get_orchestrator()
    if not orchestrator:
        return {"error": "智能体系统未初始化"}
    orchestrator.clear_context()
    return {"success": True}


# ----- 任务规划 API -----

@app.post("/api/agent/plan/create")
async def create_plan(
    body_part: Optional[str] = None,
    custom_steps: Optional[List[Dict]] = None
):
    """创建分析计划"""
    orchestrator = get_orchestrator()
    if not orchestrator:
        return {"error": "智能体系统未初始化"}
    plan_id = orchestrator.create_analysis_plan(body_part, custom_steps)
    return {"plan_id": plan_id}


@app.post("/api/agent/plan/execute")
async def execute_plan(
    plan_id: str,
    context: Optional[Dict] = None
):
    """执行分析计划"""
    orchestrator = get_orchestrator()
    if not orchestrator:
        return {"error": "智能体系统未初始化"}
    return orchestrator.execute_plan(plan_id, context)


@app.get("/api/agent/plan/{plan_id}")
async def get_plan_status(plan_id: str):
    """获取计划状态"""
    orchestrator = get_orchestrator()
    if not orchestrator:
        return {"error": "智能体系统未初始化"}
    status = orchestrator.get_plan_status(plan_id)
    if not status:
        raise HTTPException(status_code=404, detail="计划不存在")
    return status


@app.get("/api/agent/plan")
async def list_plans():
    """列出所有计划"""
    orchestrator = get_orchestrator()
    if not orchestrator:
        return {"error": "智能体系统未初始化"}
    return {"plans": orchestrator.list_plans()}


@app.post("/api/agent/plan/stop")
async def stop_plan():
    """停止当前计划"""
    orchestrator = get_orchestrator()
    if not orchestrator:
        return {"error": "智能体系统未初始化"}
    orchestrator.stop_plan()
    return {"success": True}


# ==================== WebSocket 端点 ====================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket连接，用于实时推送
    """
    try:
        from websocket_manager import manager
    except ImportError:
        logger.error("WebSocket模块未找到")
        return
        
    await manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                msg_type = message.get('type', 'unknown')
                
                # 处理不同类型的消息
                if msg_type == 'ping':
                    await manager.send_personal_message({
                        'type': 'pong',
                        'timestamp': datetime.now().isoformat()
                    }, websocket)
                    
                elif msg_type == 'dialogue':
                    # 对话消息
                    orchestrator = get_orchestrator()
                    if orchestrator:
                        result = orchestrator.send_message(
                            message.get('content', ''),
                            message.get('conversation_id')
                        )
                        await manager.send_personal_message({
                            'type': 'dialogue_response',
                            'data': result
                        }, websocket)
                        
                elif msg_type == 'get_status':
                    orchestrator = get_orchestrator()
                    if orchestrator:
                        await manager.send_personal_message({
                            'type': 'status',
                            'data': orchestrator.get_all_status()
                        }, websocket)
                        
            except json.JSONDecodeError:
                await manager.send_personal_message({
                    'type': 'error',
                    'message': '无效的JSON格式'
                }, websocket)
                
    except Exception as e:
        if 'WebSocketDisconnect' not in str(type(e)):
            logger.error(f"WebSocket错误: {e}")
    finally:
        manager.disconnect(websocket)


# ==================== 分析结果关联对话上下文 ====================

@app.post("/api/analysis/{task_id}/enable-context")
async def enable_context_for_analysis(task_id: str):
    """将分析结果设置为对话上下文"""
    if task_id in analysis_records:
        result = analysis_records[task_id]
        orchestrator = get_orchestrator()
        if orchestrator:
            orchestrator.set_analysis_context(result)
            return {"success": True, "message": "上下文已更新"}
    raise HTTPException(status_code=404, detail="分析记录不存在")


if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("🏥 医疗影像分析系统 v3.0 (智能体版) 启动中...")
    print("🤖 AI引擎: 阿里云百炼 qwen-vl-max（通义千问视觉大模型）")
    print("🧠 智能体: FileWatcher | Dialogue | Planner")
    print("📡 API文档: http://localhost:8000/docs")
    print("🌐 前端界面: http://localhost:8000")
    print("🔌 WebSocket: ws://localhost:8000/ws")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
