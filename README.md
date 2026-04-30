# 医疗影像 AI 分析智能体原型

一个面向医学影像二值分割的本地 AI 分析原型。项目使用 PyTorch U-Net 完成候选区域分割，并将分割面积、阈值、置信度、风险提示等结构化输出接入通义千问 API，生成医学影像辅助分析报告。

## 功能

- 医学影像上传与预览
- U-Net 二值分割推理
- 分割 mask、叠加图、概率图展示
- Dice、IoU、Recall、Precision、FP/FN 等指标实验分析
- 阈值搜索与低漏检部署策略
- 通义千问结构化报告生成
- 前后端本地联调页面

## 项目结果

- 验证集最佳 Dice：0.9407
- 测试集 Dice：约 0.9464
- 测试集 IoU：约 0.9002
- 部署阈值：0.36
- 低漏检优化：测试集 FN 从 33339 降至 29868

## 目录

```text
backend/                 FastAPI 后端与 U-Net/LLM 接口
backend/agents/          本地 U-Net 推理 agent
frontend/                单页前端界面
train_unet.py            U-Net 训练脚本
testset_visual_qa.ps1    测试集可视化 QA 脚本
docs/                    实验报告
```

## 本地运行

安装依赖：

```bash
pip install -r backend/requirements.txt
pip install torch torchvision numpy pillow tqdm
```

设置通义千问 API Key：

```bash
set OPENAI_API_KEY=your_dashscope_or_qwen_api_key
```

启动后端：

```bash
cd backend
python main.py
```

浏览器打开：

```text
http://localhost:8000/
```

## 说明

本仓库不包含原始医学影像数据、用户上传文件、运行日志、模型权重和任何 API Key。模型权重路径可通过环境变量配置。

本项目仅用于算法学习、项目展示和辅助分析原型验证，不可作为临床诊断依据。
