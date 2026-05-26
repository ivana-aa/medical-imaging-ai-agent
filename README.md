# 医疗影像三模型分割与 AI 分析平台

这是一个面向 CT 医学影像二值分割的本地可运行原型，统一集成三个真实推理模型：

- 初始 U-Net 基线模型
- Attention U-Net 精细分割模型
- 自监督 MIM 预训练 + 少量标注 hard-mining 微调的 ResNet-UNet ensemble

前端支持先选模型再上传影像，后端返回分割 mask、叠加图、概率图、候选区域位置、面积比例、风险等级和推理耗时。配置 API Key 后，还可生成中文辅助分析报告与临床建议提示。

## 核心指标

| 模型 | Test Dice | Test IoU | Precision | Recall | 部署阈值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 初始 U-Net | 0.94581 | 0.89933 | 0.94949 | 0.94365 | 0.36 |
| Attention U-Net | 0.95580 | 0.91630 | 0.94370 | 0.96860 | 0.23 |
| 自监督 ResNet-UNet Ensemble | 0.92074 | 0.86022 | 0.89812 | 0.95396 | 0.335 |

自监督模型还使用空掩膜 fallback 阈值 `0.0004` 和最小连通域面积 `50`。

## 仓库结构

```text
backend/                           FastAPI API、三模型 registry 与报告服务
frontend/                          单页可视化前端
models/weights/                    已训练的部署权重（Git LFS）
original_unet_project/             初始 U-Net 独立项目源码
attention_unet_project/            Attention U-Net 独立项目源码
self_supervised_learning_project/  自监督学习独立项目源码、配置与脚本
Dataset/README.md                  数据集放置规范（不包含原始影像）
setup.bat                          Windows 首次安装
start.bat                          Windows 一键启动
```

## 下载与启动

模型权重由 Git LFS 管理。克隆前请安装 [Git LFS](https://git-lfs.com/) 和 Python 3.10 或更高版本。

```powershell
git lfs install
git clone https://github.com/ivana-aa/medical-imaging-ai-agent.git
cd medical-imaging-ai-agent
git lfs pull
.\setup.bat
.\start.bat
```

浏览器访问：

```text
http://localhost:8000/
```

部署权重已包含在仓库的 LFS 内容中，因此无需数据集即可上传你自己的影像文件运行推理。

## 可选：报告生成

不配置 API Key 时，三个本地分割模型与可视化功能仍可正常使用。若需要对话和辅助报告生成，可在启动前设置通义千问兼容接口的 Key：

```powershell
$env:OPENAI_API_KEY="your_dashscope_or_qwen_api_key"
.\start.bat
```

不要提交 `.env` 或任何真实密钥。

## 可选：训练与评估

原始医学影像数据不在公开仓库中。若你具有合法授权，可按 [Dataset/README.md](Dataset/README.md) 的目录结构将数据放入 `Dataset/`，再进入三个项目目录运行其训练或评估脚本。

```text
Dataset/
  train/images/  train/labels/
  val/images/    val/labels/
  test/images/   test/labels/
```

自监督项目训练代码独立于初始 U-Net 和 Attention U-Net 项目；统一前端仅在推理阶段加载各项目训练得到的部署权重。

## 接口

- `GET /api/segmentation/models`：返回三个模型状态、指标、checkpoint 与默认阈值。
- `POST /api/segmentation/analyze`：上传图片并指定 `model_id`，返回真实推理结果。
- `/api/agent/unet/*`：保留原始 U-Net 兼容接口。

## 使用边界

本项目仅用于算法学习、项目展示和辅助分析原型验证。模型输出及生成的临床建议必须结合原始影像与临床背景由专业人员复核，不能替代医生诊断。
