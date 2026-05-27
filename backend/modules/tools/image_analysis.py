"""图像分析工具

使用多模态视觉模型分析图片内容。
支持豆包、智谱、千问等视觉模型。
"""

import json
import os
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from loguru import logger

from backend.modules.tools.base import Tool


class ImageAnalysisTool(Tool):
    """图像分析工具 - 使用视觉模型分析图片内容"""

    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self._config = None
        logger.debug(f"ImageAnalysisTool initialized: workspace={self.workspace}")

    def _load_config(self) -> Dict:
        """加载图像分析配置"""
        if self._config is not None:
            return self._config

        config_path = (
            self.workspace
            / "skills"
            / "image-analysis"
            / "scripts"
            / "config.json"
        )
        if not config_path.exists():
            raise FileNotFoundError(f"图像分析配置文件不存在: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self._config = json.load(f)
        return self._config

    def _encode_image(self, image_path: str) -> str:
        """将本地图片转换为 base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _is_local_file(self, path: str) -> bool:
        """判断是否为本地文件"""
        return os.path.exists(path) or (
            not path.startswith("http://") and not path.startswith("https://")
        )

    def _prepare_image_url(self, image_path: str) -> str:
        """准备图片 URL：本地文件转 Base64 Data URL，网络 URL 直接返回"""
        full_path = self.workspace / image_path
        if full_path.exists():
            image_path = str(full_path)

        if self._is_local_file(image_path):
            ext = Path(image_path).suffix.lower()
            mime_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
                ".bmp": "image/bmp",
            }.get(ext, "image/jpeg")

            base64_image = self._encode_image(image_path)
            return f"data:{mime_type};base64,{base64_image}"
        else:
            return image_path

    def _analyze_with_doubao(
        self,
        prompt: str,
        images: List[str],
        model: str,
        base_url: str,
        api_key: str,
    ) -> Dict:
        """使用豆包视觉模型分析"""
        content = []
        for image in images:
            image_url = self._prepare_image_url(image)
            content.append({"type": "image_url", "image_url": {"url": image_url}})

        content.append({"type": "text", "text": prompt})

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(base_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()

    def _analyze_with_openai_compat(
        self,
        prompt: str,
        images: List[str],
        model: str,
        base_url: str,
        api_key: str,
    ) -> Dict:
        """使用 OpenAI 兼容接口分析（智谱、千问等）"""
        content = []
        for image in images:
            image_url = self._prepare_image_url(image)
            content.append({"type": "image_url", "image_url": {"url": image_url}})

        content.append({"type": "text", "text": prompt})

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(base_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()

    @property
    def name(self) -> str:
        return "image_analysis"

    @property
    def description(self) -> str:
        return (
            "分析图片内容。支持图片描述、OCR文字提取、场景理解等。"
            "当用户发送图片时，使用此工具来识别和分析图片内容。"
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "images": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "图片路径列表（支持本地路径或网络 URL）。本地路径支持工作空间相对路径。",
                },
                "prompt": {
                    "type": "string",
                    "description": "分析提示词。例如：'描述这张图片'、'提取图片中的所有文字'、'这张图片里有什么'。",
                },
                "model": {
                    "type": "string",
                    "enum": ["doubao", "zhipu", "qwen"],
                    "description": "使用的视觉模型。默认使用配置文件中设定的默认模型。",
                },
            },
            "required": ["images", "prompt"],
        }

    async def execute(self, **kwargs: Any) -> str:
        """执行图像分析"""
        images = kwargs.get("images", [])
        prompt = kwargs.get("prompt", "")
        model = kwargs.get("model", None)

        if not images:
            return "错误：请至少提供一张图片"

        if not prompt:
            return "错误：请提供分析提示词"

        try:
            config = self._load_config()
        except FileNotFoundError as e:
            return f"错误：{e}"

        use_model = model or config.get("default_model", "doubao")

        # 获取模型配置
        model_config = config.get(use_model, {})
        api_key = model_config.get("api_key", "") or os.environ.get("ARK_API_KEY", "")
        model_name = model_config.get("model", "")
        base_url = model_config.get("base_url", "")

        if not api_key:
            return f"错误：{use_model} API Key 未配置。请设置环境变量 ARK_API_KEY 或在 config.json 中填写 api_key"

        if not model_name:
            return f"错误：{use_model} 模型名称未配置"

        if not base_url:
            return f"错误：{use_model} API 地址未配置"

        logger.info(
            f"图像分析: model={use_model}, images={len(images)}, prompt={prompt[:50]}..."
        )

        try:
            # 根据模型类型选择调用方式
            if use_model == "doubao":
                result = self._analyze_with_doubao(
                    prompt, images, model_name, base_url, api_key
                )
            else:
                result = self._analyze_with_openai_compat(
                    prompt, images, model_name, base_url, api_key
                )

            # 提取结果文本
            try:
                content = result["choices"][0]["message"]["content"]
                output = f"[图像分析结果 - {use_model}]\n{content}"

                if "usage" in result:
                    usage = result["usage"]
                    output += f"\n\n(Token: 输入{usage.get('prompt_tokens', 0)} | 输出{usage.get('completion_tokens', 0)})"

                return output
            except (KeyError, IndexError) as e:
                return f"解析结果失败: {str(e)}\n原始结果: {json.dumps(result, ensure_ascii=False, indent=2)}"

        except requests.exceptions.RequestException as e:
            logger.error(f"图像分析请求失败: {e}")
            return f"图像分析请求失败: {str(e)}"
        except Exception as e:
            logger.error(f"图像分析失败: {e}")
            return f"图像分析失败: {str(e)}"
