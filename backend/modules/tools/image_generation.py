"""图片生成工具

使用豆包 Seedream 模型生成图片。
支持纯文本生成图片，以及以参考图为基础生成图片。
"""

import os
import uuid
import requests
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from backend.modules.tools.base import Tool


class ImageGenerationTool(Tool):
    """图片生成工具 - 使用豆包 Seedream 模型生成图片"""

    # 支持的尺寸（API 最低要求 3686400 像素）
    SUPPORTED_SIZES = ["1920x1920", "2048x2048", "2560x1440", "1440x2560"]
    MIN_PIXELS = 3686400

    # 默认配置
    DEFAULT_MODEL = "doubao-seedream-5-0-260128"
    DEFAULT_SIZE = "1920x1920"
    API_BASE = "https://ark.cn-beijing.volces.com/api/v3/images/generations"

    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.output_dir = self.workspace / "generated_images"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"ImageGenerationTool initialized: workspace={self.workspace}")

    @property
    def name(self) -> str:
        return "image_generation"

    @property
    def description(self) -> str:
        return (
            "根据文字描述生成图片。当用户要求生成图片、画图、创作图片时使用此工具。"
            "例如：'画一只猫咪'、'生成一张风景图'、'帮我创作一个logo'。"
            "支持图生图：上传参考图 + 描述，可基于原图风格生成。"
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "图片生成的提示词，描述你想要生成的图片内容。支持中英文。",
                },
                "size": {
                    "type": "string",
                    "enum": ["1920x1920", "2048x2048", "2560x1440", "1440x2560"],
                    "description": "生成图片的尺寸。不指定时，如果有参考图则按原图尺寸生成，否则默认1920x1920。",
                },
                "reference_image": {
                    "type": "string",
                    "description": "参考图片路径。不指定尺寸时会自动按原图尺寸生成。图生图模式。",
                },
            },
            "required": ["prompt"],
        }

    def _get_image_size(self, image_path: Path) -> Optional[tuple]:
        """获取图片尺寸 (width, height)"""
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                return img.size
        except ImportError:
            try:
                import struct
                with open(image_path, "rb") as f:
                    header = f.read(32)
                    if header[:2] == b'\xff\xd8':  # JPEG
                        f.seek(0)
                        f.read(2)
                        while True:
                            marker, size = struct.unpack('>2sH', f.read(4))
                            if marker[0] != 0xFF:
                                break
                            if marker[1] in (0xC0, 0xC1, 0xC2):
                                f.read(1)
                                h, w = struct.unpack('>HH', f.read(4))
                                return (w, h)
                            f.read(size - 2)
                    elif header[:8] == b'\x89PNG\r\n\x1a\n':  # PNG
                        w = struct.unpack('>I', header[16:20])[0]
                        h = struct.unpack('>I', header[20:24])[0]
                        return (w, h)
            except Exception:
                pass
        return None

    def _match_size(self, width: int, height: int) -> str:
        """根据原图宽高比匹配最接近的支持尺寸"""
        ratio = width / height

        if ratio >= 1:
            candidates = [
                ("2560x1440", 2560 / 1440),
                ("1920x1920", 1.0),
                ("2048x2048", 1.0),
                ("1440x2560", 1440 / 2560),
            ]
        else:
            candidates = [
                ("1440x2560", 1440 / 2560),
                ("1920x1920", 1.0),
                ("2048x2048", 1.0),
                ("2560x1440", 2560 / 1440),
            ]

        best = min(candidates, key=lambda x: abs(x[1] - ratio))
        return best[0]

    def _download_image(self, url: str, filename: str) -> str:
        """下载图片到本地"""
        output_path = self.output_dir / filename
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(resp.content)
        return str(output_path)

    async def execute(self, **kwargs: Any) -> str:
        """执行图片生成"""
        prompt = kwargs.get("prompt", "")
        size = kwargs.get("size", None)
        reference_image = kwargs.get("reference_image", None)

        if not prompt:
            return "错误：请提供图片描述（prompt）"

        # 处理尺寸逻辑
        ref_full_path = None
        if reference_image:
            ref_full_path = self.workspace / reference_image
            if not ref_full_path.exists():
                ref_full_path = None

        if not size:
            if ref_full_path:
                img_size = self._get_image_size(ref_full_path)
                if img_size:
                    size = self._match_size(img_size[0], img_size[1])
                    logger.info(f"按原图尺寸匹配: {img_size[0]}x{img_size[1]} -> {size}")
                else:
                    size = self.DEFAULT_SIZE
            else:
                size = self.DEFAULT_SIZE

        if size not in self.SUPPORTED_SIZES:
            size = self.DEFAULT_SIZE

        logger.info(f"图片生成: prompt={prompt[:50]}..., size={size}")

        try:
            # 运行时读取 API Key
            api_key = os.environ.get("ARK_API_KEY", "")
            if not api_key:
                return "错误：未配置 ARK_API_KEY 环境变量。请在 .env 文件中设置 ARK_API_KEY"

            payload: Dict[str, Any] = {
                "model": self.DEFAULT_MODEL,
                "prompt": prompt,
                "size": size,
            }

            # 如果有参考图片
            if ref_full_path:
                import base64
                with open(ref_full_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")
                ext = ref_full_path.suffix.lower()
                mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png"}.get(ext.lstrip("."), "jpeg")
                payload["image"] = f"data:image/{mime};base64,{img_b64}"

            resp = requests.post(
                self.API_BASE,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=120,
            )

            data = resp.json()

            if "error" in data:
                error_msg = data["error"].get("message", str(data["error"]))
                logger.error(f"图片生成失败: {error_msg}")
                return f"图片生成失败: {error_msg}"

            image_data = data.get("data", [])
            if not image_data:
                return "图片生成失败：未返回图片数据"

            image_url = image_data[0].get("url", "")
            if not image_url:
                return "图片生成失败：未返回图片 URL"

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            short_id = uuid.uuid4().hex[:8]
            filename = f"generated_{timestamp}_{short_id}.jpeg"
            local_path = self._download_image(image_url, filename)

            file_size = os.path.getsize(local_path)
            usage = data.get("usage", {})

            output = (
                f"图片生成成功！\n"
                f"文件路径: {local_path}\n"
                f"尺寸: {size}\n"
                f"文件大小: {file_size:,} bytes"
            )

            if usage:
                output += f"\nToken 消耗: {usage.get('total_tokens', 0)}"

            output += f"\n\n请使用 send_media 工具将此图片发送给用户。"

            logger.info(f"图片已生成: {local_path} ({file_size} bytes)")
            return output

        except requests.exceptions.Timeout:
            return "图片生成超时，请稍后重试"
        except requests.exceptions.RequestException as e:
            logger.error(f"图片生成请求失败: {e}")
            return f"图片生成请求失败: {str(e)}"
        except Exception as e:
            logger.error(f"图片生成失败: {e}")
            return f"图片生成失败: {str(e)}"
