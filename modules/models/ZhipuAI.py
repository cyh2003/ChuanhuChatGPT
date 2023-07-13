from __future__ import annotations
from typing import TYPE_CHECKING, List
import zhipuai

import logging
import json
import commentjson as cjson
import os
import sys
import requests
import urllib3
import platform
import base64
from io import BytesIO
from PIL import Image

from tqdm import tqdm
import colorama
import asyncio
import aiohttp
from enum import Enum
import uuid

from ..presets import *
from ..index_func import *
from ..utils import *
from .. import shared
from ..config import retrieve_proxy, usage_limit
from modules import config
from .base_model import BaseLLMModel, ModelType


class ZhipuAI_Client(BaseLLMModel):
    def __init__(
        self,
        model_name,
        api_key,
        temperature=1.0,
        top_p=1.0,
        user_name=""
    ) -> None:
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            user=user_name
        )
        self.api_key = api_key
        self.need_api_key = True

    def get_answer_stream_iter(self):
        response = self._get_response(stream=True)
        if response is not None:
            for event in response.events():
                yield event.data
        else:
            yield STANDARD_ERROR_MSG + GENERAL_ERROR_MSG

    def get_answer_at_once(self):
        response = self._get_response()
        content = response["data"]["choices"][0]["content"]
        total_token_count = response["data"]["usage"]["total_tokens"]
        return content, total_token_count

    @shared.state.switching_api_key  # 在不开启多账号模式的时候，这个装饰器不会起作用
    def _get_response(self, stream=False):
        zhipuai.api_key = self.api_key
        history = self.history
        logging.debug(colorama.Fore.YELLOW + f"{history}" + colorama.Fore.RESET)

        if stream:
            timeout = TIMEOUT_STREAMING
        else:
            timeout = TIMEOUT_ALL

        # 如果有自定义的api-host，使用自定义host发送请求，否则使用默认设置发送请求
        if shared.state.completion_url != COMPLETION_URL:
            logging.info(f"使用自定义API URL: {shared.state.completion_url}")

        try:
            if not stream:
                response = zhipuai.model_api.invoke(
                    model=self.model_name,
                    prompt=self.history,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
            else:
                response = zhipuai.model_api.sse_invoke(
                    model=self.model_name,
                    prompt=self.history,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    incremental=False
                )
        except:
            return None
        return response

    def set_key(self, new_access_key):
        ret = super().set_key(new_access_key)
        return ret