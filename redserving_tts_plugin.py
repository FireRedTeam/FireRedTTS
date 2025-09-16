import datetime
import json
import os
import time
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple
from functools import wraps
import GPUtil
import traceback
import asyncio
from collections import Counter
import base64
from io import BytesIO

import yaml
from base_inference import AsyncDecorator, BaseInference, RpcResourceRequestType, Result

import torchaudio
from fireredtts.fireredtts import FireRedTTS
from fireredtts.fireredtts_redserving import FireRedTTS_Redserving


class Plugin(BaseInference):

    def set_init_args(self, args: str) -> None:
        # 从config.yaml 加载模型配置
        self.args = yaml.safe_load(args)
        self.logger.info(f"args is: {self.args}")
        model_path = self.args["model"]
        self.model = FireRedTTS_Redserving(
            model_config_path=os.path.join(model_path, "configs/model_config.json"),
            speaker_config_path=os.path.join(model_path, "configs/speaker_config.json"),
            pretrained_path=model_path,
        )

        # self.log_model_info()

    def pre_process(
        self, request_body: Dict[str, Any], chat_template_name: str = "default"
    ) -> Dict[str, Any]:
        speaker_id = request_body["voice"]
        text = request_body["input"]
        response_format = request_body.get("response_format", "mp3")
        return {
            "speaker_id": speaker_id,
            "text": text,
            "response_format": response_format,
        }

    @AsyncDecorator
    async def handle(self, request: str, mq_req: str) -> Any:
        # self.logger.info(f"Request type is {self.req_helper.get_req_type()}")

        # for mq
        topic_name = ""
        consumer_group_name = ""
        tag_name = ""
        if request is None:
            # rpc and mq request both are str and can convert to dict
            # only one of them is not None
            if isinstance(mq_req, tuple):
                request = mq_req[0]
                if len(mq_req) > 1:
                    mq_group = mq_req[1]  # (topic_name, group_name, tag_name)
                    if isinstance(mq_group, tuple) and len(mq_group) == 3:
                        topic_name, consumer_group_name, tag_name = mq_group
            else:
                request = mq_req
        elif self.req_helper.get_req_type() == RpcResourceRequestType:
            raw_request, _ = self.get_raw_request()
            request = raw_request.requestData["data"]

        request_body: Dict[str, Any] = json.loads(request)

        pre_outs = self.pre_process(request_body)

        start = time.time()
        out_wav = self.model.synthesize_online(
            speaker_id=pre_outs["speaker_id"], text=pre_outs["text"], lang="auto"
        )
        t1 = time.time()
        self.add_metrics_ms("tts_model_cost", (t1 - start) * 1000)

        if out_wav[0] is not None:
            self.logger.error(f"{out_wav[0]}")
        # out_wav_path_codec = os.path.join(".", "result.mp3")
        audio_bytes = BytesIO()
        if pre_outs["response_format"] == "wav":
            torchaudio.save(audio_bytes, out_wav[1], 24000, format="wav")
        elif pre_outs["response_format"] == "mp3":
            codec_config = torchaudio.io.CodecConfig(qscale=0)
            torchaudio.save(
                audio_bytes,
                out_wav[1],
                24000,
                format="mp3",
                backend="ffmpeg",
                compression=codec_config,
            )
        else:
            self.logger.error(
                f"doesn't support response_format as {pre_outs['response_format']}, only support return mp3, wav!"
            )
        t2 = time.time()
        self.add_metrics_ms("wav2mp3_cost", (t2 - t1) * 1000)

        return audio_bytes.getvalue()
