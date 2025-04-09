import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Literal, Optional

from ollama import AsyncClient, ChatResponse

from app.logger import logger  # Assuming a logger is set up in your app


# Class to handle OpenAI-style response formatting
class OpenAIResponse:
    def __init__(self, json_data):
        # 递归转换嵌套结构
        for key, value in json_data.items():
            if isinstance(value, dict):
                value = OpenAIResponse(value)
            elif isinstance(value, list):
                value = [
                    OpenAIResponse(item) if isinstance(item, dict) else item
                    for item in value
                ]
            setattr(self, key, value)

    def model_dump(self):
        def _to_dict(obj):
            if isinstance(obj, OpenAIResponse):
                return {k: _to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [_to_dict(item) for item in obj]
            else:
                return obj

        data = _to_dict(self)
        data["created_at"] = datetime.now().isoformat()  # 添加时间戳
        return data


def safe_print_response(response):
    print("=== 完整响应信息 ===")
    print(f"状态码: {response.status_code}")
    print("响应头:")
    print(response.headers)

    print("\n尝试解析响应体:")
    try:
        print("-> 作为JSON:")
        print(response.json())
    except ValueError:
        try:
            print("-> 作为文本:")
            print(response.text)
        except:
            print("-> 作为原始字节:")
            print(response.content)

    print(f"\n请求URL: {response.url}")
    print(f"耗时: {response.elapsed.total_seconds()}秒")
    print("==================\n")


# Main client class for interacting with Custom LLM Server
class OllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = AsyncClient(host=base_url)
        self.chat = Chat(self)

    """
    完整的ChatRespnse格式:
    {
        "model": "llama3-8b",
        "created_at": "2024-05-20T15:30:00Z",
        "done": true,
        "done_reason": "stop",
        "total_duration": 123456789,
        "load_duration": 5000000,
        "prompt_eval_count": 128,
        "prompt_eval_duration": 20000000,
        "eval_count": 256,
        "eval_duration": 100000000,
        "message": {
            "role": "assistant",
            "content": "content": "{ \"message\": { \"role\": \"user\", \"content\": \"量子力学是研究微观粒子运动规律的物理学分支。\"} }",
            "images": [
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
            "/path/to/physics_diagram.jpg"
            ],
            "tool_calls": [
            {
                "function": {
                "name": "calculate_energy",
                "arguments": {
                    "mass": 9.1e-31,
                    "velocity": 2.2e6
                }
                }
            }
            ]
        }
    }
    """

    async def send_request(
        self, model: str, messages, inferenceConfig: Optional[Dict] = None, **kwargs
    ):
        logger.info(f"请求数据: {json.dumps(messages, indent=2, ensure_ascii=False)}")

        response = await self.client.chat(
            model=model,
            messages=messages,
            stream=False,
            format=ChatResponse.model_json_schema(),
        )
        # logger.info(f"type(response): {type(response)}")
        # logger.info(f"响应数据: {response['message']['content']}")
        # logger.info(f"响应数据：{ChatResponse.model_validate_json(response.message.content)}")
        json_response = response.model_dump(exclude_unset=True)
        json_response["message"]["content"] = json.loads(
            response["message"]["content"].strip()
        )
        logger.info(f"响应数据: {json.dumps(json_response, indent=2, ensure_ascii=False)}")
        return json_response


# Chat interface class
class Chat:
    def __init__(self, client):
        logger.info(f"Chat client: {client}")
        self.completions = ChatCompletions(client)


# Core class handling chat completions functionality
class ChatCompletions:
    def __init__(self, client):
        self.client = client

    def _convert_openai_messages_to_deepseek_format(self, messages):
        # Convert OpenAI message format to deepseek message format
        ds_messages = []
        system_prompt = []
        for message in messages:
            if message.get("role") == "system":
                system_prompt = {"role": "system", "content": message.get("content")}
            elif message.get("role") == "user":
                ds_message = {
                    "role": message.get("role", "user"),
                    "content": f'{message.get("content")}\n',
                }
                ds_messages.append(ds_message)
            elif message.get("role") == "assistant":
                ds_message = {
                    "role": "assistant",
                    "content": message.get("content"),
                }
                ds_messages.append(ds_message)
            else:
                raise ValueError(f"Invalid role: {message.get('role')}")

        return system_prompt, ds_messages

    def _convert_deepseek_response_to_openai_format(self, deepseek_response):
        # Convert deepseek response format to OpenAI format
        content = ""
        if (
            deepseek_response.get("message", {})
            .get("content", {})
            .get("message", {})
            .get("content")
        ):
            content = deepseek_response["message"]["content"]["message"]["content"]
        if content == "":
            content = "."

        # Handle tool calls in response
        openai_tool_calls = []
        if deepseek_response.get("message", {}).get("tool_calls"):
            for content_item in deepseek_response["message"]["tool_calls"]:
                if content_item.get("function"):
                    function = content_item.function
                    openai_tool_call = {
                        "id": 0,
                        "type": "function",
                        "function": {
                            "name": function["name"],
                            "arguments": json.dumps(function["arguments"]),
                        },
                    }
                    openai_tool_calls.append(openai_tool_call)

        # Construct final OpenAI format response
        openai_format = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "created": int(time.time()),
            "object": "chat.completion",
            "system_fingerprint": None,
            "choices": [
                {
                    "finish_reason": deepseek_response.get("done_reason", "end_turn"),
                    "index": 0,
                    "message": {
                        "content": content,
                        "role": deepseek_response["message"].get("role", "assistant"),
                        "tool_calls": openai_tool_calls
                        if openai_tool_calls != []
                        else None,
                        "function_call": None,
                    },
                }
            ],
            "usage": {
                "completion_tokens": deepseek_response.get("eval_count", 0),
                "prompt_tokens": deepseek_response.get("prompt_eval_count", 0),
                "total_tokens": deepseek_response.get("eval_count", 0),
            },
        }
        # logger.info(f"openai_format: {json.dumps(openai_format, indent=2, ensure_ascii=False)}")
        return OpenAIResponse(openai_format)

    async def _invoke(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        tools: Optional[List[dict]] = None,
        tool_choice: Literal["none", "auto", "required"] = "auto",
        **kwargs,
    ) -> OpenAIResponse:
        # logger.info(f"openai format message: {json.dumps(messages, indent=2, ensure_ascii=False)}")
        """
        logger.info(f"len(ds_messages): {len(messages)}")
        for i, item in enumerate(messages):
            logger.info(f"ds_messages[{i}]: {json.dumps(item, indent=2, ensure_ascii=False)}")
        """
        response = await self.client.send_request(
            model=model,
            messages=messages,
            inferenceConfig={"temperature": temperature, "maxTokens": max_tokens},
        )
        openai_response = self._convert_deepseek_response_to_openai_format(response)
        logger.info(
            f"openai response format: {json.dumps(openai_response.model_dump(), indent=2, ensure_ascii=False)}"
        )
        return openai_response

    async def _invoke_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        tools: Optional[List[dict]] = None,
        tool_choice: Literal["none", "auto", "required"] = "auto",
        **kwargs,
    ) -> OpenAIResponse:
        # Streaming invocation of Bedrock model
        (
            system_prompt,
            bedrock_messages,
        ) = self._convert_openai_messages_to_custom_format(messages)
        """
        response = self.client.send_request_stream(
            model=model,
            system=system_prompt,
            messages=bedrock_messages,
            inferenceConfig={"temperature": temperature, "maxTokens": max_tokens},
            toolConfig={"tools": tools} if tools else None,
        )
        """

        return None

    async def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        stream: Optional[bool] = True,
        tools: Optional[List[dict]] = None,
        tool_choice: Literal["none", "auto", "required"] = "auto",
        **kwargs,
    ) -> OpenAIResponse:
        # Main entry point for chat completion
        _tools = []
        if stream:
            return await self._invoke_stream(
                model,
                messages,
                max_tokens,
                temperature,
                _tools,
                tool_choice,
                **kwargs,
            )
        else:
            return await self._invoke(
                model,
                messages,
                max_tokens,
                temperature,
                _tools,
                tool_choice,
                **kwargs,
            )
