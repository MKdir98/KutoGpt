from g4f.client import Client
from g4f.models import HuggingChat
import json
import traceback
import enum
import functools
import logging
import os
import time
from pathlib import Path
from typing import Callable, Optional, ParamSpec, TypeVar, Sequence

import tiktoken
import yaml
from pydantic import SecretStr

from ._openai_base import BaseOpenAIChatProvider, BaseOpenAIEmbeddingProvider
from forge.models.config import Configurable, UserConfigurable
from forge.models.providers import Embedding

from .schema import (
    AssistantChatMessage,
    AssistantToolCallDict,
    ChatMessage,
    ChatModelInfo,
    ChatModelResponse,
    CompletionModelFunction,
    EmbeddingModelInfo,
    EmbeddingModelResponse,
    ModelProviderBudget,
    ModelProviderConfiguration,
    ModelProviderCredentials,
    ModelProviderName,
    BaseChatModelProvider,
    ModelProviderService,
    ModelProviderSettings,
    ModelTokenizer,
)

_T = TypeVar("_T")
_P = ParamSpec("_P")

Gpt4FreeEmbeddingParser = Callable[[Embedding], Embedding]
Gpt4FreeChatParser = Callable[[str], dict]


class Gpt4FreeModelName(str, enum.Enum):
    EMBEDDING_v2 = "text-embedding-ada-002"
    EMBEDDING_v3_S = "text-embedding-3-small"
    EMBEDDING_v3_L = "text-embedding-3-large"

    GPT3_v1 = "gpt-3.5-turbo-0301"
    GPT3_v2 = "gpt-3.5-turbo-0613"
    GPT3_v2_16k = "gpt-3.5-turbo-16k-0613"
    GPT3_v3 = "gpt-3.5-turbo-1106"
    GPT3_v4 = "gpt-3.5-turbo-0125"
    GPT3_ROLLING = "gpt-3.5-turbo"
    GPT3_ROLLING_16k = "gpt-3.5-turbo-16k"
    GPT3 = GPT3_ROLLING
    GPT3_16k = GPT3_ROLLING_16k

    GPT4_v1 = "gpt-4-0314"
    GPT4_v1_32k = "gpt-4-32k-0314"
    GPT4_v2 = "gpt-4-0613"
    GPT4_v2_32k = "gpt-4-32k-0613"
    GPT4_v3 = "gpt-4-1106-preview"
    GPT4_v3_VISION = "gpt-4-1106-vision-preview"
    GPT4_v4 = "gpt-4-0125-preview"
    GPT4_v5 = "gpt-4-turbo-2024-04-09"
    GPT4_ROLLING = "gpt-4"
    GPT4_ROLLING_32k = "gpt-4-32k"
    GPT4_TURBO = "gpt-4-turbo"
    GPT4_TURBO_PREVIEW = "gpt-4-turbo-preview"
    GPT4_VISION = "gpt-4-vision-preview"
    GPT4 = GPT4_ROLLING
    GPT4_32k = GPT4_ROLLING_32k


GPT_4_FREE_EMBEDDING_MODELS = {
    info.name: info
    for info in [
        EmbeddingModelInfo(
            name=Gpt4FreeModelName.EMBEDDING_v2,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.0001 / 1000,
            max_tokens=8191,
            embedding_dimensions=1536,
        ),
        EmbeddingModelInfo(
            name=Gpt4FreeModelName.EMBEDDING_v3_S,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.00002 / 1000,
            max_tokens=8191,
            embedding_dimensions=1536,
        ),
        EmbeddingModelInfo(
            name=Gpt4FreeModelName.EMBEDDING_v3_L,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.00013 / 1000,
            max_tokens=8191,
            embedding_dimensions=3072,
        ),
    ]
}


GPT_4_FREE_CHAT_MODELS = {
    info.name: info
    for info in [
        ChatModelInfo(
            name=Gpt4FreeModelName.GPT3_v1,
            provider_name=ModelProviderName.GPT4FREE,
            prompt_token_cost=0.0015 / 1000,
            completion_token_cost=0.002 / 1000,
            max_tokens=4096,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=Gpt4FreeModelName.GPT3_v2_16k,
            provider_name=ModelProviderName.GPT4FREE,
            prompt_token_cost=0.003 / 1000,
            completion_token_cost=0.004 / 1000,
            max_tokens=16384,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=Gpt4FreeModelName.GPT3_v3,
            provider_name=ModelProviderName.GPT4FREE,
            prompt_token_cost=0.001 / 1000,
            completion_token_cost=0.002 / 1000,
            max_tokens=16384,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=Gpt4FreeModelName.GPT3_v4,
            provider_name=ModelProviderName.GPT4FREE,
            prompt_token_cost=0.0005 / 1000,
            completion_token_cost=0.0015 / 1000,
            max_tokens=16384,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=Gpt4FreeModelName.GPT4_v1,
            provider_name=ModelProviderName.GPT4FREE,
            prompt_token_cost=0.03 / 1000,
            completion_token_cost=0.06 / 1000,
            max_tokens=8191,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=Gpt4FreeModelName.GPT4_v1_32k,
            provider_name=ModelProviderName.GPT4FREE,
            prompt_token_cost=0.06 / 1000,
            completion_token_cost=0.12 / 1000,
            max_tokens=32768,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=Gpt4FreeModelName.GPT4_TURBO,
            provider_name=ModelProviderName.GPT4FREE,
            prompt_token_cost=0.01 / 1000,
            completion_token_cost=0.03 / 1000,
            max_tokens=128000,
            has_function_call_api=True,
        ),
    ]
}
GPT_4_FREE_CHAT_MODELS = {
    info.name: info
    for info in [
        ChatModelInfo(
            name=Gpt4FreeModelName.GPT3,
            service=ModelProviderService.CHAT,
            provider_name=ModelProviderName.GPT4FREE,
            prompt_token_cost=0.0015 / 1000,
            completion_token_cost=0.002 / 1000,
            max_tokens=4096,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=Gpt4FreeModelName.GPT3_16k,
            service=ModelProviderService.CHAT,
            provider_name=ModelProviderName.GPT4FREE,
            prompt_token_cost=0.003 / 1000,
            completion_token_cost=0.004 / 1000,
            max_tokens=16384,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=Gpt4FreeModelName.GPT3_v3,
            service=ModelProviderService.CHAT,
            provider_name=ModelProviderName.GPT4FREE,
            prompt_token_cost=0.001 / 1000,
            completion_token_cost=0.002 / 1000,
            max_tokens=16384,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=Gpt4FreeModelName.GPT4,
            service=ModelProviderService.CHAT,
            provider_name=ModelProviderName.GPT4FREE,
            prompt_token_cost=0.03 / 1000,
            completion_token_cost=0.06 / 1000,
            max_tokens=8191,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=Gpt4FreeModelName.GPT4_32k,
            service=ModelProviderService.CHAT,
            provider_name=ModelProviderName.GPT4FREE,
            prompt_token_cost=0.06 / 1000,
            completion_token_cost=0.12 / 1000,
            max_tokens=32768,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=Gpt4FreeModelName.GPT4_v3,
            service=ModelProviderService.CHAT,
            provider_name=ModelProviderName.GPT4FREE,
            prompt_token_cost=0.01 / 1000,
            completion_token_cost=0.03 / 1000,
            max_tokens=128000,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=Gpt4FreeModelName.GPT4_TURBO,
            service=ModelProviderService.CHAT,
            provider_name=ModelProviderName.GPT4FREE,
            prompt_token_cost=0.01 / 1000,
            completion_token_cost=0.03 / 1000,
            max_tokens=128000,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=Gpt4FreeModelName.GPT3_v4,
            service=ModelProviderService.CHAT,
            provider_name=ModelProviderName.GPT4FREE,
            prompt_token_cost=0.0005 / 1000,
            completion_token_cost=0.0015 / 1000,
            max_tokens=16384,
            has_function_call_api=True,
        ),
    ]
}
# Copy entries for models with equivalent specs
chat_model_mapping = {
    Gpt4FreeModelName.GPT3: [Gpt4FreeModelName.GPT3_v1, Gpt4FreeModelName.GPT3_v2],
    Gpt4FreeModelName.GPT3_16k: [Gpt4FreeModelName.GPT3_v2_16k],
    Gpt4FreeModelName.GPT4: [Gpt4FreeModelName.GPT4_v1, Gpt4FreeModelName.GPT4_v2],
    Gpt4FreeModelName.GPT4_32k: [
        Gpt4FreeModelName.GPT4_v1_32k,
        Gpt4FreeModelName.GPT4_v2_32k,
    ],
    Gpt4FreeModelName.GPT4_TURBO: [
        Gpt4FreeModelName.GPT4_v3,
        Gpt4FreeModelName.GPT4_VISION,
    ],
}
for base, copies in chat_model_mapping.items():
    for copy in copies:
        copy_info = ChatModelInfo(**GPT_4_FREE_CHAT_MODELS[base].__dict__)
        copy_info.name = copy
        GPT_4_FREE_CHAT_MODELS[copy] = copy_info
        if copy.endswith(("-0301", "-0314")):
            copy_info.has_function_call_api = False


GPT_4_FREE_MODELS = {
    **GPT_4_FREE_CHAT_MODELS,
    **GPT_4_FREE_EMBEDDING_MODELS,
}


class Gpt4FreeConfiguration(ModelProviderConfiguration):
    fix_failed_parse_tries: int = UserConfigurable(3)
    pass


class Gpt4FreeCredentials(ModelProviderCredentials):
    """Credentials for Gpt4Free."""

    email: SecretStr = UserConfigurable(from_env="GPT_4_FREE_USERNAME")
    password: SecretStr = UserConfigurable(from_env="GPT_4_FREE_PASSWORD")
    api_base: Optional[SecretStr] = UserConfigurable(
        default=None, from_env="GPT_4_FREE_API_BASE_URL"
    )
    organization: Optional[SecretStr] = UserConfigurable(
        from_env="GPT_4_FREE_ORGANIZATION"
    )

    api_type: str = UserConfigurable(
        default="",
        from_env=lambda: (
            "azure"
            if os.getenv("USE_AZURE") == "True"
            else os.getenv("GPT_4_FREE_API_TYPE")
        ),
    )
    api_version: str = UserConfigurable("", from_env="GPT_4_FREE_API_VERSION")
    azure_model_to_deploy_id_map: Optional[dict[str, str]] = None

    def get_api_access_kwargs(self, model: str = "") -> dict[str, str]:
        credentials = {k: v for k, v in self.unmasked().items() if type(v) is str}
        if self.api_type == "azure" and model:
            azure_credentials = self._get_azure_access_kwargs(model)
            credentials.update(azure_credentials)
        return credentials

    def load_azure_config(self, config_file: Path) -> None:
        with open(config_file) as file:
            config_params = yaml.load(file, Loader=yaml.FullLoader) or {}

        try:
            assert (
                azure_api_base := config_params.get("azure_api_base", "")
            ) != "", "Azure API base URL not set"
            assert config_params.get(
                "azure_model_map", {}
            ), "Azure model->deployment_id map is empty"
        except AssertionError as e:
            raise ValueError(*e.args)

        self.api_base = SecretStr(azure_api_base)
        self.api_type = config_params.get("azure_api_type", "azure")
        self.api_version = config_params.get("azure_api_version", "")
        self.azure_model_to_deploy_id_map = config_params.get("azure_model_map")

    def _get_azure_access_kwargs(self, model: str) -> dict[str, str]:
        """Get the kwargs for the Azure API."""

        if not self.azure_model_to_deploy_id_map:
            raise ValueError("Azure model deployment map not configured")

        if model not in self.azure_model_to_deploy_id_map:
            raise ValueError(f"No Azure deployment ID configured for model '{model}'")
        deployment_id = self.azure_model_to_deploy_id_map[model]

        if model in GPT_4_FREE_EMBEDDING_MODELS:
            return {"engine": deployment_id}
        else:
            return {"deployment_id": deployment_id}


class GPT4FreeProvider(
        BaseChatModelProvider[Gpt4FreeModelName, ModelProviderSettings]
):
    async def get_available_models(self) -> list[ChatModelInfo]:
        return list(GPT_4_FREE_CHAT_MODELS.values())

    async def get_available_chat_models(
        self,
    ) -> Sequence[ChatModelInfo[Gpt4FreeModelName]]:
        return list(GPT_4_FREE_CHAT_MODELS.values())

    default_settings = ModelProviderSettings(
        name="GPT_4_FREE_provider",
        description="Provides access to Gpt4Free's API.",
        configuration=ModelProviderConfiguration(
            retries_per_request=7,
        ),
        budget=ModelProviderBudget(),
    )

    _budget: ModelProviderBudget


    def __init__(
        self,
        settings: Optional[ModelProviderSettings] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super(GPT4FreeProvider, self).__init__(settings=settings, logger=logger)
        self._budget = self._settings.budget or ModelProviderBudget()

    def get_token_limit(self, model_name: str) -> int:
        """Get the token limit for a given model."""
        return GPT_4_FREE_MODELS[model_name].max_tokens

    def get_remaining_budget(self) -> float:
        """Get the remaining budget."""
        return self._budget.remaining_budget

    @classmethod
    def get_tokenizer(cls, model_name: Gpt4FreeModelName) -> ModelTokenizer:
        return tiktoken.encoding_for_model(model_name)

    @classmethod
    def count_tokens(cls, text: str, model_name: Gpt4FreeModelName) -> int:
        encoding = cls.get_tokenizer(model_name)
        return len(encoding.encode(text))

    @classmethod
    def count_message_tokens(
        cls,
        messages: ChatMessage | list[ChatMessage],
        model_name: Gpt4FreeModelName,
    ) -> int:
        if isinstance(messages, ChatMessage):
            messages = [messages]

        if model_name.startswith("gpt-3.5-turbo"):
            tokens_per_message = (
                4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            )
            tokens_per_name = -1  # if there's a name, the role is omitted
            encoding_model = "gpt-3.5-turbo"
        elif model_name.startswith("gpt-4"):
            tokens_per_message = 3
            tokens_per_name = 1
            encoding_model = "gpt-4"
        else:
            raise NotImplementedError(
                f"count_message_tokens() is not implemented for model {model_name}.\n"
                " See https://github.com/GPT_4_FREE/openai-python/blob/main/chatml.md for"
                " information on how messages are converted to tokens."
            )
        try:
            encoding = tiktoken.encoding_for_model(encoding_model)
        except KeyError:
            cls._logger.warning(
                f"Model {model_name} not found. Defaulting to cl100k_base encoding."
            )
            encoding = tiktoken.get_encoding("cl100k_base")

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.dict().items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    async def create_chat_completion(
        self,
        model_prompt: list[ChatMessage],
        model_name: Gpt4FreeModelName,
        completion_parser: Callable[[AssistantChatMessage], _T] = lambda _: None,
        functions: Optional[list[CompletionModelFunction]] = None,
        **kwargs,
    ) -> ChatModelResponse[_T]:
        """Create a completion using the Gpt4Free API."""
        completion_kwargs = self._get_completion_kwargs(model_name, functions, **kwargs)
        tool_calls_compat_mode = functions and "tools" not in completion_kwargs
        if "messages" in completion_kwargs:
            model_prompt += completion_kwargs["messages"]
            del completion_kwargs["messages"]

        attempts = 0
        while True:
            try:
                json_messages = []
                for message in model_prompt:
                    json_message = {
                        "role": message.role,
                        "content": message.content,
                    }
                    json_messages.append(json_message)
                # if not "The user is going to give you a text" in json_messages[0]['content']:
                #     additional_prompt = "additional promt tip: "\
                #             "- don't use \\ to escape _ or ] or [ in your resposne \n" \
                #             "- All path file should be relative. You can't create files outside of workspace folder. If you want to create somthing outside of workspace ask user to do it \n" \
                #             "- your json reponse structure should be like example. don't add field by yourself \n" \
                #             "- You can not say anything after and before you json response. What should I see should be a json message, nothing else \n" \
                #             "- Your call structure should be like example call. no more fields \n" \
                #             "- If you should say a command you can not send it with empty content \n" \
                #             "- Command part should not be empty \n" \
                #             "- Don't bring texts after and before the json. just send json message \n" \
                #             "- Put json reponse in json block code always \n"
                #     json_messages.insert(1, {'role': 'SYSYTEM', 'content': additional_prompt})

                import nest_asyncio

                nest_asyncio.apply()
                client = Client()
                # print(json_messages)
                gptResponse = client.chat.completions.create(
                    model="llama3-70b", messages=json_messages,
                    provider=HuggingChat
                )
                response = str(gptResponse.choices[0].message.content)
                # print("response")
                print(response)
                response = response.replace("\x00", "")
                if "```" in response:
                    response_text = response.replace("```", "")
                    if "{" in response_text and "[" in response_text:
                        json_obj_start = min(
                            response_text.index("{"), response_text.index("[")
                        )
                    elif "{" in response_text:
                        json_obj_start = response_text.index("{")
                    else:
                        json_obj_start = response_text.index("[")
                    response_text = response_text[json_obj_start:]
                else:
                    response_text = response
                response_text = response_text.replace("\\_", "_").replace("\n", "")
                # response_text = response
                # matches = re.findall(r'```(.*?)```', response, re.DOTALL)
                # if matches:
                #     response_text = '\n'.join(matches[0].split('\n')[1:])
                # else:
                #     response_text = response
                response_args = {
                    "model_info": GPT_4_FREE_CHAT_MODELS[model_name],
                    "prompt_tokens_used": 0,
                    "completion_tokens_used": 0,
                }
                try:
                    attempts += 1
                    tool_calls = None
                    if (
                        "Your job is to respond to a user-defined task"
                        in model_prompt[0].content
                    ):
                        tool_calls = _tool_calls_compat_extract_calls(response_text)
                    response = AssistantChatMessage(
                        role="assistant",
                        content=response_text,
                        # tool_calls= self.string_to_tool_calls(response_text)
                        tool_calls=tool_calls,
                    )
                    parsed_response = completion_parser(response)
                    return ChatModelResponse(
                        response=response,
                        parsed_result=parsed_response,
                        **response_args,
                    )
                except Exception as e:
                    traceback.print_exc()
                    self._logger.warning(f"Parsing attempt #{attempts} failed: {e}")
                    self._logger.debug(
                        f"Parsing failed on response: '''{response_text}'''"
                    )
                    if attempts < self._configuration.retries_per_request:
                        model_prompt.append(
                            ChatMessage.system(
                                f"your json is wrong. please fix it and send it again. just send json (with no saying sorry or somehitng like that) ERROR PARSING YOUR RESPONSE:\n\n{e}"
                            )
                        )
                    else:
                        raise
            except Exception as ex:
                traceback.print_exc()
                return await self.create_chat_completion(
                    model_prompt,
                    model_name,
                    completion_parser,
                    functions,
                    **kwargs,
                )

    async def create_embedding(
        self,
        text: str,
        model_name: Gpt4FreeModelName,
        embedding_parser: Callable[[Embedding], Embedding],
        **kwargs,
    ) -> EmbeddingModelResponse:
        """Create an embedding using the Gpt4Free API."""
        embedding_kwargs = self._get_embedding_kwargs(model_name, **kwargs)
        response = await self._create_embedding(text=text, **embedding_kwargs)

        response_args = {
            "model_info": GPT_4_FREE_EMBEDDING_MODELS[model_name],
            "prompt_tokens_used": response.usage.prompt_tokens,
            "completion_tokens_used": response.usage.completion_tokens,
        }
        response = EmbeddingModelResponse(
            **response_args,
            embedding=embedding_parser(response.embeddings[0]),
        )
        self._budget.update_usage_and_cost(response)
        return response

    def _get_completion_kwargs(
        self,
        model_name: Gpt4FreeModelName,
        functions: Optional[list[CompletionModelFunction]] = None,
        **kwargs,
    ) -> dict:
        """Get kwargs for completion API call.

        Args:
            model: The model to use.
            kwargs: Keyword arguments to override the default values.

        Returns:
            The kwargs for the chat API call.

        """
        completion_kwargs = {
            "model": model_name,
            **kwargs,
            # **self._credentials.get_api_access_kwargs(model_name),
        }

        if functions:
            if GPT_4_FREE_CHAT_MODELS[model_name].has_function_call_api:
                completion_kwargs["tools"] = [
                    {"type": "function", "function": f.schema} for f in functions
                ]
                if len(functions) == 1:
                    # force the model to call the only specified function
                    completion_kwargs["tool_choice"] = {
                        "type": "function",
                        "function": {"name": functions[0].name},
                    }
            else:
                # Provide compatibility with older models
                _functions_compat_fix_kwargs(functions, completion_kwargs)

        if extra_headers := self._configuration.extra_request_headers:
            if completion_kwargs.get("headers"):
                completion_kwargs["headers"].update(extra_headers)
            else:
                completion_kwargs["headers"] = extra_headers.copy()

        return completion_kwargs

    def _get_embedding_kwargs(
        self,
        model_name: Gpt4FreeModelName,
        **kwargs,
    ) -> dict:
        """Get kwargs for embedding API call.

        Args:
            model: The model to use.
            kwargs: Keyword arguments to override the default values.

        Returns:
            The kwargs for the embedding API call.

        """
        embedding_kwargs = {
            "model": model_name,
            **kwargs,
            **self._credentials.unmasked(),
        }

        if extra_headers := self._configuration.extra_request_headers:
            if embedding_kwargs.get("headers"):
                embedding_kwargs["headers"].update(extra_headers)
            else:
                embedding_kwargs["headers"] = extra_headers.copy()

        return embedding_kwargs

    def __repr__(self):
        return "Gpt4FreeProvider()"


# async def _create_embedding(text: str, *_, **kwargs) -> GPT_4_FREE.Embedding:
#     """Embed text using the Gpt4Free API.

#     Args:
#         text str: The text to embed.
#         model str: The name of the model to use.

#     Returns:
#         str: The embedding.
#     """
#     return await GPT_4_FREE.Embedding.acreate(
#         input=[text],
#         **kwargs,
#     )


# async def _create_chat_completion(
#     messages: list[ChatMessage], *_, **kwargs
# ) -> GPT_4_FREE.Completion:
#     """Create a chat completion using the Gpt4Free API.

#     Args:
#         messages: The prompt to use.

#     Returns:
#         The completion.
#     """
#     raw_messages = [
#         message.dict(include={"role", "content", "tool_calls", "name"})
#         for message in messages
#     ]
#     return await GPT_4_FREE.ChatCompletion.acreate(
#         messages=raw_messages,
#         **kwargs,
#     )


class _Gpt4FreeRetryHandler:
    """Retry Handler for Gpt4Free API call.

    Args:
        num_retries int: Number of retries. Defaults to 10.
        backoff_base float: Base for exponential backoff. Defaults to 2.
        warn_user bool: Whether to warn the user. Defaults to True.
    """

    _retry_limit_msg = "Error: Reached rate limit, passing..."
    _api_key_error_msg = (
        "Please double check that you have setup a PAID Gpt4Free API Account. You can "
        "read more here: https://docs.agpt.co/setup/#getting-an-GPT_4_FREE-api-key"
    )
    _backoff_msg = "Error: API Bad gateway. Waiting {backoff} seconds..."

    def __init__(
        self,
        logger: logging.Logger,
        num_retries: int = 10,
        backoff_base: float = 2.0,
        warn_user: bool = True,
    ):
        self._logger = logger
        self._num_retries = num_retries
        self._backoff_base = backoff_base
        self._warn_user = warn_user

    def _log_rate_limit_error(self) -> None:
        self._logger.debug(self._retry_limit_msg)
        if self._warn_user:
            self._logger.warning(self._api_key_error_msg)
            self._warn_user = False

    def _backoff(self, attempt: int) -> None:
        backoff = self._backoff_base ** (attempt + 2)
        self._logger.debug(self._backoff_msg.format(backoff=backoff))
        time.sleep(backoff)

    def __call__(self, func: Callable[_P, _T]) -> Callable[_P, _T]:
        @functools.wraps(func)
        async def _wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            num_attempts = self._num_retries + 1  # +1 for the first attempt
            for attempt in range(1, num_attempts + 1):
                try:
                    return await func(*args, **kwargs)

                except RateLimitError:
                    if attempt == num_attempts:
                        raise
                    self._log_rate_limit_error()

                except APIError as e:
                    if (e.http_status != 502) or (attempt == num_attempts):
                        raise

                self._backoff(attempt)

        return _wrapped


def format_function_specs_as_typescript_ns(
    functions: list[CompletionModelFunction],
) -> str:
    """Returns a function signature block in the format used by Gpt4Free internally:
    https://community.GPT_4_FREE.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/18

    For use with `count_tokens` to determine token usage of provided functions.

    Example:
    ```ts
    namespace functions {

    // Get the current weather in a given location
    type get_current_weather = (_: {
    // The city and state, e.g. San Francisco, CA
    location: string,
    unit?: "celsius" | "fahrenheit",
    }) => any;

    } // namespace functions
    ```
    """

    return (
        "namespace functions {\n\n"
        + "\n\n".join(format_GPT_4_FREE_function_for_prompt(f) for f in functions)
        + "\n\n} // namespace functions"
    )


def format_GPT_4_FREE_function_for_prompt(func: CompletionModelFunction) -> str:
    """Returns the function formatted similarly to the way Gpt4Free does it internally:
    https://community.GPT_4_FREE.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/18

    Example:
    ```ts
    // Get the current weather in a given location
    type get_current_weather = (_: {
    // The city and state, e.g. San Francisco, CA
    location: string,
    unit?: "celsius" | "fahrenheit",
    }) => any;
    ```
    """

    def param_signature(name: str, spec: JSONSchema) -> str:
        return (
            f"// {spec.description}\n" if spec.description else ""
        ) + f"{name}{'' if spec.required else '?'}: {spec.typescript_type},"

    return "\n".join(
        [
            f"// {func.description}",
            f"type {func.name} = (_ :{{",
            *[param_signature(name, p) for name, p in func.parameters.items()],
            "}) => any;",
        ]
    )


def count_GPT_4_FREE_functions_tokens(
    functions: list[CompletionModelFunction], count_tokens: Callable[[str], int]
) -> int:
    """Returns the number of tokens taken up by a set of function definitions

    Reference: https://community.GPT_4_FREE.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/18  # noqa: E501
    """
    return count_tokens(
        "# Tools\n\n"
        "## functions\n\n"
        f"{format_function_specs_as_typescript_ns(functions)}"
    )


def _functions_compat_fix_kwargs(
    functions: list[CompletionModelFunction],
    completion_kwargs: dict,
):
    function_definitions = format_function_specs_as_typescript_ns(functions)
    function_call_schema = JSONSchema(
        type=JSONSchema.Type.OBJECT,
        properties={
            "name": JSONSchema(
                description="The name of the function to call",
                enum=[f.name for f in functions],
                required=True,
            ),
            "arguments": JSONSchema(
                description="The arguments for the function call",
                type=JSONSchema.Type.OBJECT,
                required=True,
            ),
        },
    )
    tool_calls_schema = JSONSchema(
        type=JSONSchema.Type.ARRAY,
        items=JSONSchema(
            type=JSONSchema.Type.OBJECT,
            properties={
                "type": JSONSchema(
                    type=JSONSchema.Type.STRING,
                    enum=["function"],
                ),
                "function": function_call_schema,
            },
        ),
    )
    completion_kwargs["messages"] = [
        ChatMessage.system(
            "# tool usage instructions\n\n"
            "Specify a '```tool_calls' block in your response,"
            " with a valid JSON object that adheres to the following schema:\n\n"
            f"{tool_calls_schema.to_dict()}\n\n"
            "Specify any tools that you need to use through this JSON object.\n\n"
            "Put the tool_calls block at the end of your response"
            " and include its fences if it is not the only content.\n\n"
            "## functions\n\n"
            "For the function call itself, use one of the following"
            f" functions:\n\n{function_definitions}"
        ),
    ]


def string_to_tool_calls(input_string: str) -> list[AssistantToolCallDict]:
    prefix = "```json"
    suffix = "```"
    if input_string.startswith(prefix) and input_string.endswith(suffix):
        input_string = input_string[len(prefix) : -len(suffix)]
    tool_calls_list = json.loads(input_string)
    for tool_call in tool_calls_list:
        if isinstance(tool_call, dict) and "function" in tool_call.keys():
            tool_call["function"]["arguments"] = tool_call["function"]["arguments"]

    return tool_calls_list


def _tool_calls_compat_extract_calls(response: str) -> list[AssistantToolCallDict]:
    # import json
    # import re

    logging.debug(f"Trying to extract tool calls from response:\n{response}")

    tool_calls: list[AssistantToolCallDict] = string_to_tool_calls(response)
    # if response[0] == "[":
    # else:
    #     block = re.search(r"```(?:tool_calls)?\n(.*)\n```\s*$", response, re.DOTALL)
    #     if not block:
    #         raise ValueError("Could not find tool calls block in response")
    #     tool_calls: list[AssistantToolCallDict] = json.loads(block.group(1))

    # for t in tool_calls:
    #     t["function"]["arguments"] = str(t["function"]["arguments"])  # HACK

    return tool_calls
