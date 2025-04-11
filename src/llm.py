from llama_index.core.llms import LLM, CompletionResponse, LLMMetadata
from transformers import pipeline
from typing import List, Any
from pydantic import PrivateAttr
from src.config import Config

class HuggingFaceLLM(LLM):
    _generator: Any = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "_generator", pipeline(
            "text-generation",
            model=Config.MODEL_NAME,
            token=Config.HF_TOKEN,
            trust_remote_code=True
        ))

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        print(">>> PROMPT RECEIVED BY LLM >>>")
        print(prompt)       
        output = self._generator(prompt, max_new_tokens=256, num_return_sequences=1)
        gen_text = output[0]["generated_text"]
        cleaned_text = gen_text[len(prompt):].strip()
        return CompletionResponse(text=cleaned_text)

    def chat(self, messages: List[dict], **kwargs) -> CompletionResponse:
        prompt = "\n".join(m.get("content", "") for m in messages)
        return self.complete(prompt)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=Config.MODEL_NAME,
            context_window=2048,
            num_output=256,
            is_chat_model=False,
            is_function_calling_model=False,
        )
    def _as_query_component(self):
        return self


    def stream_chat(self, messages: List[dict], **kwargs):
        raise NotImplementedError("stream_chat not implemented.")

    def stream_complete(self, prompt: str, **kwargs):
        raise NotImplementedError("stream_complete not implemented.")

    async def achat(self, messages: List[dict], **kwargs):
        raise NotImplementedError("achat not implemented.")

    async def acomplete(self, prompt: str, **kwargs):
        raise NotImplementedError("acomplete not implemented.")

    async def astream_chat(self, messages: List[dict], **kwargs):
        raise NotImplementedError("astream_chat not implemented.")

    async def astream_complete(self, prompt: str, **kwargs):
        raise NotImplementedError("astream_complete not implemented.")
