import os
import random
from time import sleep
from typing import Any, Union, Dict, List, Optional
import requests
from json import JSONDecodeError
import json

RATE_LIMIT_RETRY_DELAY = 60
RATE_LIMIT_RETRY_ATTEMPTS = 10


class Agent:

    def __init__(
        self,
        system: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        base_url: Optional[str] = None,
        api_keys: Optional[Union[str, List[str]]] = None,
        request_kwargs: Optional[Dict[str, Any]] = None,

    ):
        self.system = system
        if self.system is None:
            self.history = []
        else:
            self.history = [{"role": "system", "content": self.system}]
        self.model = model
        self.base_url = base_url


        if api_keys is not None:
            if isinstance(api_keys, str):
                api_keys = [api_keys]
        else:
            api_keys = [os.getenv("OPENAI_API_KEY", "EMPTY")]
        self.api_keys = api_keys

        self.request_kwargs = {
            "max_tokens": 2048,
            "temperature": 0.6,
            "top_p": 0.95,
            "seed": 100745534,
        }
        if request_kwargs is not None:
            self.request_kwargs.update(request_kwargs)

    def chat_completion_openai(
        self, messages, stream: bool = True, ttl: int = RATE_LIMIT_RETRY_ATTEMPTS
    ) -> str:
        response = ""
        if ttl >= 0:
            api_key = random.choice(self.api_keys)
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": stream,
                **self.request_kwargs,
            }
            
            url = f"{self.base_url or 'https://api.openai.com/v1'}/chat/completions"
            
            try:
                if stream:
                    with requests.post(url, headers=headers, json=payload, stream=True) as r:
                        r.raise_for_status()
                        for line in r.iter_lines():
                            if line:
                                line = line.decode('utf-8')
                                if line.startswith('data: '):
                                    line = line[6:]
                                    if line.strip() == '[DONE]':
                                        continue
                                    try:
                                        chunk = json.loads(line)
                                        if (
                                            not chunk['choices'][0]['finish_reason']
                                            and chunk['choices'][0]['delta'].get('content')
                                        ):
                                            content = chunk['choices'][0]['delta']['content']
                                            print(content, end="", flush=True)
                                            response += content
                                    except JSONDecodeError:
                                        continue
                    print()
                else:
                    r = requests.post(url, headers=headers, json=payload)
                    r.raise_for_status()
                    print("r.json()", r.json())
                    response = r.json()['choices'][0]['message']['content']
            except requests.exceptions.RequestException as e:
                if "rate_limit" in str(e).lower():
                    print(
                        f"Rate limit exceeded, waiting for {RATE_LIMIT_RETRY_DELAY} seconds and retrying... (ttl={ttl}): {e}"
                    )
                    sleep(RATE_LIMIT_RETRY_DELAY)
                    return self.chat_completion_openai(
                        messages, stream=False, ttl=ttl - 1
                    )
                
                if hasattr(e, 'response') and e.response is not None:
                    print(f"Error Status Code: {e.response.status_code}")
                    print(f"Error Message: {e.response.text}")
                    print(f"Request URL: {e.response.url}")
                    print(f"Request Headers: {e.response.request.headers}")
                else:
                    print(f"Network error (no response): {str(e)}")
                    if ttl > 0:
                        print(f"Retrying in {RATE_LIMIT_RETRY_DELAY} seconds... (ttl={ttl})")
                        sleep(RATE_LIMIT_RETRY_DELAY)
                        return self.chat_completion_openai(
                            messages, stream=False, ttl=ttl - 1
                        )
                
                payload_size = len(json.dumps(payload))
                print(f"Payload Size: {payload_size} bytes")
                raise
        return response


    def chat_completion(
        self, messages, stream: bool = True, ttl: int = RATE_LIMIT_RETRY_ATTEMPTS
    ) -> str:
        if self.base_url == "vllm":
            raise NotImplementedError
        return self.chat_completion_openai(messages, stream=stream, ttl=ttl)

    def __call__(self, prompt, stream: bool = True) -> Optional[str]:
        self.history.append({"role": "user", "content": prompt})
        try:
            response = self.chat_completion(self.history, stream=stream)
            assert response is not None
        except Exception as e:  # pylint: disable=W0718:broad-exception-caught
            self.history.pop()
            print(e)
            return None
        self.history.append({"role": "assistant", "content": response})
        return response

    def get_last_reply(self) -> Optional[str]:
        if self.history[-1]["role"] == "assistant":
            return self.history[-1]["content"]
        return None

    def forget_last_turn(self) -> None:
        while self.history[-1]["role"] != "user":
            self.history.pop()
        if self.history[-1]["role"] == "user":
            self.history.pop()
