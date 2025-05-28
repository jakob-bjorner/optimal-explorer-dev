import os
import json
import aiohttp
from dotenv import load_dotenv

load_dotenv()

async def llm_call(
        system="You are a helpful assistant", 
        user="What is the color of loneliness?",
        model="anthropic/claude-3.7-sonnet",
        temperature=0.7,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        repetition_penalty=1,
        top_k=0
    ):
    """Send a POST request to OpenRouter API with the provided system and user messages."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    api_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "repetition_penalty": repetition_penalty,
        "top_k": top_k,
    }
    
    async with aiohttp.ClientSession() as session:
        for attempt in range(10):
            try:
                async with session.post(api_url, headers=headers, data=json.dumps(payload)) as response:
                    if response is None:
                        print("API request failed: response is None.")
                        continue
                    if response.status == 200:
                        data = await response.json()
                        if "choices" not in data:
                            print("API request failed: 'choices' key not in response.")
                            continue
                        if not data["choices"]:
                            print("API request failed: 'choices' key is empty in response.")
                            continue
                        return data["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"API request failed. Retrying... ({attempt + 1}/5)")
                continue
        return ''