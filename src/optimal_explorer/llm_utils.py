import os
import json
import aiohttp
from dotenv import load_dotenv

from transformers import AutoTokenizer



load_dotenv()

def create_prompt_string(messages):
    prompt_string = ""
    for message in messages:
        prompt_string += f"role: {message['role']}\n\n{message['content']}\n\n"
    return prompt_string + "role: assistant"

async def llm_call(
        system="You are a helpful assistant",
        user="What is the color of loneliness?",
        model="anthropic/claude-3.7-sonnet",
        temperature=0.7,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        repetition_penalty=1,
        top_k=0,
        messages=None,
        get_everything=False, # option for getting not just the content, for reasoning logging.
        reasoning_effort=None,
        url=None
    ):
    """Send a POST request to OpenRouter API with the provided system and user messages."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if messages is None:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    payload = {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "repetition_penalty": repetition_penalty,
        "top_k": top_k,
    }
    if "openai/o3" in model:
        api_url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}","Content-Type": "application/json",}
        payload = {
            "model": 'o3',
        }
    elif url is None:
        api_url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
    else:
        api_url = url # this to support calling local model for base model testing.
        headers = None
        payload['max_tokens'] = 2000 # this is the max number of generated/completion tokens
        payload['top_k'] = -1

    if not model.lower().endswith('base'):
        payload['messages'] = messages
    else:
        api_url = api_url.replace("/chat/completions", "/completions")
        if user == "What is the color of loneliness?":
            # some automatic way of converting system, user, and assistant into a single string
            payload['prompt'] = create_prompt_string(messages)
        else:
            payload['prompt'] = user
        payload['stop'] = ["role:"]

    if reasoning_effort:
        payload["reasoning"] = {
            "effort": reasoning_effort,
        }
    
    
    if 'qwen3' in model.lower():
        tokenizer = AutoTokenizer.from_pretrained(model)
        end_think_text = "\nConsidering the limited time by the user, I have to give my response now.\n</think>.\n\n"
    
    async with aiohttp.ClientSession() as session:
        for attempt in range(5):
            try:
                async with session.post(api_url, headers=headers, json=payload) as response:
                    if response is None:
                        print("API request failed: response is None.")
                        continue
                    elif response.status != 200:
                        import ipdb; ipdb.set_trace()
                        print("API request failed: status "+ str(await response.json()))
                        continue
                    elif response.status == 200:
                        data = await response.json()
                        if "choices" not in data:
                            print("API request failed: 'choices' key not in response.")
                            continue
                        if not data["choices"]:
                            print("API request failed: 'choices' key is empty in response.")
                            continue
                        if '</think>' not in data["choices"][0]["message"]["content"]:
                            messages = payload['messages'] + [{"role": "assistant", "content": data["choices"][0]["message"]["content"] + end_think_text}]
                            prompt_chat_str = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)[:-13]
                            generation_url = api_url.replace("v1/chat/completions", "generate")
                            payload2 = payload.copy()
                            payload2["text"] = prompt_chat_str
                            payload2["max_new_tokens"] = 2000

                            async with session.post(generation_url, headers=headers, json=payload2) as response:
                                if response is None:
                                    print("API request failed: response is None.")
                                    continue
                                elif response.status != 200:
                                    import ipdb; ipdb.set_trace()
                                    print("API request failed: status "+ str(await response.json()))
                                    continue
                                elif response.status == 200:
                                    data2 = await response.json()
                                    if "text" not in data2:
                                        print("API request failed: 'text' key not in response.")
                                        continue
                                    if not data2["text"]:
                                        print("API request failed: 'choices' key is empty in response.")
                                        continue
                                    data['usage']['completion_tokens'] = data['usage']['completion_tokens'] + data2['meta_info']['completion_tokens']
                                    data['usage']['total_tokens'] = data['usage']['total_tokens'] +data2['meta_info']['completion_tokens']
                                    data["choices"][0]["message"]["content"] = data["choices"][0]["message"]["content"] + end_think_text + data2['text']
                        if get_everything:
                            return data
                        else:
                            return data["choices"][0]["message"]["content"]
                        
            except Exception as e:
                print(f"API request failed. Retrying... ({attempt + 1}/5)")
                continue
        return ''