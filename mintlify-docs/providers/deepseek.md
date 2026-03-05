
# Deepseek
https://deepseek.com/

**We support ALL Deepseek models, just set `deepseek/` as a prefix when sending completion requests**

## API Key
```python
# env variable
os.environ['DEEPSEEK_API_KEY']
```

## Sample Usage
```python
from litellm import completion
import os

os.environ['DEEPSEEK_API_KEY'] = ""
response = completion(
    model="deepseek/deepseek-chat", 
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
)
print(response)
```

## Sample Usage - Streaming
```python
from litellm import completion
import os

os.environ['DEEPSEEK_API_KEY'] = ""
response = completion(
    model="deepseek/deepseek-chat", 
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
    stream=True
)

for chunk in response:
    print(chunk)
```


## Supported Models - ALL Deepseek Models Supported!
We support ALL Deepseek models, just set `deepseek/` as a prefix when sending completion requests

| Model Name               | Function Call                                                                                                                                                      |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| deepseek-chat | `completion(model="deepseek/deepseek-chat", messages)` | 
| deepseek-coder | `completion(model="deepseek/deepseek-coder", messages)` | 


## Reasoning Models
| Model Name               | Function Call                                                                                                                                                      |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| deepseek-reasoner | `completion(model="deepseek/deepseek-reasoner", messages)` |

### Thinking / Reasoning Mode

Enable thinking mode for DeepSeek reasoner models using `thinking` or `reasoning_effort` parameters:

<Tabs>
<Tab title="thinking param">

```python
from litellm import completion
import os

os.environ['DEEPSEEK_API_KEY'] = ""

resp = completion(
    model="deepseek/deepseek-reasoner",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    thinking={"type": "enabled"},
)
print(resp.choices[0].message.reasoning_content)  # Model's reasoning
print(resp.choices[0].message.content)  # Final answer
```

</Tab>
<Tab title="reasoning_effort param">

```python
from litellm import completion
import os

os.environ['DEEPSEEK_API_KEY'] = ""

resp = completion(
    model="deepseek/deepseek-reasoner",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    reasoning_effort="medium",  # low, medium, high all map to thinking enabled
)
print(resp.choices[0].message.reasoning_content)  # Model's reasoning
print(resp.choices[0].message.content)  # Final answer
```

</Tab>
</Tabs>

<Note title="DeepSeek only supports `{&quot;type&quot;: &quot;enabled&quot;}` - unlike Anthropic, it doesn't support `budget_tokens`. Any `reasoning_effort` value other than `&quot;none&quot;` enables thinking mode.">

</Note>

### Basic Usage

<Tabs>
<Tab title="SDK">

```python
from litellm import completion
import os

os.environ['DEEPSEEK_API_KEY'] = ""
resp = completion(
    model="deepseek/deepseek-reasoner",
    messages=[{"role": "user", "content": "Tell me a joke."}],
)

print(
    resp.choices[0].message.reasoning_content
)
```

</Tab>
<Tab title="PROXY">

1. Setup config.yaml

```yaml
model_list:
  - model_name: deepseek-reasoner
    litellm_params:
        model: deepseek/deepseek-reasoner
        api_key: os.environ/DEEPSEEK_API_KEY
```

2. Run proxy

```bash
python litellm/proxy/main.py
```

3. Test it!

```bash
curl -L -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "model": "deepseek-reasoner",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Hi, how are you ?"
          }
        ]
      }
    ]
}'
```

</Tab>

</Tabs>