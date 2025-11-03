# Quickly Evaluate LiteLLM

LiteLLM provides access to various providers, all in openai compatible endpoints. Users can track their personal, team's usage. 

In this example, we are using LiteLLM CLI. See deployment options [here](./deploy.md)

:::info
Required env vars
1. LITELLM_MASTER_KEY="sk-1234"
2. LITELLM_SALT_KEY="sk-8u83714" # secure hash
3. DATABASE_URL="postgres://..." # 
4. OPENAI_API_KEY="..."
:::

## Setup config.yml
```
model_list:
  - model_name: bedrock-claude-3-5-sonnet
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: os.environ/AWS_REGION_NAME

  - model_name: gemini-2.5-pro
    litellm_params:
      model: gemini/gemini-2.5-pro
      api_key: os.environ/GEMINI_API_KEY

  - model_name: gpt-5-codex
    litellm_params:
      model: azure/gpt-5-codex
      api_base: os.environ/AZURE_API_BASE
      api_key: os.environ/AZURE_API_KEY

  - model_name: gpt-5-pro
    litellm_params:
      model: openai/gpt-5-pro
      api_key: os.environ/OPENAI_API_KEY

litellm_settings:
    master_key: os.environ/LITELLM_MASTER_KEY
    database_url: os.environ/DATABASE_URL

general_settings:
    callbacks: ["langfuse"] # optional: add logging
```

[See More Config Settings..](./config_settings.md)

## Test Requests

### Chat Completions

<Tabs>
<TabItem value="bedrock-claude" label="Bedrock Claude">

```bash title="Test Bedrock Claude 3.5 Sonnet"
curl -X POST 'http://localhost:4000/chat/completions' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-1234' \
  -d '{
    "model": "bedrock-claude-3-5-sonnet",
    "messages": [
      {
        "role": "user",
        "content": "What is the capital of France?"
      }
    ]
  }'
```

</TabItem>

<TabItem value="gemini" label="Gemini">

```bash title="Test Gemini 2.5 Pro"
curl -X POST 'http://localhost:4000/chat/completions' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-1234' \
  -d '{
    "model": "gemini-2.5-pro",
    "messages": [
      {
        "role": "user",
        "content": "What is the capital of France?"
      }
    ]
  }'
```

</TabItem>

<TabItem value="azure-codex" label="Azure GPT-5 Codex">

```bash title="Test Azure GPT-5 Codex"
curl -X POST 'http://localhost:4000/chat/completions' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-1234' \
  -d '{
    "model": "gpt-5-codex",
    "messages": [
      {
        "role": "user",
        "content": "Write a Python function to calculate factorial"
      }
    ]
  }'
```

</TabItem>

<TabItem value="openai-pro" label="OpenAI GPT-5 Pro">

```bash title="Test OpenAI GPT-5 Pro"
curl -X POST 'http://localhost:4000/chat/completions' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-1234' \
  -d '{
    "model": "gpt-5-pro",
    "messages": [
      {
        "role": "user",
        "content": "Explain quantum computing in simple terms"
      }
    ]
  }'
```

</TabItem>
</Tabs>

### Responses API

<Tabs>
<TabItem value="bedrock-claude-responses" label="Bedrock Claude">

```bash title="Test Bedrock Claude with Responses API"
curl -X POST 'http://localhost:4000/v1/responses' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-1234' \
  -d '{
    "model": "bedrock-claude-3-5-sonnet",
    "input": "What is the capital of France?"
  }'
```

</TabItem>

<TabItem value="gemini-responses" label="Gemini">

```bash title="Test Gemini with Responses API"
curl -X POST 'http://localhost:4000/v1/responses' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-1234' \
  -d '{
    "model": "gemini-2.5-pro",
    "input": "What is the capital of France?"
  }'
```

</TabItem>

<TabItem value="azure-codex-responses" label="Azure GPT-5 Codex">

```bash title="Test Azure GPT-5 Codex with Responses API"
curl -X POST 'http://localhost:4000/v1/responses' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-1234' \
  -d '{
    "model": "gpt-5-codex",
    "input": "Write a Python function to calculate factorial"
  }'
```

</TabItem>

<TabItem value="openai-pro-responses" label="OpenAI GPT-5 Pro">

```bash title="Test OpenAI GPT-5 Pro with Responses API"
curl -X POST 'http://localhost:4000/v1/responses' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-1234' \
  -d '{
    "model": "gpt-5-pro",
    "input": "Explain quantum computing"
  }'
```

</TabItem>
</Tabs>

## Cost Tracking

### Daily Activity

<Tabs>
<TabItem value="user" label="User">


</TabItem>

</Tabs>