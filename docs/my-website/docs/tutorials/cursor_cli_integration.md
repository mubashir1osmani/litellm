# Cursor CLI Integration

Route Cursor CLI (headless mode) requests through LiteLLM for unified logging, budget controls, and model access.

:::info
Cursor CLI is in beta. See [Cursor CLI docs](https://cursor.com/docs/cli/headless) for the latest features.
:::

## Quick Reference

| Setting | Value |
|---------|-------|
| Environment Variable | `OPENAI_BASE_URL` |
| Base URL | `<LITELLM_PROXY_BASE_URL>/cursor` |
| API Key Variable | `OPENAI_API_KEY` |
| API Key | Your LiteLLM Virtual Key |

---

## Installation

Install Cursor CLI:

```bash
curl https://cursor.com/install -fsS | bash
```

Verify installation:

```bash
cursor --version
```

---

## Setup with LiteLLM

### 1. Create Virtual Key

In LiteLLM Dashboard, go to **Virtual Keys → + Create New Key**.

Name your key and select which models it can access, then click **Create Key** and copy it.

### 2. Configure Environment

Set the required environment variables to route requests through LiteLLM:

```bash
export OPENAI_BASE_URL="https://your-litellm-proxy.com/cursor"
export OPENAI_API_KEY="sk-your-litellm-virtual-key"
```

For persistent configuration, add these to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.):

```bash
echo 'export OPENAI_BASE_URL="https://your-litellm-proxy.com/cursor"' >> ~/.zshrc
echo 'export OPENAI_API_KEY="sk-your-litellm-virtual-key"' >> ~/.zshrc
source ~/.zshrc
```

### 3. Test Connection

Run a simple command to verify the setup:

```bash
cursor "What is 2+2?"
```

---

## Headless Mode Usage

Cursor CLI supports headless mode for automation and CI/CD pipelines.

### Print Mode (Non-Interactive)

Use `--print` (`-p`) for non-interactive output:

```bash
cursor --print "Explain this code" ./src/main.py
```

### Force Mode (Auto-Accept Changes)

Combine `--print` with `--force` to allow file modifications without confirmation:

```bash
cursor --print --force "Add error handling to this function" ./src/utils.py
```

### Example: CI/CD Pipeline

```yaml
# .github/workflows/code-review.yml
name: Cursor Code Review
on: [pull_request]

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Cursor CLI
        run: curl https://cursor.com/install -fsS | bash

      - name: Run Code Review
        env:
          OPENAI_BASE_URL: ${{ secrets.LITELLM_BASE_URL }}/cursor
          OPENAI_API_KEY: ${{ secrets.LITELLM_API_KEY }}
        run: |
          cursor --print "Review the changes in this PR for potential issues" .
```

---

## Model Selection

To use a specific model from your LiteLLM deployment, set the model:

```bash
cursor --model "claude-sonnet-4-20250514" "Explain this code"
```

Ensure the model name matches the **Public Model Name** configured in LiteLLM.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Connection refused | Verify `OPENAI_BASE_URL` is correct and ends with `/cursor` |
| Authentication failed | Check `OPENAI_API_KEY` is a valid LiteLLM virtual key |
| Model not found | Ensure model name matches LiteLLM's public model name |
| Rate limited | Check budget/rate limits on your LiteLLM virtual key |

### Debug Mode

For troubleshooting, enable verbose output:

```bash
cursor --verbose "Your prompt here"
```

---

## Additional Resources

- [Cursor CLI Documentation](https://cursor.com/docs/cli/overview)
- [Cursor Headless Mode](https://cursor.com/docs/cli/headless)
- [Cursor IDE Integration with LiteLLM](./cursor_integration.md)
