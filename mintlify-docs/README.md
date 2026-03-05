# LiteLLM Mintlify Docs

This directory contains the **Mintlify** version of the LiteLLM documentation, migrating from the existing Docusaurus site at `docs/my-website/`.

## Local Development

```bash
# Install Mintlify CLI
npm install -g mintlify

# Run local dev server from this directory
mintlify dev
```

## Structure

```
mintlify-docs/
├── docs.json              # Main config: navigation, colors, branding
├── index.mdx              # Home / Getting Started page
├── favicon.ico            # Site favicon
├── logo/                  # Logo files (light/dark variants)
├── images/                # Static images
├── snippets/              # Reusable MDX snippets
├── completion/            # completion() SDK docs
├── providers/             # Provider-specific docs (OpenAI, Anthropic, Bedrock, etc.)
├── proxy/                 # LiteLLM Proxy/Gateway docs
├── guides/                # How-to guides
├── tutorials/             # Step-by-step tutorials
├── integrations/          # Third-party integrations overview
├── observability/         # Observability integrations (Langfuse, MLflow, etc.)
├── caching/               # Caching docs
├── embedding/             # Embedding docs
├── pass_through/          # Pass-through endpoint docs
├── secret_managers/       # Secret manager integrations
├── troubleshoot/          # Troubleshooting guides
├── adding_provider/       # Contributing: adding providers
├── contributing/          # Contributing guides
├── extras/                # Additional docs
├── langchain/             # LangChain integration
├── vector_stores/         # Vector store docs
├── search/                # Search endpoint docs
├── anthropic_unified/     # Anthropic unified API docs
├── a2a/                   # A2A Agent Gateway docs
├── mcp/                   # Model Context Protocol docs
└── projects/              # Projects built on LiteLLM
```

## Migration from Docusaurus

### What changed

| Docusaurus | Mintlify |
|---|---|
| `docusaurus.config.js` + `sidebars.js` | `docs.json` |
| `@theme/Tabs` / `@theme/TabItem` | `<Tabs>` / `<Tab>` (built-in) |
| `:::note`, `:::warning` admonitions | `<Note>`, `<Warning>`, `<Tip>`, `<Info>` |
| `.md` files | `.mdx` files (`.md` also supported) |
| Mermaid diagrams | Mermaid supported natively |
| Custom CSS | `docs.json` colors + Mintlify theming |

### Migrating a doc page

1. **Copy** the `.md` file from `docs/my-website/docs/` to the corresponding path here
2. **Update frontmatter** — keep `title` and `description`, remove Docusaurus-specific fields
3. **Replace component imports**:
   - Remove `import Tabs from '@theme/Tabs'` and `import TabItem from '@theme/TabItem'`
   - Replace `<Tab title="Y&quot;>` with `<Tab title=&quot;Y">`
   - Replace `<Tabs>` stays as `<Tabs>`, but wrap in Mintlify's built-in
4. **Replace admonitions**:
   - `<Note title="... :::` → `<Note>...</Note>`">
   - `
</Note>warning ... :::` → `<Warning>...</Warning>`
   - `<Tip title="... :::` → `<Tip>...</Tip>`">
   - `
</Tip>info ... :::` → `<Info>...</Info>`
5. **Images**: Move to `images/` directory, update paths

### Bulk migration script

For bulk migration, run from the repo root:

```bash
# Copy all docs content
cp -r docs/my-website/docs/* mintlify-docs/

# Copy static images
cp -r docs/my-website/static/img/* mintlify-docs/images/

# Copy favicon
cp docs/my-website/static/img/favicon.ico mintlify-docs/

# Copy logo
cp docs/my-website/static/img/logo.svg mintlify-docs/logo/
```

After copying, run a search-replace for Docusaurus-specific syntax:

```bash
# Replace Docusaurus tab imports (can be safely removed in Mintlify)
find mintlify-docs -name "*.md" -o -name "*.mdx" | \
  xargs sed -i '' \
  -e '/^import Tabs from/d' \
  -e '/^import TabItem from/d' \
  -e 's/<Tab title="\([^">/<\/Tab>/g'

# Replace Docusaurus admonitions
find mintlify-docs -name "*.md" -o -name "*.mdx" | \
  xargs sed -i '' \
  -e 's/:::note/<Note>/g' \
  -e 's/:::warning/<Warning>/g' \
  -e 's/:::tip/<Tip>/g' \
  -e 's/:::info/<Info>/g' \
  -e 's/:::danger/<Warning>/g' \
  -e 's/:::/>/g'
```

> **Note**: The sed replacements are approximate — some pages with complex JSX or custom components will need manual review.

## Key Mintlify Features to Use

- **`<CodeGroup>`** — group multiple code blocks by language
- **`<Card>` / `<CardGroup>`** — feature cards for landing pages
- **`<Accordion>`** — collapsible sections
- **`<Tabs>` / `<Tab>`** — tabbed content (replaces Docusaurus Tabs)
- **`<Note>`, `<Warning>`, `<Tip>`, `<Info>`** — callout boxes
- **API playground** — auto-generated from OpenAPI spec (add `openapi.yaml`)
- **Mermaid diagrams** — supported natively, no plugin needed

## Deployment

Mintlify deploys automatically from GitHub. Connect the repo via the [Mintlify dashboard](https://dashboard.mintlify.com) and set the docs directory to `mintlify-docs/`.
