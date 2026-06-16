# OpenAI

A Spacelift Flows app for generating text and running tool-using agents with OpenAI models, built on the OpenAI [Responses API](https://platform.openai.com/docs/api-reference/responses).

## Config

- **OpenAI API Key** (required) - your OpenAI API key (starts with `sk-`). Create one at https://platform.openai.com/api-keys.
- **Default model** (optional) - the model used when a block doesn't specify one.
- **Base URL** (optional) - custom base URL for OpenAI-compatible endpoints.
- **Organization ID** (optional) - scopes API usage to a specific OpenAI organization.

## Blocks

- **Generate response** (`generateMessage`)
  - Generates a response from a prompt. Supports schema-based structured output, custom tool calls, remote MCP servers, reasoning effort, retries, and temperature. Tool calls are dispatched to connected **Tool definition** blocks and the conversation continues automatically once results return.

- **Agent** (`agent`)
  - A tool-using agent. Tools are configured statically and each becomes a dedicated output on the block; results are routed back through the **Agent Tool Result** block. Supports structured output via an output schema, reasoning effort, retries, and temperature.

- **Agent Tool Result** (`agentToolResult`)
  - Routes a processed tool result back to the **Agent** block. Connect it after your tool-processing pipeline.

- **Tool definition** (`toolDefinition`)
  - Defines a custom tool (name, description, JSON schema) that the **Generate response** block can call. Exposes the definition via a signal and returns the tool's result via its `Process result` input.

- **Remote MCP server** (`remoteMcpServer`)
  - Configures a remote MCP server the model can use while generating responses. Validates connectivity, lists available tools, and exposes its configuration via a signal for use by the **Generate response** block.

- **HTTP Request** (`httpRequest`)
  - Escape hatch for making an authenticated request to any OpenAI API endpoint not covered by a dedicated block.

## Structured output

OpenAI's Responses API supports structured output in the same request as tool use. When a `Schema` (Generate response) or `Output schema` (Agent) is provided, the model is constrained to produce JSON matching that schema, returned on the `object` / `output` field of the result.

## Reasoning

For reasoning-capable models (e.g. the GPT‑5 family and o‑series), set **Reasoning effort** to control how much the model reasons before responding. Leave it unset to use the model's default. Reasoning models do not support the `temperature` parameter.

## Development

```bash
npm install
npm run typecheck
npm run format
npm run bundle
```
