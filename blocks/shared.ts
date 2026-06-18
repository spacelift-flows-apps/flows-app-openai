import OpenAI from "openai";
import { events, kv, timers } from "@slflows/sdk/v1";
import { OpenAIAuth, createClient } from "../client";

export type InputItem = OpenAI.Responses.ResponseInputItem;
export type ResponseTool = OpenAI.Responses.Tool;
export type ReasoningEffort = OpenAI.Reasoning["effort"];

// Hard cap on tool-call turns per run, so a model that keeps calling tools (or a
// forced tool choice) can't loop forever.
export const MAX_TOOL_TURNS = 16;

export interface ToolDefinition {
  blockId?: string;
  name: string;
  description?: string;
  schema?: Record<string, any>;
}

export interface McpServer {
  name: string;
  url: string;
  authorizationToken?: string;
  allowedTools?: string[];
}

export interface JsonSchema {
  name: string;
  schema: Record<string, any>;
}

function strictify(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(strictify);
  }

  if (!value || typeof value !== "object") {
    return value;
  }

  return toStrictSchema(value as Record<string, unknown>);
}

// Strict structured outputs require every property to be present — there is no
// absent key in strict mode. So we recursively force every property into
// `required`, set additionalProperties:false on each object, and encode optional
// fields (those absent from `required`) as nullable via `anyOf[…, {type:null}]`.
export function toStrictSchema(node: Record<string, unknown>) {
  const result: Record<string, unknown> = {};

  for (const key of Object.keys(node)) {
    result[key] = strictify(node[key]);
  }

  const properties = result.properties;

  if (
    result.type === "object" &&
    properties &&
    typeof properties === "object"
  ) {
    const props = properties as Record<string, unknown>;
    const required = new Set(
      Array.isArray(result.required) ? result.required : [],
    );

    for (const name of Object.keys(props)) {
      if (!required.has(name)) {
        props[name] = { anyOf: [props[name], { type: "null" }] };
      }
    }

    result.required = Object.keys(props);
    result.additionalProperties = false;
  }

  return result;
}

// OpenAI function/tool and MCP server labels must match ^[a-zA-Z0-9_-]{1,64}$.
// https://platform.openai.com/docs/guides/function-calling
export function cleanToolName(name: string) {
  return name
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "_")
    .replace(/[^a-z0-9_-]/g, "_")
    .replace(/[_-]{2,}/g, "_")
    .replace(/^[_-]+|[_-]+$/g, "")
    .slice(0, 64);
}

export function buildToolChoice(force: boolean | string, hasTools: boolean) {
  if (!hasTools) {
    return undefined;
  }

  if (typeof force === "string" && force) {
    return {
      type: "function" as const,
      name: cleanToolName(force),
    } satisfies OpenAI.Responses.ToolChoiceFunction;
  }

  return force === true ? ("required" as const) : ("auto" as const);
}

export function streamResponse(params: {
  auth: OpenAIAuth;
  model: string;
  maxOutputTokens: number;
  temperature?: number;
  input: InputItem[];
  instructions?: string;
  tools: ResponseTool[];
  toolChoice?: ReturnType<typeof buildToolChoice>;
  reasoningEffort?: ReasoningEffort;
  schema?: JsonSchema;
  maxRetries: number;
}) {
  const {
    auth,
    model,
    maxOutputTokens,
    temperature,
    input,
    instructions,
    tools,
    toolChoice,
    reasoningEffort,
    schema,
    maxRetries,
  } = params;

  const client = createClient(auth, maxRetries);

  return client.responses.stream({
    model,
    input,
    instructions,
    max_output_tokens: maxOutputTokens,
    tools: tools.length > 0 ? tools : undefined,
    tool_choice: toolChoice,
    // Reasoning models reject `temperature`; only send it when explicitly set.
    ...(temperature !== undefined ? { temperature } : {}),
    ...(reasoningEffort ? { reasoning: { effort: reasoningEffort } } : {}),
    text: schema
      ? {
          format: {
            type: "json_schema",
            name: schema.name,
            schema: toStrictSchema(schema.schema),
            strict: true,
          },
        }
      : undefined,
    // Stateless: we carry the full conversation ourselves. Encrypted reasoning
    // lets reasoning items survive across tool-call turns without server storage.
    store: false,
    include: reasoningEffort ? ["reasoning.encrypted_content"] : undefined,
  });
}

// Consumes the stream to drive live pending-status updates and returns the raw
// response from the terminal event. Avoids stream.finalResponse(), which injects
// non-input fields (parsed_arguments/parsed) that break re-feeding output as the
// next turn's input.
export async function streamToResponse(
  pendingId: string,
  stream: ReturnType<typeof streamResponse>,
) {
  let response: OpenAI.Responses.Response | undefined;

  for await (const event of stream) {
    if (
      event.type === "response.completed" ||
      event.type === "response.incomplete"
    ) {
      response = event.response;
      continue;
    }

    if (event.type !== "response.output_item.added") {
      continue;
    }

    switch (event.item.type) {
      case "reasoning":
        await events.updatePending(pendingId, {
          statusDescription: "Reasoning...",
        });
        break;
      case "function_call":
        await events.updatePending(pendingId, {
          statusDescription: "Preparing tool call...",
        });
        break;
      case "mcp_call":
        await events.updatePending(pendingId, {
          statusDescription: `Calling "${event.item.name}" on "${event.item.server_label}"`,
        });
        break;
      case "message":
        await events.updatePending(pendingId, {
          statusDescription: "Generating response...",
        });
        break;
    }
  }

  if (!response) {
    throw new Error("Stream ended without a completed response");
  }

  return response;
}

// Carry the model's full output into the next turn's input. With store:false we
// own the conversation, so every item must be replayed - dropping server-side
// tool artifacts (mcp_call, ...) would lose context and break prompt caching.
// message/reasoning/function_call are valid input as-is; hosted-tool items are
// too, but their input variants are nominally distinct, so each is coerced alone.
export function toInputItems(output: OpenAI.Responses.ResponseOutputItem[]) {
  const items: InputItem[] = [];

  for (const item of output) {
    if (
      item.type === "message" ||
      item.type === "reasoning" ||
      item.type === "function_call"
    ) {
      items.push(item);
    } else {
      items.push(item as unknown as InputItem);
    }
  }

  return items;
}

export function getFunctionCalls(response: OpenAI.Responses.Response) {
  return response.output.filter(
    (item): item is OpenAI.Responses.ResponseFunctionToolCall =>
      item.type === "function_call",
  );
}

// `response.output_text` is only computed by the SDK for auto-parseable requests
// (zod text formats / strict tools); with plain prompts and raw JSON schemas it
// stays undefined. Aggregate the assistant text from the output items, the same
// way the SDK's own addOutputText() does.
export function extractOutputText(response: OpenAI.Responses.Response) {
  return response.output
    .filter(
      (item): item is OpenAI.Responses.ResponseOutputMessage =>
        item.type === "message",
    )
    .flatMap((item) => item.content)
    .filter(
      (part): part is OpenAI.Responses.ResponseOutputText =>
        part.type === "output_text",
    )
    .map((part) => part.text)
    .join("");
}

export function joinToolNames(
  toolCalls: OpenAI.Responses.ResponseFunctionToolCall[],
  toolNames: Record<string, string>,
) {
  const label = (call: OpenAI.Responses.ResponseFunctionToolCall) =>
    `"${toolNames[call.name] ?? call.name}"`;

  if (toolCalls.length === 0) {
    return "";
  }

  if (toolCalls.length === 1) {
    return label(toolCalls[0]);
  }

  return `${toolCalls.slice(0, -1).map(label).join(", ")} and ${label(
    toolCalls[toolCalls.length - 1],
  )}`;
}

// Both Agent and Generate blocks run the same stateless multi-turn loop:
// dispatch tool calls, wait for results, then continue. These helpers own the `timer-<id>` and
// `process-<id>-<turn>` keys; `id` is whatever the block uses to scope a run
// (an event id for the Generate block, an execution id for the Agent block).

export async function setTimeoutTimer(id: string, turn: number) {
  const timerId = await timers.block.set(60 * 2, {
    description: "Waiting for tool results",
    inputPayload: { id, turn },
  });

  await kv.block.set({ key: `timer-${id}`, value: timerId, ttl: 60 * 5 });
}

export async function clearTimeoutTimer(id: string) {
  const { value } = await kv.block.get(`timer-${id}`);

  if (value) {
    await timers.block.unset(value);
    await kv.block.delete([`timer-${id}`]);
  }
}

// Per-turn atomic lock so concurrent tool results (parallel tool calls) and a
// firing timeout can't both advance the same turn. Whoever wins drives the
// continuation; everyone else records their result and returns.
export async function tryAcquireProcessingLock(
  id: string,
  turn: number,
  lockId: string,
) {
  return kv.block.set({
    key: `process-${id}-${turn}`,
    value: { lockId },
    lock: { id: lockId, timeout: 300 },
  });
}

export async function releaseProcessingLock(
  id: string,
  turn: number,
  lockId: string,
) {
  await kv.block.releaseLock({
    key: `process-${id}-${turn}`,
    lockId,
  });
}

// One pending per event id, so a redelivery reuses it instead of orphaning another.
export async function getOrCreatePending(
  id: string,
  statusDescription: string,
) {
  const { value } = await kv.block.get(`pending-${id}`);

  if (typeof value === "string") {
    return value;
  }

  const pendingId = await events.createPending({ statusDescription });
  await kv.block.set({ key: `pending-${id}`, value: pendingId, ttl: 60 * 60 });

  return pendingId;
}

// Cancel the pending if still ours; clear the marker so a retry starts fresh.
export async function failPending(
  id: string,
  pendingId: string,
  reason: string,
) {
  try {
    await events.cancelPending(pendingId, reason);
  } catch {
    return;
  }

  await kv.block.delete([`pending-${id}`]);
}
