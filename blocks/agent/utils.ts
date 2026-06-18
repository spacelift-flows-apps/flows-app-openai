import OpenAI from "openai";
import { events, kv } from "@slflows/sdk/v1";
import { OpenAIAuth, defaultModel, resolveAuth } from "../../client";
import {
  InputItem,
  JsonSchema,
  ReasoningEffort,
  ToolDefinition,
  buildToolChoice,
  cleanToolName,
  failPending,
  extractOutputText,
  getFunctionCalls,
  joinToolNames,
  MAX_TOOL_TURNS,
  setTimeoutTimer,
  streamResponse,
  streamToResponse,
  toInputItems,
} from "../shared";

interface CallState {
  blockId: string;
  eventIds: string[];
  inputItems: InputItem[];
  toolCallIds: string[];
  pendingId: string;
  toolDefinitions: ToolDefinition[];
  force: boolean | string;
  maxOutputTokens: number;
  model: string;
  instructions: string | undefined;
  turn: number;
  maxRetries: number;
  schema: JsonSchema | undefined;
  reasoningEffort: ReasoningEffort | undefined;
  temperature: number | undefined;
  originalEventId: string;
}

export function createInvocationId(
  blockId: string,
  executionId: string,
  toolCallId: string,
) {
  return `${blockId}::${executionId}::${toolCallId}`;
}

export function parseInvocationId(invocationId: string) {
  const parts = invocationId.split("::");

  if (parts.length !== 3) {
    throw new Error(`Invalid invocation ID: ${invocationId}`);
  }

  return {
    blockId: parts[0],
    executionId: parts[1],
    toolCallId: parts[2],
  };
}

export function getToolDefinitionOutputKey(name: string) {
  return `tool_definition_${name.toLowerCase().replace(/\s+/g, "_")}`;
}

function processToolDefinitions(toolDefinitions: ToolDefinition[]) {
  const tools: OpenAI.Responses.FunctionTool[] = [];
  const toolNames: Record<string, string> = {};
  const toolOutputKeys: Record<string, string> = {};

  for (const toolDefinition of toolDefinitions) {
    const cleanedName = cleanToolName(toolDefinition.name);

    tools.push({
      type: "function",
      name: cleanedName,
      description: toolDefinition.description,
      parameters: toolDefinition.schema ?? {
        type: "object",
        properties: {},
        additionalProperties: false,
      },
      strict: false,
    });

    toolNames[cleanedName] = toolDefinition.name;
    toolOutputKeys[cleanedName] = getToolDefinitionOutputKey(
      toolDefinition.name,
    );
  }

  return { tools, toolNames, toolOutputKeys };
}

export function validateConfig(
  appConfig: Record<string, any>,
  staticConfig: Record<string, any>,
  inputConfig: Record<string, any>,
) {
  const auth = resolveAuth(appConfig);
  const model = staticConfig.model ?? defaultModel(appConfig);

  const schema = staticConfig.outputSchema as Record<string, any> | undefined;

  return {
    auth,
    model,
    toolDefinitions: (staticConfig.toolDefinitions ?? []) as ToolDefinition[],
    prompt: inputConfig.prompt as string,
    maxOutputTokens: inputConfig.maxTokens as number,
    instructions: inputConfig.systemPrompt as string | undefined,
    force: (inputConfig.force ?? false) as boolean | string,
    reasoningEffort: inputConfig.reasoningEffort as ReasoningEffort | undefined,
    schema:
      schema && Object.keys(schema).length > 0
        ? ({ name: "output", schema } satisfies JsonSchema)
        : undefined,
    maxRetries: (inputConfig.maxRetries ?? 2) as number,
    temperature: inputConfig.temperature as number | undefined,
  };
}

async function emitResult(params: {
  pendingId: string;
  result: {
    output: unknown;
    usage: { inputTokens: number; outputTokens: number };
  };
  originalEventId: string;
  eventIds: string[];
}) {
  // Parent is the most recent tool result, or the original input if no tools were called.
  const parentEventId =
    params.eventIds.length > 0 ? params.eventIds[0] : params.originalEventId;

  // Other tool results as secondary parents (the original is already an ancestor).
  const secondaryParentEventIds =
    params.eventIds.length > 1 ? params.eventIds.slice(1) : undefined;

  await events.emit(
    {
      output: params.result.output,
      usage: params.result.usage,
    },
    {
      outputKey: "result",
      complete: params.pendingId,
      parentEventId,
      secondaryParentEventIds,
    },
  );
}

async function storeCallState(params: {
  executionId: string;
  blockId: string;
  eventIds: string[];
  originalEventId: string;
  pendingId: string;
  inputItems: InputItem[];
  toolCalls: OpenAI.Responses.ResponseFunctionToolCall[];
  toolDefinitions: ToolDefinition[];
  force: boolean | string;
  maxOutputTokens: number;
  model: string;
  instructions: string | undefined;
  maxRetries: number;
  schema: JsonSchema | undefined;
  turn: number;
  reasoningEffort: ReasoningEffort | undefined;
  temperature: number | undefined;
}) {
  const { executionId, toolCalls, ...rest } = params;

  await kv.block.set({
    key: `call-${executionId}`,
    value: {
      toolCallIds: toolCalls.map((toolCall) => toolCall.call_id),
      ...rest,
    } satisfies CallState,
    ttl: 60 * 60,
  });
}

export async function loadCallState(executionId: string) {
  const { value } = await kv.block.get(`call-${executionId}`);
  return value as CallState | undefined;
}

export async function deleteCallState(executionId: string) {
  await kv.block.delete([`call-${executionId}`]);
}

export async function storeToolResult(params: {
  executionId: string;
  toolCallId: string;
  result: unknown;
  turn: number;
  eventId: string;
}) {
  const { executionId, toolCallId, result, turn, eventId } = params;

  await kv.block.set({
    key: `result-${executionId}-${turn}-${toolCallId}`,
    value: {
      toolCallId,
      result:
        typeof result === "string" ? result : JSON.stringify(result ?? null),
      eventId,
    },
    ttl: 60 * 5,
  });
}

export async function loadToolResults(params: {
  executionId: string;
  turn: number;
  toolCallIds: string[];
}) {
  const { executionId, turn, toolCallIds } = params;

  const results = await kv.block.list({
    keyPrefix: `result-${executionId}-${turn}-`,
  });

  const toolResults = results.pairs.reduce(
    (acc, { value }) => {
      acc[value.toolCallId] = value.result;
      return acc;
    },
    {} as Record<string, string>,
  );

  const haveAllResults = toolCallIds.every(
    (toolCallId) => typeof toolResults[toolCallId] !== "undefined",
  );

  return {
    haveAllResults,
    toolResults,
    eventIds: results.pairs.map(({ value }) => value.eventId),
  };
}

export async function continueTurn(params: {
  executionId: string;
  blockId: string;
  eventIds: string[];
  pendingId: string;
  inputItems: InputItem[];
  toolCallIds: string[];
  toolDefinitions: ToolDefinition[];
  toolResults: Record<string, string>;
  force: boolean | string;
  model: string;
  maxOutputTokens: number;
  instructions: string | undefined;
  turn: number;
  maxRetries: number;
  schema: JsonSchema | undefined;
  auth: OpenAIAuth;
  reasoningEffort: ReasoningEffort | undefined;
  temperature: number | undefined;
  originalEventId: string;
}) {
  const { pendingId, inputItems, toolCallIds, toolResults } = params;

  await events.updatePending(pendingId, {
    statusDescription: `Received results from tool${
      toolCallIds.length === 1 ? "" : "s"
    }...`,
  });

  const nextInputItems: InputItem[] = [
    ...inputItems,
    ...toolCallIds
      .filter((id) => typeof toolResults[id] !== "undefined")
      .map(
        (id) =>
          ({
            type: "function_call_output" as const,
            call_id: id,
            output: toolResults[id],
          }) satisfies OpenAI.Responses.ResponseInputItem.FunctionCallOutput,
      ),
  ];

  return executeTurn({ ...params, inputItems: nextInputItems });
}

export async function executeTurn(params: {
  executionId: string;
  blockId: string;
  pendingId: string;
  originalEventId: string;
  eventIds: string[];
  inputItems: InputItem[];
  toolDefinitions: ToolDefinition[];
  force: boolean | string;
  model: string;
  maxOutputTokens: number;
  instructions: string | undefined;
  turn: number;
  auth: OpenAIAuth;
  maxRetries: number;
  schema: JsonSchema | undefined;
  reasoningEffort: ReasoningEffort | undefined;
  temperature: number | undefined;
}) {
  const {
    executionId,
    pendingId,
    inputItems,
    toolDefinitions,
    force,
    model,
    maxOutputTokens,
    instructions,
    auth,
    maxRetries,
    reasoningEffort,
    temperature,
    schema,
    blockId,
    eventIds,
    originalEventId,
  } = params;

  try {
    // Backstop against runaway loops: cap the number of model round-trips.
    if (params.turn >= MAX_TOOL_TURNS) {
      await failPending(
        originalEventId,
        pendingId,
        `Reached the maximum of ${MAX_TOOL_TURNS} model turns without a final answer.`,
      );
      await deleteCallState(executionId);
      return;
    }

    const { tools, toolNames, toolOutputKeys } =
      processToolDefinitions(toolDefinitions);

    const stream = streamResponse({
      auth,
      model,
      maxOutputTokens,
      temperature,
      input: inputItems,
      instructions,
      tools,
      // Force a tool only on the opening turn; forcing every turn would never
      // let the model produce a final answer.
      toolChoice: buildToolChoice(
        params.turn === 0 ? force : false,
        tools.length > 0,
      ),
      reasoningEffort,
      schema,
      maxRetries,
    });

    const response = await streamToResponse(pendingId, stream);

    const functionCalls = getFunctionCalls(response);

    const nextInputItems: InputItem[] = [
      ...inputItems,
      ...toInputItems(response.output),
    ];

    if (functionCalls.length === 0) {
      try {
        // An incomplete response (e.g. max_output_tokens) often has no message
        // text - surface why instead of silently emitting an empty result.
        if (response.status === "incomplete") {
          const reason = response.incomplete_details?.reason ?? "unknown";
          return await failPending(
            originalEventId,
            pendingId,
            `Response incomplete: ${reason}`,
          );
        }

        const text = extractOutputText(response);
        let output: unknown = text;

        if (schema) {
          try {
            output = JSON.parse(text);
          } catch {
            console.error("Failed to parse structured output");
          }
        }

        return await emitResult({
          pendingId,
          result: {
            output,
            usage: {
              inputTokens: response.usage?.input_tokens ?? 0,
              outputTokens: response.usage?.output_tokens ?? 0,
            },
          },
          eventIds,
          originalEventId,
        });
      } finally {
        await deleteCallState(executionId);
      }
    }

    await events.updatePending(pendingId, {
      statusDescription:
        functionCalls.length === 1
          ? `Calling tool: ${joinToolNames(functionCalls, toolNames)}`
          : `Calling tools: ${joinToolNames(functionCalls, toolNames)}`,
    });

    const nextTurn = params.turn + 1;

    // Persist state before dispatching so a fast tool result always finds it.
    await storeCallState({
      executionId,
      blockId,
      eventIds,
      originalEventId,
      pendingId,
      inputItems: nextInputItems,
      toolCalls: functionCalls,
      toolDefinitions,
      force: params.force,
      maxOutputTokens: params.maxOutputTokens,
      model: params.model,
      instructions: params.instructions,
      maxRetries: params.maxRetries,
      schema: params.schema,
      turn: nextTurn,
      reasoningEffort: params.reasoningEffort,
      temperature: params.temperature,
    });

    // A call to a tool with no matching output (a name the model invented) can't
    // be dispatched - answer it with an error inline so the turn still resolves
    // instead of hanging until the timeout.
    await Promise.all(
      functionCalls
        .filter((toolCall) => !toolOutputKeys[toolCall.name])
        .map((toolCall) =>
          storeToolResult({
            executionId,
            toolCallId: toolCall.call_id,
            result: `Error: tool "${
              toolNames[toolCall.name] ?? toolCall.name
            }" is not available.`,
            turn: nextTurn,
            eventId: originalEventId,
          }),
        ),
    );

    await Promise.all(
      functionCalls
        .filter((toolCall) => toolOutputKeys[toolCall.name])
        .map((toolCall) => {
          let parameters: unknown = {};
          try {
            parameters = toolCall.arguments
              ? JSON.parse(toolCall.arguments)
              : {};
          } catch {
            parameters = {};
          }

          return events.emit(
            {
              parameters,
              invocationId: createInvocationId(
                blockId,
                executionId,
                toolCall.call_id,
              ),
            },
            {
              outputKey: toolOutputKeys[toolCall.name],
              parentEventId: originalEventId,
              secondaryParentEventIds:
                eventIds.length > 0 ? eventIds : undefined,
            },
          );
        }),
    );

    return setTimeoutTimer(executionId, nextTurn);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    await failPending(
      originalEventId,
      pendingId,
      `API call failed: ${message}`,
    );
    await deleteCallState(executionId);
    throw error;
  }
}
