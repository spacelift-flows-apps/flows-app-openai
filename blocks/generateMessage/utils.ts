import OpenAI from "openai";
import { events, kv, messaging } from "@slflows/sdk/v1";
import { OpenAIAuth, defaultModel, resolveAuth } from "../../client";
import {
  InputItem,
  JsonSchema,
  McpServer,
  ReasoningEffort,
  ToolDefinition,
  buildToolChoice,
  cleanToolName,
  clearPendingMarker,
  describeFailure,
  failPending,
  extractOutputText,
  getFunctionCalls,
  joinToolNames,
  MAX_TOOL_TURNS,
  setTimeoutTimer,
  streamResponse,
  streamToResponse,
  toInputItems,
  validateMcpServerUrl,
} from "../shared";

function processToolDefinitions(toolDefinitions: ToolDefinition[]) {
  const tools: OpenAI.Responses.FunctionTool[] = [];
  const toolNames: Record<string, string> = {};
  const toolBlockIds: Record<string, string> = {};

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
      // User-provided schemas rarely satisfy strict-mode constraints (every
      // property required, additionalProperties: false), so keep it lenient.
      strict: false,
    });

    toolNames[cleanedName] = toolDefinition.name;

    if (toolDefinition.blockId) {
      toolBlockIds[cleanedName] = toolDefinition.blockId;
    }
  }

  return { tools, toolNames, toolBlockIds };
}

function buildMcpTools(mcpServers: McpServer[]) {
  return mcpServers.map(
    (server) =>
      ({
        type: "mcp",
        server_label: cleanToolName(server.name),
        server_url: server.url,
        authorization: server.authorizationToken,
        allowed_tools: server.allowedTools,
        require_approval: "never",
      }) satisfies OpenAI.Responses.Tool.Mcp,
  );
}

interface CallState {
  inputItems: InputItem[];
  toolCallIds: string[];
  pendingId: string;
  toolDefinitions: ToolDefinition[];
  force: boolean | string;
  maxOutputTokens: number;
  mcpServers: McpServer[];
  model: string;
  instructions: string | undefined;
  turn: number;
  maxRetries: number;
  schema: JsonSchema | undefined;
  reasoningEffort: ReasoningEffort | undefined;
  temperature: number | undefined;
}

export function validateConfig(
  appConfig: Record<string, any>,
  inputConfig: Record<string, any>,
) {
  const auth = resolveAuth(appConfig);
  const model = inputConfig.model ?? defaultModel(appConfig);

  const schema = inputConfig.schema as Record<string, any> | undefined;

  const mcpServers = (inputConfig.mcpServers ?? []) as McpServer[];
  for (const server of mcpServers) {
    const urlError = validateMcpServerUrl(server.url);
    if (urlError) {
      throw new Error(`MCP server "${server.name}": ${urlError}`);
    }
  }

  return {
    auth,
    model,
    toolDefinitions: (inputConfig.toolDefinitions ?? []) as ToolDefinition[],
    mcpServers,
    prompt: inputConfig.prompt as string,
    maxOutputTokens: inputConfig.maxTokens as number,
    instructions: inputConfig.systemPrompt as string | undefined,
    force: (inputConfig.force ?? false) as boolean | string,
    reasoningEffort: inputConfig.reasoningEffort as ReasoningEffort | undefined,
    schema:
      schema && Object.keys(schema).length > 0
        ? ({ name: "response", schema } satisfies JsonSchema)
        : undefined,
    maxRetries: (inputConfig.maxRetries ?? 2) as number,
    temperature: inputConfig.temperature as number | undefined,
  };
}

export async function emitResult(
  pendingId: string,
  result: {
    text: string | null;
    object: unknown;
    usage: { inputTokens: number; outputTokens: number };
  },
  parentEventId: string,
) {
  await events.emit(
    {
      text: result.text,
      object: result.object,
      usage: result.usage,
    },
    {
      complete: pendingId,
      parentEventId,
    },
  );
}

export async function storeCallState(params: {
  eventId: string;
  pendingId: string;
  inputItems: InputItem[];
  toolCalls: OpenAI.Responses.ResponseFunctionToolCall[];
  toolDefinitions: ToolDefinition[];
  force: boolean | string;
  maxOutputTokens: number;
  mcpServers: McpServer[];
  model: string;
  instructions: string | undefined;
  maxRetries: number;
  schema: JsonSchema | undefined;
  turn: number;
  reasoningEffort: ReasoningEffort | undefined;
  temperature: number | undefined;
}) {
  const { eventId, toolCalls, ...rest } = params;

  await kv.block.set({
    key: `call-${eventId}`,
    value: {
      toolCallIds: toolCalls.map((toolCall) => toolCall.call_id),
      ...rest,
    } satisfies CallState,
    ttl: 60 * 60,
  });
}

export async function loadCallState(eventId: string) {
  const { value } = await kv.block.get(`call-${eventId}`);
  return value as CallState | undefined;
}

export async function deleteCallState(eventId: string) {
  await kv.block.delete([`call-${eventId}`]);
}

export async function storeToolResult(params: {
  eventId: string;
  toolCallId: string;
  result: unknown;
  turn: number;
}) {
  const { eventId, toolCallId, result, turn } = params;

  await kv.block.set({
    key: `result-${eventId}-${turn}-${toolCallId}`,
    value: {
      toolCallId,
      result:
        typeof result === "string" ? result : JSON.stringify(result ?? null),
    },
    ttl: 60 * 5,
  });
}

export async function loadToolResults(params: {
  eventId: string;
  turn: number;
  toolCallIds: string[];
}) {
  const { eventId, turn, toolCallIds } = params;

  const results = await kv.block.list({
    keyPrefix: `result-${eventId}-${turn}-`,
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

  return { haveAllResults, toolResults };
}

export async function continueTurn(params: {
  eventId: string;
  pendingId: string;
  inputItems: InputItem[];
  toolCallIds: string[];
  toolDefinitions: ToolDefinition[];
  toolResults: Record<string, string>;
  force: boolean | string;
  model: string;
  maxOutputTokens: number;
  mcpServers: McpServer[];
  instructions: string | undefined;
  turn: number;
  maxRetries: number;
  schema: JsonSchema | undefined;
  auth: OpenAIAuth;
  blockId: string;
  reasoningEffort: ReasoningEffort | undefined;
  temperature: number | undefined;
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
  pendingId: string;
  eventId: string;
  blockId: string;
  inputItems: InputItem[];
  toolDefinitions: ToolDefinition[];
  force: boolean | string;
  model: string;
  maxOutputTokens: number;
  mcpServers: McpServer[];
  instructions: string | undefined;
  turn: number;
  auth: OpenAIAuth;
  maxRetries: number;
  schema: JsonSchema | undefined;
  reasoningEffort: ReasoningEffort | undefined;
  temperature: number | undefined;
}) {
  const {
    pendingId,
    eventId,
    inputItems,
    toolDefinitions,
    force,
    model,
    maxOutputTokens,
    mcpServers,
    instructions,
    auth,
    maxRetries,
    reasoningEffort,
    temperature,
    schema,
    blockId,
  } = params;

  try {
    // Backstop against runaway loops: cap the number of model round-trips.
    if (params.turn >= MAX_TOOL_TURNS) {
      await failPending(
        eventId,
        pendingId,
        `Reached the maximum of ${MAX_TOOL_TURNS} model turns without a final answer.`,
      );
      await deleteCallState(eventId);
      return;
    }

    const { tools, toolNames, toolBlockIds } =
      processToolDefinitions(toolDefinitions);

    const streamResult = streamResponse({
      auth,
      model,
      maxOutputTokens,
      temperature,
      input: inputItems,
      instructions,
      tools: [...tools, ...buildMcpTools(mcpServers)],
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

    const response = await streamToResponse(pendingId, streamResult);

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
            eventId,
            pendingId,
            `Response incomplete: ${reason}`,
          );
        }

        const text = extractOutputText(response);

        let object: unknown = null;
        if (schema) {
          try {
            object = JSON.parse(text);
          } catch {
            object = null;
          }
        }

        return await emitResult(
          pendingId,
          {
            text,
            object,
            usage: {
              inputTokens: response.usage?.input_tokens ?? 0,
              outputTokens: response.usage?.output_tokens ?? 0,
            },
          },
          eventId,
        );
      } finally {
        await deleteCallState(eventId);
        await clearPendingMarker(eventId);
      }
    }

    await events.updatePending(pendingId, {
      statusDescription:
        functionCalls.length === 1
          ? `Calling tool: ${joinToolNames(functionCalls, toolNames)}`
          : `Calling tools: ${joinToolNames(functionCalls, toolNames)}`,
    });

    const nextTurn = params.turn + 1;

    // Persist state before dispatching so a fast tool response always finds it.
    await storeCallState({
      eventId,
      pendingId,
      inputItems: nextInputItems,
      toolCalls: functionCalls,
      toolDefinitions,
      force: params.force,
      maxOutputTokens: params.maxOutputTokens,
      mcpServers: params.mcpServers,
      model: params.model,
      instructions: params.instructions,
      maxRetries: params.maxRetries,
      schema: params.schema,
      turn: nextTurn,
      reasoningEffort: params.reasoningEffort,
      temperature: params.temperature,
    });

    // A call whose tool has no routable block (advertised without a block, or a
    // name the model invented) can't be dispatched - answer it with an error
    // inline so the turn still resolves instead of hanging until the timeout.
    await Promise.all(
      functionCalls
        .filter((toolCall) => !toolBlockIds[toolCall.name])
        .map((toolCall) =>
          storeToolResult({
            eventId,
            toolCallId: toolCall.call_id,
            result: `Error: tool "${
              toolNames[toolCall.name] ?? toolCall.name
            }" is not available.`,
            turn: nextTurn,
          }),
        ),
    );

    await Promise.all(
      functionCalls
        .filter((toolCall) => toolBlockIds[toolCall.name])
        .map((toolCall) => {
          let parameters: unknown = {};
          try {
            parameters = toolCall.arguments
              ? JSON.parse(toolCall.arguments)
              : {};
          } catch {
            parameters = {};
          }

          return messaging.sendToBlocks({
            body: {
              blockId,
              eventId,
              parameters,
              toolCallId: toolCall.call_id,
            },
            blockIds: [toolBlockIds[toolCall.name]],
          });
        }),
    );

    return setTimeoutTimer(eventId, nextTurn);
  } catch (error) {
    await failPending(eventId, pendingId, describeFailure(error));
    await deleteCallState(eventId);
  }
}
