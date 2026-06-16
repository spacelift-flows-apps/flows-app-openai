import { AppBlock, AppBlockComponentOutput, events } from "@slflows/sdk/v1";
import { randomUUID } from "node:crypto";

import {
  executeTurn,
  loadCallState,
  deleteCallState,
  storeToolResult,
  loadToolResults,
  continueTurn,
  validateConfig,
  getToolDefinitionOutputKey,
} from "./utils";
import {
  setTimeoutTimer,
  clearTimeoutTimer,
  tryAcquireProcessingLock,
  releaseProcessingLock,
} from "../shared";
import { resolveAuth } from "../../client";

export const agent: AppBlock = {
  name: "Agent",
  category: "Core",
  description: "An agent that can call tools and generate structured output.",
  autoconfirm: true,
  config: {
    model: {
      name: "Model",
      type: "string",
      description:
        "The model to use for the agent. Defaults to the default model set in the app config.",
      required: false,
    },
    toolDefinitions: {
      name: "Tools",
      description:
        "Array of tools the agent can call. Each tool becomes a dedicated output on this block.",
      type: {
        type: "array",
        items: {
          type: "object",
          properties: {
            name: {
              title: "Name",
              type: "string",
              description: "Name of the tool",
            },
            description: {
              title: "Description",
              type: "string",
              description: "Description of what the tool does",
            },
            schema: {
              title: "Schema",
              type: "jsonSchemaObject",
              description: "Schema of the tool's input",
            },
          },
          required: ["name", "description"],
        },
      },
      required: false,
    },
    outputSchema: {
      name: "Output schema",
      description:
        "Used to constrain the output of the model to follow a specific schema, ensuring valid, parseable output for downstream processing.",
      type: {
        type: "jsonSchemaObject",
      },
      required: false,
    },
  },
  inputs: {
    default: {
      config: {
        prompt: {
          name: "Prompt",
          description: "The input to the model.",
          type: "string",
          required: true,
        },
        systemPrompt: {
          name: "System prompt",
          description:
            "Instructions inserted into the model's context, such as specifying a particular goal or role.",
          type: "string",
          required: false,
        },
        maxTokens: {
          name: "Max output tokens",
          description:
            "Upper bound for the number of tokens generated for a response, including visible output and reasoning tokens.",
          type: "number",
          default: 4096,
          required: false,
        },
        maxRetries: {
          name: "Max retries",
          description: "How many times to retry a failed request.",
          type: "number",
          required: false,
        },
        reasoningEffort: {
          name: "Reasoning effort",
          description:
            "Constrains reasoning effort for reasoning models. Higher effort produces more thorough responses at higher cost and latency. Leave unset to use the model default.",
          type: {
            enum: ["none", "minimal", "low", "medium", "high", "xhigh"],
          },
          required: false,
        },
        force: {
          name: "Force",
          description:
            "Force the model to call a tool. Provide a name to call a specific tool or `true` to always call any tool.",
          type: {
            anyOf: [
              {
                type: "string",
                description: "The name of the tool to call.",
              },
              { type: "boolean", description: "Always call any tool." },
            ],
          },
          required: false,
          default: false,
        },
        temperature: {
          name: "Temperature",
          description:
            "Sampling temperature between 0 and 2. Lower values are more focused and deterministic, higher values more random. Not supported by reasoning models.",
          type: "number",
          required: false,
        },
      },
      onEvent: async (input) => {
        const {
          toolDefinitions,
          prompt,
          model,
          maxOutputTokens,
          instructions,
          force,
          auth,
          schema,
          maxRetries,
          temperature,
          reasoningEffort,
        } = validateConfig(
          input.app.config,
          input.block.config,
          input.event.inputConfig,
        );

        const pendingId = await events.createPending({
          statusDescription: "Calling OpenAI model...",
        });

        const executionId = randomUUID();

        return executeTurn({
          executionId,
          blockId: input.block.id,
          pendingId,
          eventIds: [],
          inputItems: [{ role: "user", content: prompt }],
          toolDefinitions,
          force,
          model,
          maxOutputTokens,
          instructions,
          turn: 0,
          auth,
          maxRetries,
          schema,
          reasoningEffort,
          temperature,
          originalEventId: input.event.id,
        });
      },
    },
  },

  onInternalMessage: async (input) => {
    const { executionId, toolCallId, result, eventId } = input.message.body;
    const state = await loadCallState(executionId);

    if (!state) {
      return;
    }

    await storeToolResult({
      executionId,
      toolCallId,
      result,
      turn: state.turn,
      eventId,
    });

    const lockId = randomUUID();
    const acquired = await tryAcquireProcessingLock(
      executionId,
      state.turn,
      lockId,
    );

    if (!acquired) {
      return;
    }

    try {
      await clearTimeoutTimer(executionId);

      const { haveAllResults, toolResults, eventIds } = await loadToolResults({
        executionId,
        turn: state.turn,
        toolCallIds: state.toolCallIds,
      });

      if (!haveAllResults) {
        await setTimeoutTimer(executionId, state.turn);
        return;
      }

      await continueTurn({
        executionId,
        blockId: state.blockId,
        eventIds,
        pendingId: state.pendingId,
        inputItems: state.inputItems,
        toolCallIds: state.toolCallIds,
        toolDefinitions: state.toolDefinitions,
        toolResults,
        force: state.force,
        model: state.model,
        maxOutputTokens: state.maxOutputTokens,
        instructions: state.instructions,
        turn: state.turn,
        maxRetries: state.maxRetries,
        schema: state.schema,
        reasoningEffort: state.reasoningEffort,
        temperature: state.temperature,
        auth: resolveAuth(input.app.config),
        originalEventId: state.originalEventId,
      });
    } finally {
      await releaseProcessingLock(executionId, state.turn, lockId);
    }
  },

  onTimer: async (input) => {
    const { id: executionId, turn: timerTurn } = input.timer.payload;

    const state = await loadCallState(executionId);

    if (!state || state.turn !== timerTurn) {
      return;
    }

    const lockId = randomUUID();
    const acquired = await tryAcquireProcessingLock(
      executionId,
      state.turn,
      lockId,
    );

    if (!acquired) {
      return;
    }

    try {
      const { haveAllResults, toolResults, eventIds } = await loadToolResults({
        executionId,
        turn: state.turn,
        toolCallIds: state.toolCallIds,
      });

      if (!haveAllResults) {
        await events.cancelPending(
          state.pendingId,
          "Timeout waiting for tool results",
        );
        await deleteCallState(executionId);
        return;
      }

      await continueTurn({
        executionId,
        blockId: state.blockId,
        eventIds,
        pendingId: state.pendingId,
        inputItems: state.inputItems,
        toolCallIds: state.toolCallIds,
        toolDefinitions: state.toolDefinitions,
        toolResults,
        force: state.force,
        model: state.model,
        maxOutputTokens: state.maxOutputTokens,
        instructions: state.instructions,
        turn: state.turn,
        maxRetries: state.maxRetries,
        schema: state.schema,
        reasoningEffort: state.reasoningEffort,
        temperature: state.temperature,
        auth: resolveAuth(input.app.config),
        originalEventId: state.originalEventId,
      });
    } finally {
      await releaseProcessingLock(executionId, state.turn, lockId);
    }
  },

  onSync: async (input) => {
    const outputUpdates: Record<
      string,
      Partial<AppBlockComponentOutput> | null
    > = {};

    for (const outputKey of Object.keys(input.block.outputs)) {
      const output = input.block.outputs[outputKey];

      if (output && output.modified) {
        outputUpdates[outputKey] = null;
      }
    }

    outputUpdates["result"] = {
      name: "Result",
      default: true,
      possiblePrimaryParents: ["default"],
      type: {
        type: "object",
        properties: {
          output: input.block.config.outputSchema ?? {
            type: "string",
            description: "The generated response",
          },
          usage: {
            type: "object",
            properties: {
              inputTokens: { type: "number" },
              outputTokens: { type: "number" },
            },
          },
        },
        required: ["output", "usage"],
      },
    };

    let order = 10;

    for (const toolDefinition of input.block.config.toolDefinitions ?? []) {
      outputUpdates[getToolDefinitionOutputKey(toolDefinition.name)] = {
        name: toolDefinition.name,
        description: toolDefinition.description,
        type: {
          type: "object",
          properties: {
            parameters: toolDefinition.schema,
            invocationId: { type: "string" },
          },
          required: ["parameters", "invocationId"],
        },
        possiblePrimaryParents: ["default"],
        order: order++,
        secondary: true,
      };
    }

    return {
      newStatus: "ready",
      outputUpdates:
        Object.keys(outputUpdates).length > 0 ? outputUpdates : undefined,
    };
  },
};
