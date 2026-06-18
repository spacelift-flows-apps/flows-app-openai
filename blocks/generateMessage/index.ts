import { AppBlock } from "@slflows/sdk/v1";
import { randomUUID } from "node:crypto";

import {
  executeTurn,
  validateConfig,
  loadCallState,
  deleteCallState,
  storeToolResult,
  loadToolResults,
  continueTurn,
} from "./utils";
import {
  setTimeoutTimer,
  clearTimeoutTimer,
  tryAcquireProcessingLock,
  releaseProcessingLock,
  getOrCreatePending,
  failPending,
} from "../shared";
import { resolveAuth } from "../../client";

export const generateMessage: AppBlock = {
  name: "Generate response",
  category: "Core",
  description: "Generates a response based on provided input and parameters.",
  inputs: {
    default: {
      config: {
        model: {
          name: "Model",
          description: "The model that will generate the response.",
          type: "string",
          required: false,
        },
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
        schema: {
          name: "Schema",
          description:
            "JSON schema to generate a structured object from. When set, the model is constrained to produce output matching this schema.",
          type: {
            type: "jsonSchemaObject",
          },
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
        toolDefinitions: {
          name: "Tools",
          description:
            "Array of tool blocks to use.\nYou can specify them like this:\n```javascript\n[signals.toolBlock.definition, ...]\n```",
          type: {
            type: "array",
            items: {
              type: "object",
              properties: {
                blockId: {
                  type: "string",
                  description: "ID of the tool definition block",
                },
                name: {
                  type: "string",
                  description: "Name of the tool",
                },
                description: {
                  type: "string",
                  description: "Description of the tool",
                },
                schema: {
                  type: "object",
                  description: "Schema of the tool",
                },
              },
              required: ["name", "description", "schema"],
            },
          },
          required: false,
        },
        mcpServers: {
          name: "Remote MCP servers",
          description: "Array of remote MCP servers to use",
          type: {
            type: "array",
            items: {
              type: "object",
              properties: {
                name: {
                  type: "string",
                  description: "Name of the MCP server",
                },
                url: {
                  type: "string",
                  description: "URL of the MCP server",
                },
                authorizationToken: {
                  type: "string",
                  description: "Authorization token for the MCP server",
                },
                allowedTools: {
                  type: "array",
                  description: "Allowed tools for the MCP server",
                },
              },
              required: ["name", "url"],
            },
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
          mcpServers,
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
        } = validateConfig(input.app.config, input.event.inputConfig);

        const pendingId = await getOrCreatePending(
          input.event.id,
          "Calling OpenAI model...",
        );

        return executeTurn({
          pendingId,
          eventId: input.event.id,
          blockId: input.block.id,
          inputItems: [{ role: "user", content: prompt }],
          toolDefinitions,
          force,
          model,
          maxOutputTokens,
          mcpServers,
          instructions,
          turn: 0,
          auth,
          maxRetries,
          schema,
          reasoningEffort,
          temperature,
        });
      },
    },
  },

  outputs: {
    result: {
      name: "Result",
      description: "The generated response",
      default: true,
      possiblePrimaryParents: ["default"],
      type: {
        type: "object",
        properties: {
          text: { type: "string", description: "The generated text" },
          object: {
            anyOf: [
              { type: "object", additionalProperties: true },
              { type: "null" },
            ],
            description: "The generated object if a schema is provided",
          },
          usage: {
            type: "object",
            properties: {
              inputTokens: { type: "number" },
              outputTokens: { type: "number" },
            },
          },
        },
        required: ["usage"],
      },
    },
  },

  onTimer: async (input) => {
    const { id: eventId, turn: timerTurn } = input.timer.payload;

    const state = await loadCallState(eventId);

    // Ignore a timer that has been superseded by a newer turn or a finished call.
    if (!state || state.turn !== timerTurn) {
      return;
    }

    const lockId = randomUUID();
    const acquired = await tryAcquireProcessingLock(
      eventId,
      state.turn,
      lockId,
    );

    if (!acquired) {
      return;
    }

    try {
      const { haveAllResults, toolResults } = await loadToolResults({
        eventId,
        turn: state.turn,
        toolCallIds: state.toolCallIds,
      });

      if (!haveAllResults) {
        await failPending(
          eventId,
          state.pendingId,
          "Timeout waiting for tool results",
        );
        await deleteCallState(eventId);
        return;
      }

      await continueTurn({
        eventId,
        blockId: input.block.id,
        pendingId: state.pendingId,
        inputItems: state.inputItems,
        toolCallIds: state.toolCallIds,
        toolDefinitions: state.toolDefinitions,
        toolResults,
        force: state.force,
        model: state.model,
        maxOutputTokens: state.maxOutputTokens,
        mcpServers: state.mcpServers,
        instructions: state.instructions,
        turn: state.turn,
        maxRetries: state.maxRetries,
        schema: state.schema,
        reasoningEffort: state.reasoningEffort,
        temperature: state.temperature,
        auth: resolveAuth(input.app.config),
      });
    } finally {
      await releaseProcessingLock(eventId, state.turn, lockId);
    }
  },

  onInternalMessage: async (input) => {
    const { eventId, toolCallId, result } = input.message.body;

    const state = await loadCallState(eventId);

    if (!state) {
      return;
    }

    // Store each result under its own key so concurrent results don't clobber.
    await storeToolResult({ eventId, toolCallId, result, turn: state.turn });

    // Only the handler that wins the per-turn lock advances; the rest just
    // recorded their result above and return.
    const lockId = randomUUID();
    const acquired = await tryAcquireProcessingLock(
      eventId,
      state.turn,
      lockId,
    );

    if (!acquired) {
      return;
    }

    try {
      await clearTimeoutTimer(eventId);

      const { haveAllResults, toolResults } = await loadToolResults({
        eventId,
        turn: state.turn,
        toolCallIds: state.toolCallIds,
      });

      if (!haveAllResults) {
        await setTimeoutTimer(eventId, state.turn);
        return;
      }

      await continueTurn({
        eventId,
        blockId: input.block.id,
        pendingId: state.pendingId,
        inputItems: state.inputItems,
        toolCallIds: state.toolCallIds,
        toolDefinitions: state.toolDefinitions,
        toolResults,
        force: state.force,
        model: state.model,
        maxOutputTokens: state.maxOutputTokens,
        mcpServers: state.mcpServers,
        instructions: state.instructions,
        turn: state.turn,
        maxRetries: state.maxRetries,
        schema: state.schema,
        reasoningEffort: state.reasoningEffort,
        temperature: state.temperature,
        auth: resolveAuth(input.app.config),
      });
    } finally {
      await releaseProcessingLock(eventId, state.turn, lockId);
    }
  },
};
