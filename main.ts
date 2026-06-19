import { defineApp } from "@slflows/sdk/v1";

import { blocks } from "./blocks/index";
import {
  OPENAI_DEFAULT_MODEL,
  createClient,
  defaultModel,
  resolveAuth,
} from "./client";

export const app = defineApp({
  name: "OpenAI",
  installationInstructions: [
    "To install:",
    "",
    "1. Visit https://platform.openai.com/api-keys and create an API key",
    "2. Paste it into the 'OpenAI API Key' config field",
    `3. Optionally set a 'Default model' (defaults to ${OPENAI_DEFAULT_MODEL})`,
    "",
    "Then click 'Confirm' to complete the installation.",
  ].join("\n"),
  config: {
    apiKey: {
      name: "OpenAI API Key",
      description: "Your OpenAI API key (starts with 'sk-').",
      type: "string",
      required: true,
      sensitive: true,
    },
    defaultModel: {
      name: "Default model",
      description: `The default model to use for API calls. If left empty, defaults to '${OPENAI_DEFAULT_MODEL}'.`,
      type: "string",
      required: false,
    },
    baseURL: {
      name: "Base URL",
      description:
        "Optional custom base URL for OpenAI-compatible endpoints. Leave empty to use the default OpenAI API.",
      type: "string",
      required: false,
    },
    organization: {
      name: "Organization ID",
      description: "Optional OpenAI organization ID to scope API usage.",
      type: "string",
      required: false,
    },
  },
  blocks,
  async onSync(input) {
    const auth = resolveAuth(input.app.config);
    const client = createClient(auth);
    const model = defaultModel(input.app.config);

    await client.responses.create({
      model,
      input: "hi",
      max_output_tokens: 16,
    });

    return { newStatus: "ready" };
  },
});
