import { AppBlock, events } from "@slflows/sdk/v1";
import { resolveAuth, createClient } from "../client";

export const httpRequest: AppBlock = {
  name: "HTTP Request",
  description:
    "Make a direct request to the OpenAI API. Use this as an escape hatch for any API endpoint not covered by dedicated blocks.",
  category: "Request",
  inputs: {
    default: {
      config: {
        method: {
          name: "HTTP Method",
          description: "HTTP method for the request.",
          type: {
            enum: ["GET", "POST", "PUT", "PATCH", "DELETE"],
          },
          required: true,
        },
        path: {
          name: "Path",
          description:
            'The path appended to the base URL (e.g., "/models" or "/responses").',
          type: "string",
          required: true,
        },
        baseURL: {
          name: "Base URL",
          description:
            'Override the base URL for this request. Defaults to the app\'s base URL ("https://api.openai.com/v1").',
          type: "string",
          required: false,
        },
        queryParams: {
          name: "Query Parameters",
          description: "Optional query parameters as key-value pairs.",
          type: {
            type: "object",
            additionalProperties: true,
          },
          required: false,
        },
        headers: {
          name: "Headers",
          description: "Optional extra HTTP headers as key-value pairs.",
          type: {
            type: "object",
            additionalProperties: { type: "string" },
          },
          required: false,
        },
        requestBody: {
          name: "Body",
          description: "Optional request body as key-value pairs.",
          type: {
            type: "object",
            additionalProperties: true,
          },
          required: false,
        },
      },
      onEvent: async (input) => {
        const auth = resolveAuth(input.app.config);
        const { method, path, baseURL, requestBody, queryParams, headers } =
          input.event.inputConfig;

        if (typeof method !== "string") {
          throw new Error("HTTP method must be a string.");
        }

        if (typeof path !== "string") {
          throw new Error("Path must be a string.");
        }

        if (baseURL !== undefined && typeof baseURL !== "string") {
          throw new Error("Base URL must be a string.");
        }

        const client = createClient(baseURL ? { ...auth, baseURL } : auth);
        const options = { query: queryParams, headers };

        let data: unknown;
        switch (method.toUpperCase()) {
          case "GET":
            data = await client.get(path, options);
            break;
          case "DELETE":
            data = await client.delete(path, options);
            break;
          case "POST":
            data = await client.post(path, { ...options, body: requestBody });
            break;
          case "PUT":
            data = await client.put(path, { ...options, body: requestBody });
            break;
          case "PATCH":
            data = await client.patch(path, { ...options, body: requestBody });
            break;
          default:
            throw new Error(`Unsupported HTTP method: ${method}`);
        }

        await events.emit(data);
      },
    },
  },
  outputs: {
    result: {
      name: "Result",
      description: "The API response.",
      default: true,
      possiblePrimaryParents: ["default"],
      type: {
        type: "object",
        additionalProperties: true,
      },
    },
  },
};
