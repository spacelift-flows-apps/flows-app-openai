import { AppBlock } from "@slflows/sdk/v1";

async function listMcpTools(
  url: string,
  authorizationToken: string | undefined,
) {
  const { Client } = await import("@modelcontextprotocol/sdk/client/index.js");

  const baseUrl = new URL(url);
  const requestInit = authorizationToken
    ? { headers: { Authorization: `Bearer ${authorizationToken}` } }
    : undefined;

  let client: typeof Client.prototype;

  try {
    const { StreamableHTTPClientTransport } =
      await import("@modelcontextprotocol/sdk/client/streamableHttp.js");

    client = new Client({ name: "streamable-http-client", version: "1.0.0" });
    await client.connect(
      new StreamableHTTPClientTransport(baseUrl, { requestInit }),
    );
  } catch {
    const { SSEClientTransport } =
      await import("@modelcontextprotocol/sdk/client/sse.js");

    client = new Client({ name: "sse-client", version: "1.0.0" });
    await client.connect(new SSEClientTransport(baseUrl, { requestInit }));
  }

  const { tools } = await client.listTools();
  return tools;
}

export const remoteMcpServer: AppBlock = {
  name: "Remote MCP server",
  category: "Definition",
  description:
    "Configure a remote MCP server to be utilized by the model when generating responses.",
  config: {
    url: {
      name: "URL",
      description: "The URL of the MCP server. Must start with https://",
      type: "string",
      required: true,
    },
    authorizationToken: {
      name: "Authorization token",
      description: "OAuth authorization token if required by the MCP server.",
      type: "string",
      required: false,
      sensitive: true,
    },
    toolAccess: {
      name: "Tool access",
      description:
        "Which of the server's tools the model may call. Choose 'Only selected tools' to restrict to the tools picked below.",
      type: {
        type: "string",
        oneOf: [
          { const: "all", title: "All tools" },
          { const: "selected", title: "Only selected tools" },
          { const: "none", title: "No tools" },
        ],
      },
      required: false,
      default: "all",
    },
    allowedTools: {
      name: "Allowed tools",
      description:
        "The tools the model may call. Only applies when Tool access is set to 'Only selected tools'.",
      type: {
        type: "array",
        items: {
          type: "string",
        },
      },
      required: false,
      suggestValues: async ({ block, searchPhrase }) => {
        const { url, authorizationToken } = block.config;

        if (!url) {
          return {
            suggestedValues: [],
            message: "Set the URL to see available tools.",
          };
        }

        try {
          const tools = await listMcpTools(url, authorizationToken);
          const query = searchPhrase?.toLowerCase();

          return {
            suggestedValues: tools.flatMap((tool) =>
              query && !tool.name.toLowerCase().includes(query)
                ? []
                : [{ label: tool.name, value: tool.name }],
            ),
          };
        } catch {
          return {
            suggestedValues: [],
            message: "Could not connect to the MCP server.",
          };
        }
      },
    },
  },

  onSync: async (input) => {
    const config = input.block.config;
    const toolAccess = config.toolAccess ?? "all";

    const allowedTools =
      toolAccess === "all"
        ? undefined
        : toolAccess === "none"
          ? []
          : (config.allowedTools ?? []);

    try {
      const tools = await listMcpTools(config.url, config.authorizationToken);
      const toolNames = tools.map((tool) => tool.name);

      if (allowedTools) {
        for (const toolName of allowedTools) {
          if (!toolNames.includes(toolName)) {
            return {
              newStatus: "failed",
              customStatusDescription: `Tool "${toolName}" not found`,
            };
          }
        }
      }

      return {
        newStatus: "ready",
        signalUpdates: {
          definition: {
            name: input.block.name,
            url: config.url,
            authorizationToken: config.authorizationToken,
            allowedTools,
          },
          availableTools: tools.map((tool) => ({
            name: tool.name,
            description: tool.description,
          })),
        },
      };
    } catch {
      return {
        newStatus: "failed",
        customStatusDescription: "Failed to connect",
      };
    }
  },
};
