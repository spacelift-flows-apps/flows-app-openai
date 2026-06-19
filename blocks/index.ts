import { generateMessage } from "./generateMessage/index";
import { toolDefinition } from "./toolDefinition";
import { remoteMcpServer } from "./remoteMcpServer";
import { httpRequest } from "./httpRequest";
import { agent } from "./agent/index";
import { agentToolResult } from "./agentToolResult";

export const blocks = {
  agent,
  agentToolResult,
  generateMessage,
  toolDefinition,
  remoteMcpServer,
  httpRequest,
} as const;
