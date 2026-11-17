import OpenAI from "openai";

export const OPENAI_DEFAULT_MODEL = "gpt-6.1-terra";

export interface OpenAIAuth {
  apiKey: string;
  baseURL?: string;
  organization?: string;
}

export function resolveAuth(appConfig: Record<string, unknown>) {
  const { apiKey, baseURL, organization } = appConfig;

  if (typeof apiKey !== "string" || apiKey.trim() === "") {
    throw new Error("OpenAI API key must be a non-empty string.");
  }

  if (baseURL !== undefined && typeof baseURL !== "string") {
    throw new Error("Base URL must be a string.");
  }

  if (organization !== undefined && typeof organization !== "string") {
    throw new Error("Organization ID must be a string.");
  }

  return {
    apiKey,
    baseURL,
    organization,
  } satisfies OpenAIAuth;
}

export function defaultModel(appConfig: Record<string, unknown>) {
  const { defaultModel } = appConfig;

  if (defaultModel !== undefined && typeof defaultModel !== "string") {
    throw new Error("Default model must be a string.");
  }

  return defaultModel || OPENAI_DEFAULT_MODEL;
}

export function createClient(auth: OpenAIAuth, maxRetries?: number) {
  return new OpenAI({
    apiKey: auth.apiKey,
    baseURL: auth.baseURL,
    organization: auth.organization,
    maxRetries,
  });
}
