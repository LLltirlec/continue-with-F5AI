import { ModelProvider } from "../types.js";

export const F5Ai: ModelProvider = {
  models: [
    {
      model: "gpt-4o",
      displayName: "GPT-4o",
      contextLength: 128000,
      recommendedFor: ["chat"],
    },
    {
      model: "gpt-4o-mini",
      displayName: "GPT-4o Mini",
      contextLength: 128000,
      recommendedFor: ["chat"],
    },
    // o1
    {
      model: "o1",
      displayName: "o1",
      contextLength: 128000,
      maxCompletionTokens: 32768,
      recommendedFor: ["chat"],
    },
    {
      model: "o1-mini",
      displayName: "o1 Mini",
      contextLength: 128000,
      maxCompletionTokens: 65536,
      recommendedFor: ["chat"],
    },
    // o3
    {
      model: "o3",
      displayName: "o3",
      contextLength: 128000,
      maxCompletionTokens: 65536,
      recommendedFor: ["chat"],
    },
    {
      model: "o3-mini",
      displayName: "o3 Mini",
      contextLength: 128000,
      maxCompletionTokens: 65536,
      recommendedFor: ["chat"],
    },
    {
      model: "o4-mini",
      displayName: "o4 Mini",
      contextLength: 128000,
      maxCompletionTokens: 65536,
      recommendedFor: ["chat"],
    },
    // 4.1
    {
      model: "gpt-4.1",
      displayName: "GPT 4.1",
      contextLength: 1047576,
      maxCompletionTokens: 32768,
      recommendedFor: ["chat"],
    },
    {
      model: "gpt-4.1-mini",
      displayName: "GPT 4.1 Mini",
      contextLength: 1047576,
      maxCompletionTokens: 32768,
      recommendedFor: ["chat"],
    },
    {
      model: "gpt-4.1-nano",
      displayName: "GPT 4.1 Nano",
      contextLength: 1047576,
      maxCompletionTokens: 32768,
      recommendedFor: ["chat"],
    },
    // 4.5
    {
      model: "gpt-4.5-preview",
      displayName: "GPT 4.5 Preview",
      contextLength: 128000,
      maxCompletionTokens: 16384,
      recommendedFor: ["chat"],
    },
    // embed
    {
      model: "text-embedding-3-large",
      displayName: "Text Embedding 3-Large",
      recommendedFor: ["embed"],
    },
    {
      model: "text-embedding-3-small",
      displayName: "Text Embedding 3-Small",
    },
    {
      model: "text-embedding-ada-002",
      displayName: "Text Embedding Ada-002",
    },
  ],
  id: "f5ai",
  displayName: "F5AI",
};
