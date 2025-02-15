import { ModelProvider } from "../types.js";

export const F5Ai: ModelProvider = {
  models: [
    // gpt-4o
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
      contextLength: 200000,
      maxCompletionTokens: 100000,
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
      model: "o3-mini",
      displayName: "o3 Mini",
      contextLength: 200000,
      maxCompletionTokens: 100000,
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
