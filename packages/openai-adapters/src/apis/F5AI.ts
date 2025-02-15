import { streamSse } from "@continuedev/fetch";
import fetch from "node-fetch";
import { OpenAI } from "openai/index";
import {
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionCreateParamsStreaming,
  Completion,
  CompletionCreateParamsNonStreaming,
  CompletionCreateParamsStreaming,
  Model,
} from "openai/resources/index";
import { z } from "zod";
import { OpenAIConfigSchema } from "../types.js";
import { maybeCustomFetch } from "../util.js";
import {
  BaseLlmApi,
  CreateRerankResponse,
  FimCreateParamsStreaming,
  RerankCreateParams,
} from "./base.js";

export class F5AIApi implements BaseLlmApi {
  openai: OpenAI;
  apiBase: string = "https://dev.api.f5ai.ru/v1/";

  constructor(protected config: z.infer<typeof OpenAIConfigSchema>) {
    this.apiBase = config.apiBase ?? this.apiBase;
    this.openai = new OpenAI({
      apiKey: config.apiKey,
      baseURL: this.apiBase,
      fetch: maybeCustomFetch(config.requestOptions),
    });
  }

  async chatCompletionNonStream(
    body: ChatCompletionCreateParamsNonStreaming,
    signal: AbortSignal,
  ): Promise<ChatCompletion> {
    const response = await this.openai.chat.completions.create(body, {
      signal,
    });
    return response;
  }

  async *chatCompletionStream(
    body: ChatCompletionCreateParamsStreaming,
    signal: AbortSignal,
  ): AsyncGenerator<ChatCompletionChunk> {
    // Принудительно устанавливаем stream: false
    const response = await this.openai.chat.completions.create(
      { ...body, stream: false },
      { signal }
    );

    // Эмулируем единый "чанк" с полным ответом
    yield {
      id: response.id,
      object: "chat.completion.chunk",
      created: response.created,
      model: response.model,
      choices: response.choices.map(choice => ({
        index: choice.index,
        delta: {
          role: choice.message.role,
          content: choice.message.content || "",
        },
        finish_reason: choice.finish_reason,
      })),
    };
  }

  async completionNonStream(
    body: CompletionCreateParamsNonStreaming,
    signal: AbortSignal,
  ): Promise<Completion> {
    const response = await this.openai.completions.create(body, { signal });
    return response;
  }

  async *completionStream(
    body: CompletionCreateParamsStreaming,
    signal: AbortSignal,
  ): AsyncGenerator<Completion> {
    const response = await this.openai.completions.create(
      { ...body, stream: false },
      { signal }
    );

    // Возвращаем как единый объект
    yield response;
  }

  async *fimStream(
    body: FimCreateParamsStreaming,
    signal: AbortSignal,
  ): AsyncGenerator<ChatCompletionChunk> {
    const endpoint = new URL("fim/completions", this.apiBase);

    const resp = await fetch(endpoint, {
      method: "POST",
      body: JSON.stringify({
        ...body,
        stream: false // Всегда false
      }),
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.config.apiKey}`,
      },
      signal,
    });

    const data = await resp.json();

    // Единый чанк с преобразованием структуры
    yield {
      id: data.id || `fim-${Date.now()}`,
      object: "chat.completion.chunk",
      created: data.created || Math.floor(Date.now()/1000),
      model: data.model || body.model,
      choices: [{
        index: 0,
        delta: {
          content: data.choices?.[0]?.text || "",
        },
        finish_reason: data.choices?.[0]?.finish_reason || "stop",
      }],
    };
  }

  async embed(
    body: OpenAI.Embeddings.EmbeddingCreateParams,
  ): Promise<OpenAI.Embeddings.CreateEmbeddingResponse> {
    const response = await this.openai.embeddings.create(body);
    return response;
  }

  async rerank(body: RerankCreateParams): Promise<CreateRerankResponse> {
    const endpoint = new URL("rerank", this.apiBase);
    const response = await fetch(endpoint, {
      method: "POST",
      body: JSON.stringify(body),
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
        "X-Auth-Token": this.config.apiKey ?? "",
        Authorization: `Bearer ${this.config.apiKey}`,
      },
    });
    const data = await response.json();
    return data as any;
  }

  async list(): Promise<Model[]> {
    return (await this.openai.models.list()).data;
  }
}
