import { OpenAI } from "openai/index";
import {
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionCreateParams,
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionCreateParamsStreaming,
  Completion,
  CompletionCreateParamsNonStreaming,
  CompletionCreateParamsStreaming,
  Model,
} from "openai/resources/index";
import { z } from "zod";
import { OpenAIConfigSchema } from "../types.js";
import { customFetch } from "../util.js";
import {
  BaseLlmApi,
  CreateRerankResponse,
  FimCreateParamsStreaming,
  RerankCreateParams,
} from "./base.js";

export class F5AIApi implements BaseLlmApi {
  openai: OpenAI;
  apiBase: string = "https://api.f5ai.ru/v1/";

  constructor(protected config: z.infer<typeof OpenAIConfigSchema>) {
    this.apiBase = config.apiBase ?? this.apiBase;
    this.openai = new OpenAI({
      apiKey: config.apiKey,
      baseURL: this.apiBase,
      fetch: customFetch(config.requestOptions),
    });
  }

  modifyChatBody<T extends ChatCompletionCreateParams>(body: T): T {
    // o-series models - only apply for official OpenAI API
    const isOfficialOpenAIAPI = this.apiBase === "https://api.f5ai.ru/v1/";
    if (isOfficialOpenAIAPI) {
      if (body.model.startsWith("o")) {
        // a) use max_completion_tokens instead of max_tokens
        body.max_completion_tokens = body.max_tokens;
        body.max_tokens = undefined;

        // b) use "developer" message role rather than "system"
        body.messages = body.messages.map((message) => {
          if (message.role === "system") {
            return { ...message, role: "developer" } as any;
          }
          return message;
        });
      }
      if (body.tools?.length && !body.model.startsWith("o1") && !body.model.startsWith("o3") && !body.model.startsWith("o4")) {
        body.parallel_tool_calls = false;
      }
    }
    // Принудительно устанавливаем stream в false
    body.stream = false;
    return body;
  }

  async chatCompletionNonStream(
    body: ChatCompletionCreateParamsNonStreaming,
    signal: AbortSignal,
  ): Promise<ChatCompletion> {
    const response = await this.openai.chat.completions.create(
      this.modifyChatBody(body),
      {
        signal,
      },
    );
    return response;
  }

  async *chatCompletionStream(
    body: ChatCompletionCreateParamsStreaming,
    signal: AbortSignal,
  ): AsyncGenerator<ChatCompletionChunk, any, unknown> {
    // Эмулируем streaming поведение через non-streaming запрос
    const nonStreamBody: ChatCompletionCreateParamsNonStreaming = {
      ...body,
      stream: false,
    };
    
    const response = await this.chatCompletionNonStream(nonStreamBody, signal);
    
    // Конвертируем обычный ответ в формат chunk для совместимости
    const chunk: ChatCompletionChunk = {
      id: response.id,
      object: "chat.completion.chunk",
      created: response.created,
      model: response.model,
      choices: response.choices.map(choice => ({
        index: choice.index,
        delta: {
          role: choice.message?.role,
          content: choice.message?.content,
          tool_calls: choice.message?.tool_calls,
        },
        logprobs: choice.logprobs,
        finish_reason: choice.finish_reason,
      })),
      usage: response.usage,
    };
    
    yield chunk;
  }

  async completionNonStream(
    body: CompletionCreateParamsNonStreaming,
    signal: AbortSignal,
  ): Promise<Completion> {
    const modifiedBody = { ...body, stream: false };
    const response = await this.openai.completions.create(modifiedBody, { signal });
    return response;
  }

  async *completionStream(
    body: CompletionCreateParamsStreaming,
    signal: AbortSignal,
  ): AsyncGenerator<Completion, any, unknown> {
    // Эмулируем streaming поведение через non-streaming запрос
    const nonStreamBody: CompletionCreateParamsNonStreaming = {
      ...body,
      stream: false,
    };
    
    const response = await this.completionNonStream(nonStreamBody, signal);
    yield response;
  }

  async *fimStream(
    body: FimCreateParamsStreaming,
    signal: AbortSignal,
  ): AsyncGenerator<ChatCompletionChunk, any, unknown> {
    const endpoint = new URL("fim/completions", this.apiBase);
    const resp = await customFetch(this.config.requestOptions)(endpoint, {
      method: "POST",
      body: JSON.stringify({
        model: body.model,
        prompt: body.prompt,
        suffix: body.suffix,
        max_tokens: body.max_tokens,
        max_completion_tokens: (body as any).max_completion_tokens,
        temperature: body.temperature,
        top_p: body.top_p,
        frequency_penalty: body.frequency_penalty,
        presence_penalty: body.presence_penalty,
        stop: body.stop,
        stream: false, // Принудительно отключаем стриминг
      }),
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
        "X-Auth-Token": this.config.apiKey ?? "",
        Authorization: `Bearer ${this.config.apiKey}`,
      },
      signal,
    });
    
    const data = await resp.json();
    
    // Эмулируем chunk формат для совместимости
    if (data.choices && data.choices.length > 0) {
      const chunk: ChatCompletionChunk = {
        id: data.id || "",
        object: "chat.completion.chunk",
        created: data.created || Date.now(),
        model: data.model || body.model,
        choices: data.choices.map((choice: any, index: number) => ({
          index: index,
          delta: {
            content: choice.text || choice.message?.content,
          },
          finish_reason: choice.finish_reason,
        })),
      };
      yield chunk;
    }
  }

  async embed(
    body: OpenAI.Embeddings.EmbeddingCreateParams,
  ): Promise<OpenAI.Embeddings.CreateEmbeddingResponse> {
    const response = await this.openai.embeddings.create(body);
    return response;
  }

  async rerank(body: RerankCreateParams): Promise<CreateRerankResponse> {
    const endpoint = new URL("rerank", this.apiBase);
    const response = await customFetch(this.config.requestOptions)(endpoint, {
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