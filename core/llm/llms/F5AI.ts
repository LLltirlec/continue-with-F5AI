import {
  ChatCompletionCreateParams,
  ChatCompletionMessageParam,
} from "openai/resources/index";

import {
  F5AIChatCompletionCreateParams,
} from "./openai-overrides";

import {
  ChatMessage,
  CompletionOptions,
  LLMOptions,
  Tool,
} from "../../index.js";
import { renderChatMessage } from "../../util/messageContent.js";
import { BaseLLM } from "../index.js";
import {
  fromChatCompletionChunk,
  LlmApiRequestType,
  toChatBody,
} from "../openaiTypeConverters.js";
import { streamSse } from "../stream.js";

const NON_CHAT_MODELS = [
  "text-davinci-002",
  "text-davinci-003",
  "code-davinci-002",
  "text-ada-001",
  "text-babbage-001",
  "text-curie-001",
  "davinci",
  "curie",
  "babbage",
  "ada",
];

const CHAT_ONLY_MODELS = [
  "gpt-4o",
  "gpt-4o-mini",
  "o1",
  "o1-mini",
  "o3-mini",
];

const formatMessageForO1 = (messages: ChatCompletionMessageParam[]) => {
  return messages?.map((message: any) => {
    if (message?.role === "system") {
      return {
        ...message,
        role: "user",
      };
    }

    return message;
  });
};

class F5AI extends BaseLLM {
  public useLegacyCompletionsEndpoint: boolean | undefined = undefined;

  constructor(options: LLMOptions) {
    super(options);
    this.useLegacyCompletionsEndpoint = options.useLegacyCompletionsEndpoint;
    this.apiVersion = options.apiVersion ?? "2023-07-01-preview";
  }

  static providerName = "f5ai";
  static defaultOptions: Partial<LLMOptions> | undefined = {
    apiBase: "https://dev.api.f5ai.ru/v1/",
    maxEmbeddingBatchSize: 128,
  };

  protected useOpenAIAdapterFor: (LlmApiRequestType | "*")[] = [
    "chat",
    "embed",
    "list",
    "rerank",
    "streamChat",
    "streamFim",
  ];

  protected _convertModelName(model: string): string {
    return model;
  }

  private isO1Model(model?: string): boolean {
    return !!model && model.startsWith("o1");
  }

  private isO3Model(model?: string): boolean {
    return !!model && model.startsWith("o3");
  }

  private isGPTModel(model?: string): boolean {
    return !!model && model.startsWith("gpt");
  }

  protected supportsPrediction(model: string): boolean {
    const SUPPORTED_MODELS = ["gpt-4o-mini", "gpt-4o"];
    return SUPPORTED_MODELS.some((m) => model.includes(m));
  }

  private convertTool(tool: Tool): any {
    return {
      type: tool.type,
      function: {
        name: tool.function.name,
        description: tool.function.description,
        parameters: tool.function.parameters,
        strict: tool.function.strict,
      },
    };
  }

  protected getMaxStopWords(): number {
    const url = new URL(this.apiBase!);

    if (this.maxStopWords !== undefined) {
      return this.maxStopWords;
    } else if (url.host === "api.deepseek.com") {
      return 16;
    } else if (
      url.port === "1337" ||
      url.host === "dev.api.f5ai.ru" ||
      url.host === "api.openai.com" ||
      url.host === "api.groq.com" ||
      this.apiType === "azure"
    ) {
      return 4;
    } else {
      return Infinity;
    }
  }

  protected _convertArgs(
    options: CompletionOptions,
    messages: ChatMessage[],
  ): F5AIChatCompletionCreateParams {
    const finalOptions = toChatBody(messages, options) as F5AIChatCompletionCreateParams;

    finalOptions.stop = options.stop?.slice(0, this.getMaxStopWords());

    // OpenAI o1-preview and o1-mini:
    if (this.isO1Model(options.model) || this.isO3Model(options.model)) {
      // a) use max_completion_tokens instead of max_tokens
      finalOptions.max_completion_tokens = options.maxTokens?.toString();
      finalOptions.max_tokens = undefined;

      // b) don't support system message
      let newMessages = formatMessageForO1(finalOptions.messages);
      newMessages = newMessages.map((msg: ChatMessage) => {
        if (msg.role === "user") {
          msg.content =
            "Always use markdown formatting. \nInclude descriptions/explanations for your code.\nOnly send the complete code if requested, otherwise, only send the modified/new sections of code.\n" + msg.content;
        }
        return msg;
      });
      finalOptions.messages = newMessages;
    }

    // OpenAI gpt4:
    if (this.isGPTModel(options.model)) {
      // a) use max_completion_tokens instead of max_tokens
      finalOptions.max_completion_tokens = undefined;
      finalOptions.max_tokens = options.maxTokens?.toString();

      // b) don't support system message
      finalOptions.messages = formatMessageForO1(finalOptions.messages);
    }

    if (options.prediction && this.supportsPrediction(options.model)) {
      if (finalOptions.presence_penalty) {
        // prediction doesn't support > 0
        finalOptions.presence_penalty = undefined;
      }
      if (finalOptions.frequency_penalty) {
        // prediction doesn't support > 0
        finalOptions.frequency_penalty = undefined;
      }
      finalOptions.max_completion_tokens = undefined;

      finalOptions.prediction = options.prediction;
    } else {
      finalOptions.prediction = undefined;
    }

    return finalOptions;
  }

  protected _getHeaders() {
    return {
      "Content-Type": "application/json",
      Authorization: `Bearer ${this.apiKey}`,
      "X-Auth-Token": this.apiKey ?? "", // For Azure
    };
  }

  protected async _complete(
    prompt: string,
    signal: AbortSignal,
    options: CompletionOptions,
  ): Promise<string> {
    let completion = "";
    if (this.isO1Model(options.model) || this.isO3Model(options.model)) {
      prompt = "Always use markdown formatting. \nInclude descriptions/explanations for your code.\nOnly send the complete code if requested, otherwise, only send the modified/new sections of code.\n" + prompt;
    }
    for await (const chunk of this._streamChat(
      [{ role: "user", content: prompt }],
      signal,
      options,
    )) {
      completion += chunk.content;
    }

    return completion;
  }

  protected _getEndpoint(
    endpoint: "chat/completions" | "completions" | "models",
  ) {
    if (this.apiType === "azure") {
      return new URL(
        `openai/deployments/${this.deployment}/${endpoint}?api-version=${this.apiVersion}`,
        this.apiBase,
      );
    }
    if (!this.apiBase) {
      throw new Error(
        "No API base URL provided. Please set the 'apiBase' option in config.json",
      );
    }

    return new URL(endpoint, this.apiBase);
  }

  protected async *_streamComplete(
    prompt: string,
    signal: AbortSignal,
    options: CompletionOptions,
  ): AsyncGenerator<string> {
    for await (const chunk of this._streamChat(
      [{ role: "user", content: prompt }],
      signal,
      options,
    )) {
      yield renderChatMessage(chunk);
    }
  }

  protected modifyChatBody(
    body: ChatCompletionCreateParams,
  ): ChatCompletionCreateParams {
    // Приводим body к локальному типу для выполнения преобразований
    const f5Body = body as F5AIChatCompletionCreateParams;

    // Обрезаем stop-слова
    f5Body.stop = f5Body.stop?.slice(0, this.getMaxStopWords());

    // Для моделей o1-preview и o1-mini
    if (this.isO1Model(f5Body.model) || this.isO3Model(f5Body.model)) {
      // Преобразуем max_tokens (число) в строку и присваиваем в max_completion_tokens
      f5Body.max_completion_tokens =
        f5Body.max_tokens !== undefined ? f5Body.max_tokens.toString() : undefined;
      f5Body.max_tokens = undefined;
      f5Body.messages = formatMessageForO1(f5Body.messages);
    }

    if (this.isGPTModel(f5Body.model)) {
      // Преобразуем max_tokens (число) в строку и присваиваем в max_completion_tokens
      f5Body.max_completion_tokens = undefined;
      f5Body.max_tokens =
        f5Body.max_tokens !== undefined ? f5Body.max_tokens.toString() : undefined;
      f5Body.messages = formatMessageForO1(f5Body.messages);
    }

    if (f5Body.prediction && this.supportsPrediction(f5Body.model)) {
      if (f5Body.presence_penalty) {
        f5Body.presence_penalty = undefined;
      }
      if (f5Body.frequency_penalty) {
        f5Body.frequency_penalty = undefined;
      }
      f5Body.max_completion_tokens = undefined;
    }

    if (f5Body.tools?.length) {
      // Для соблюдения схемы (например, для параллельного вызова функций)
      f5Body.parallel_tool_calls = false;
    }

    // Приводим обратно к типу ChatCompletionCreateParams.
    // Если прямое приведение не проходит, можно воспользоваться двойным приведением:
    return f5Body as unknown as ChatCompletionCreateParams;
  }

  protected async *_legacystreamComplete(
    prompt: string,
    signal: AbortSignal,
    options: CompletionOptions,
  ): AsyncGenerator<string> {
    const args: any = this._convertArgs(options, []);
    args.prompt = prompt;
    args.messages = undefined;

    const response = await this.fetch(this._getEndpoint("completions"), {
      method: "POST",
      headers: this._getHeaders(),
      body: JSON.stringify({
        ...args,
        stream: false,
      }),
      signal,
    });

    for await (const value of streamSse(response)) {
      if (value.choices?.[0]?.text && value.finish_reason !== "eos") {
        yield value.choices[0].text;
      }
    }
  }

  protected async *_streamChat(
    messages: ChatMessage[],
    signal: AbortSignal,
    options: CompletionOptions,
  ): AsyncGenerator<ChatMessage> {
    if (
      !CHAT_ONLY_MODELS.includes(options.model) &&
      this.supportsCompletions() &&
      (NON_CHAT_MODELS.includes(options.model) ||
        this.useLegacyCompletionsEndpoint ||
        options.raw)
    ) {
      for await (const content of this._legacystreamComplete(
        renderChatMessage(messages[messages.length - 1]),
        signal,
        options,
      )) {
        yield {
          role: "assistant",
          content,
        };
      }
      return;
    }

    const body = this._convertArgs(options, messages);
    body.stream = false; // Отключаем стриминг

    const response = await this.fetch(this._getEndpoint("chat/completions"), {
      method: "POST",
      headers: this._getHeaders(),
      body: JSON.stringify(body),
      signal,
    });

    // Обрабатываем как нестриминговый ответ
    const data = await response.json();
    yield data.choices[0].message;
  }

  protected async *_streamFim(
    prefix: string,
    suffix: string,
    signal: AbortSignal,
    options: CompletionOptions,
  ): AsyncGenerator<string> {
    const endpoint = new URL("fim/completions", this.apiBase);
    const resp = await this.fetch(endpoint, {
      method: "POST",
      body: JSON.stringify({
        model: options.model,
        prompt: prefix,
        suffix,
        max_tokens: options.maxTokens?.toString,
        temperature: options.temperature,
        top_p: options.topP,
        frequency_penalty: options.frequencyPenalty,
        presence_penalty: options.presencePenalty,
        stop: options.stop,
        stream: false,
      }),
      headers: {
        "Content-Type": "application/json",

        Accept: "application/json",
        "X-Auth-Token": this.apiKey ?? "",
        Authorization: `Bearer ${this.apiKey}`,
      },
      signal,
    });
    for await (const chunk of streamSse(resp)) {
      yield chunk.choices[0].delta.content;
    }
  }

  async listModels(): Promise<string[]> {
    const response = await this.fetch(this._getEndpoint("models"), {
      method: "GET",
      headers: this._getHeaders(),
    });

    const data = await response.json();
    return data.data.map((m: any) => m.id);
  }

  private _getEmbedEndpoint() {
    if (!this.apiBase) {
      throw new Error(
        "No API base URL provided. Please set the 'apiBase' option in config.json",
      );
    }

    if (this.apiType === "azure") {
      return new URL(
        `openai/deployments/${this.deployment}/embeddings?api-version=${this.apiVersion}`,
        this.apiBase,
      );
    }
    return new URL("embeddings", this.apiBase);
  }

  protected async _embed(chunks: string[]): Promise<number[][]> {
    const resp = await this.fetch(this._getEmbedEndpoint(), {
      method: "POST",
      body: JSON.stringify({
        input: chunks,
        model: this.model,
      }),
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
        "X-Auth-Token": this.apiKey ?? "", // For Azure
      },
    });

    if (!resp.ok) {
      throw new Error(await resp.text());
    }

    const data = (await resp.json()) as any;
    return data.data.map((result: { embedding: number[] }) => result.embedding);
  }
}

export default F5AI;
