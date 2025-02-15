import type {
  ChatCompletionCreateParams as OriginalChatCompletionCreateParams,
} from "openai/resources/index";

// Создаём новый тип для F5AI, где поля max_completion_tokens и max_tokens – строки
type F5AIChatCompletionCreateParams = Omit<
  OriginalChatCompletionCreateParams,
  "max_completion_tokens" | "max_tokens"
> & {
  max_completion_tokens?: string;
  max_tokens?: string;
};
