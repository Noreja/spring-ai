/*
 * Copyright 2025-present the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.springframework.ai.openai;

import java.util.List;
import java.util.Optional;

import com.openai.client.OpenAIClient;
import com.openai.client.OpenAIClientAsync;
import com.openai.core.JsonValue;
import com.openai.models.chat.completions.ChatCompletion;
import com.openai.models.chat.completions.ChatCompletionCreateParams;
import com.openai.models.chat.completions.ChatCompletionMessage;
import com.openai.models.completions.CompletionUsage;
import com.openai.services.blocking.ChatService;
import com.openai.services.blocking.chat.ChatCompletionService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.Prompt;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Unit tests for the noreja-fork addition that emits {@code reasoning_content} as its own
 * {@link Generation} (with {@code properties["reasoning"] = TRUE}) preceding the
 * assistant-text Generation. Mirrors the Anthropic thinking-block pattern so
 * cross-provider consumers can classify generations via a single property key.
 *
 * @author Noreja
 */
@ExtendWith(MockitoExtension.class)
class OpenAiChatModelReasoningTests {

	@Mock
	OpenAIClient openAiClient;

	@Mock
	OpenAIClientAsync openAiClientAsync;

	private void stubCompletion(ChatCompletionMessage message) {
		ChatService chatService = mock(ChatService.class);
		ChatCompletionService chatCompletionService = mock(ChatCompletionService.class);
		when(this.openAiClient.chat()).thenReturn(chatService);
		when(chatService.completions()).thenReturn(chatCompletionService);
		when(chatCompletionService.create(any(ChatCompletionCreateParams.class))).thenReturn(ChatCompletion.builder()
			.id("gen-reasoning-1")
			.created(1777799928)
			.model("deepseek-reasoner")
			.usage(CompletionUsage.builder().promptTokens(1).completionTokens(1).totalTokens(2).build())
			.addChoice(ChatCompletion.Choice.builder()
				.finishReason(ChatCompletion.Choice.FinishReason.STOP)
				.index(0)
				.logprobs(Optional.empty())
				.message(message)
				.build())
			.build());
	}

	private OpenAiChatModel buildModel() {
		OpenAiChatOptions options = OpenAiChatOptions.builder().model("deepseek-reasoner").build();
		return OpenAiChatModel.builder()
			.openAiClient(this.openAiClient)
			.openAiClientAsync(this.openAiClientAsync)
			.options(options)
			.build();
	}

	@Test
	void emitsReasoningContentAsSeparateGeneration() {
		stubCompletion(ChatCompletionMessage.builder()
			.content("The answer is 42.")
			.refusal(Optional.empty())
			.role(JsonValue.from("assistant"))
			.annotations(List.of())
			.toolCalls(List.of())
			.putAdditionalProperty("reasoning_content", JsonValue.from("Let me think step by step."))
			.build());

		ChatResponse response = buildModel().call(new Prompt("hi"));

		List<Generation> generations = response.getResults();
		assertThat(generations).hasSize(2);

		// First generation is the reasoning, tagged with reasoning=true.
		Generation reasoning = generations.get(0);
		assertThat(reasoning.getOutput().getText()).isEqualTo("Let me think step by step.");
		assertThat(reasoning.getOutput().getMetadata()).containsEntry("reasoning", Boolean.TRUE);

		// Second generation is the assistant text, NOT tagged as reasoning.
		Generation text = generations.get(1);
		assertThat(text.getOutput().getText()).isEqualTo("The answer is 42.");
		assertThat(text.getOutput().getMetadata().get("reasoning")).isNull();
	}

	@Test
	void singleGenerationWhenNoReasoningContent() {
		stubCompletion(ChatCompletionMessage.builder()
			.content("The answer is 42.")
			.refusal(Optional.empty())
			.role(JsonValue.from("assistant"))
			.annotations(List.of())
			.toolCalls(List.of())
			.build());

		ChatResponse response = buildModel().call(new Prompt("hi"));

		assertThat(response.getResults()).hasSize(1);
		assertThat(response.getResult().getOutput().getText()).isEqualTo("The answer is 42.");
		assertThat(response.getResult().getOutput().getMetadata().get("reasoning")).isNull();
	}

}
