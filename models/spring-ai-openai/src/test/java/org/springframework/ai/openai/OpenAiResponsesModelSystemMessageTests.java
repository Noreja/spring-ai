/*
 * Copyright 2023-present the original author or authors.
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

import com.openai.models.responses.ResponseCreateParams;
import org.junit.jupiter.api.Test;

import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * The Responses API takes system text only through its single instructions field.
 */
class OpenAiResponsesModelSystemMessageTests {

	private final OpenAiResponsesModel model = OpenAiResponsesModel.builder()
		.defaultOptions(OpenAiResponsesOptions.builder().model("gpt-test").apiKey("test-key").build())
		.build();

	private final OpenAiResponsesOptions options = OpenAiResponsesOptions.builder().model("gpt-test").build();

	@Test
	void everySystemMessageReachesTheInstructionsInOrder() {
		ResponseCreateParams request = this.model.createRequest(new Prompt(List.of(new SystemMessage("system prompt"),
				new UserMessage("question"), new SystemMessage("summary of earlier turns")), this.options));

		assertThat(request.instructions()).contains("system prompt\n\nsummary of earlier turns");
	}

	@Test
	void aSingleSystemMessageIsTheInstructions() {
		ResponseCreateParams request = this.model.createRequest(
				new Prompt(List.of(new SystemMessage("system prompt"), new UserMessage("question")), this.options));

		assertThat(request.instructions()).contains("system prompt");
	}

}
