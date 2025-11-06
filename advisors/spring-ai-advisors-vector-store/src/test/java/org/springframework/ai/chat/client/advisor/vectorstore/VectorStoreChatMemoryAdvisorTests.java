/*
 * Copyright 2025-2025 the original author or authors.
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

package org.springframework.ai.chat.client.advisor.vectorstore;

import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.Mockito;
import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.advisor.api.AdvisorChain;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.VectorStore;

import java.util.Collection;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.Mockito.atLeastOnce;
import static org.mockito.Mockito.verify;

/**
 * Unit tests for {@link VectorStoreChatMemoryAdvisor}.
 *
 * @author Thomas Vitale
 */
class VectorStoreChatMemoryAdvisorTests {

	@Test
	void whenVectorStoreIsNullThenThrow() {
		assertThatThrownBy(() -> VectorStoreChatMemoryAdvisor.builder(null).build())
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("vectorStore cannot be null");
	}

	@Test
	void whenDefaultConversationIdIsNullThenThrow() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);

		assertThatThrownBy(() -> VectorStoreChatMemoryAdvisor.builder(vectorStore).conversationId(null).build())
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("defaultConversationId cannot be null or empty");
	}

	@Test
	void whenDefaultConversationIdIsEmptyThenThrow() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);

		assertThatThrownBy(() -> VectorStoreChatMemoryAdvisor.builder(vectorStore).conversationId(null).build())
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("defaultConversationId cannot be null or empty");
	}

	@Test
	void whenSchedulerIsNullThenThrow() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);

		assertThatThrownBy(() -> VectorStoreChatMemoryAdvisor.builder(vectorStore).scheduler(null).build())
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("scheduler cannot be null");
	}

	@Test
	void whenSystemPromptTemplateIsNullThenThrow() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);

		assertThatThrownBy(() -> VectorStoreChatMemoryAdvisor.builder(vectorStore).systemPromptTemplate(null).build())
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("systemPromptTemplate cannot be null");
	}

	@Test
	void whenDefaultTopKIsZeroThenThrow() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);

		assertThatThrownBy(() -> VectorStoreChatMemoryAdvisor.builder(vectorStore).defaultTopK(0).build())
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("topK must be greater than 0");
	}

	@Test
	void whenDefaultTopKIsNegativeThenThrow() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);

		assertThatThrownBy(() -> VectorStoreChatMemoryAdvisor.builder(vectorStore).defaultTopK(-1).build())
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("topK must be greater than 0");
	}

	@Test
	void whenCustomFilterExpressionIsNullThenDoNotThrow() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);
		assertThatNoException()
			.isThrownBy(() -> VectorStoreChatMemoryAdvisor.builder(vectorStore).customFilterExpression(null).build());
	}

	@Test
	void whenCustomMetaDataIsNullThenThrow() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);

		assertThatThrownBy(() -> VectorStoreChatMemoryAdvisor.builder(vectorStore).customMetaData(null).build())
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("customMetaData cannot be null");
	}

	@Test
	void whenCustomMetaDataIsAppliedThenDocumentContainsCustomMetaData() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);
		var key = "CustomKey";
		var value = "CustomValue";
		var advisor = VectorStoreChatMemoryAdvisor.builder(vectorStore).customMetaData(Map.of(key, value)).build();

		var advisorChain = Mockito.mock(AdvisorChain.class);
		var request = ChatClientRequest.builder().prompt(Prompt.builder().content("Some content").build()).build();

		advisor.before(request, advisorChain);

		// then: capture every batch of Documents written and assert metadata
		@SuppressWarnings("unchecked")
		ArgumentCaptor<List<Document>> docsCaptor = ArgumentCaptor.forClass(List.class);

		verify(vectorStore, atLeastOnce()).write(docsCaptor.capture());

		List<Document> allDocs = docsCaptor.getAllValues().stream().flatMap(Collection::stream).toList();

		assertThat(allDocs).as("VectorStore.write should receive at least one Document").isNotEmpty();

		assertThat(allDocs).allSatisfy(doc -> {
			Map<String, Object> meta = doc.getMetadata();
			assertThat(meta).as("Document metadata should contain the custom key").containsEntry(key, value);
		});
	}

}
