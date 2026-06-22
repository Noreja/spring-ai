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

package org.springframework.ai.model.openai.autoconfigure;

import java.util.List;

import com.openai.client.OpenAIClient;
import com.openai.client.OpenAIClientAsync;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.observation.ObservationRegistry;
import org.jspecify.annotations.Nullable;

import org.springframework.ai.chat.observation.ChatModelObservationConvention;
import org.springframework.ai.model.tool.ToolCallingManager;
import org.springframework.ai.model.tool.autoconfigure.ToolCallingAutoConfiguration;
import org.springframework.ai.openai.OpenAiResponsesModel;
import org.springframework.ai.openai.setup.OpenAiSetup;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;

/**
 * {@link AutoConfiguration Auto-configuration} for the OpenAI Responses API model, backed
 * by the official OpenAI Java SDK. Disabled by default; enable with
 * {@code spring.ai.openai.responses.enabled=true}.
 *
 * @author Noreja
 */
@AutoConfiguration(after = { ToolCallingAutoConfiguration.class })
@EnableConfigurationProperties({ OpenAiCommonProperties.class, OpenAiResponsesProperties.class })
@ConditionalOnProperty(name = "spring.ai.openai.responses.enabled", havingValue = "true")
public class OpenAiResponsesAutoConfiguration {

	@Bean
	public OpenAiResponsesModel openAiResponsesModel(OpenAiCommonProperties commonProperties,
			OpenAiResponsesProperties responsesProperties, ToolCallingManager toolCallingManager,
			ObjectProvider<ObservationRegistry> observationRegistry, ObjectProvider<MeterRegistry> meterRegistry,
			ObjectProvider<ChatModelObservationConvention> observationConvention) {

		var resolvedProperties = OpenAiAutoConfigurationUtil.resolveCommonProperties(commonProperties,
				responsesProperties);

		MeterRegistry meterRegistryToUse = resolvedProperties.isConnectionPoolMetricsEnabled()
				? meterRegistry.getIfAvailable() : null;

		OpenAIClient openAIClient = this.openAiClient(resolvedProperties, observationRegistry, meterRegistryToUse);
		OpenAIClientAsync openAIClientAsync = this.openAiClientAsync(resolvedProperties, observationRegistry,
				meterRegistryToUse);

		var responsesModel = new OpenAiResponsesModel(openAIClient, openAIClientAsync, responsesProperties.toOptions(),
				toolCallingManager, observationRegistry.getIfUnique(() -> ObservationRegistry.NOOP));

		observationConvention.ifAvailable(responsesModel::setObservationConvention);

		return responsesModel;
	}

	private OpenAIClient openAiClient(OpenAiCommonProperties resolved,
			ObjectProvider<ObservationRegistry> observationRegistry, @Nullable MeterRegistry meterRegistry) {

		return OpenAiSetup.setupSyncClient(resolved.getBaseUrl(), resolved.getApiKey(), resolved.getCredential(),
				resolved.getMicrosoftDeploymentName(), resolved.getMicrosoftFoundryServiceVersion(),
				resolved.getOrganizationId(), resolved.isMicrosoftFoundry(), resolved.isGitHubModels(),
				resolved.getModel(), resolved.getTimeout(), resolved.getMaxRetries(), resolved.getProxy(),
				resolved.getCustomHeaders(), observationRegistry.getIfUnique(() -> ObservationRegistry.NOOP),
				meterRegistry, List.of());
	}

	private OpenAIClientAsync openAiClientAsync(OpenAiCommonProperties resolved,
			ObjectProvider<ObservationRegistry> observationRegistry, @Nullable MeterRegistry meterRegistry) {

		return OpenAiSetup.setupAsyncClient(resolved.getBaseUrl(), resolved.getApiKey(), resolved.getCredential(),
				resolved.getMicrosoftDeploymentName(), resolved.getMicrosoftFoundryServiceVersion(),
				resolved.getOrganizationId(), resolved.isMicrosoftFoundry(), resolved.isGitHubModels(),
				resolved.getModel(), resolved.getTimeout(), resolved.getMaxRetries(), resolved.getProxy(),
				resolved.getCustomHeaders(), observationRegistry.getIfUnique(() -> ObservationRegistry.NOOP),
				meterRegistry, List.of());
	}

}
