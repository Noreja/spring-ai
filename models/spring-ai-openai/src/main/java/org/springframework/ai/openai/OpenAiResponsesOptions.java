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

import java.net.Proxy;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import com.openai.azure.AzureOpenAIServiceVersion;
import com.openai.credential.Credential;
import org.jspecify.annotations.Nullable;

import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.model.tool.DefaultToolCallingChatOptions;
import org.springframework.ai.model.tool.ToolCallingChatOptions;
import org.springframework.ai.tool.ToolCallback;

/**
 * Configuration options for the OpenAI Responses API model implementation, backed by the
 * official OpenAI Java SDK. The Responses API exposes built-in tools like
 * {@code web_search}, {@code file_search}, and {@code code_interpreter} which are only
 * available through the Responses endpoint.
 *
 * <p>
 * Mirrors the immutable, builder-based design of {@link OpenAiChatOptions}: the builder
 * extends {@link DefaultToolCallingChatOptions.Builder} so tool-calling and base
 * {@link ChatOptions} plumbing ({@code model}, {@code temperature},
 * {@code toolCallbacks}, {@code mutate()}, {@code combineWith}, …) is inherited.
 *
 * @author Noreja
 */
public class OpenAiResponsesOptions implements ToolCallingChatOptions {

	public static final String DEFAULT_MODEL = "gpt-4o";

	// Connection options (mirrors AbstractOpenAiOptions, which is immutable and cannot be
	// extended mutably here).
	private @Nullable String baseUrl;

	private @Nullable String apiKey;

	private @Nullable Credential credential;

	private @Nullable String model;

	private @Nullable String microsoftDeploymentName;

	private @Nullable AzureOpenAIServiceVersion microsoftFoundryServiceVersion;

	private @Nullable String organizationId;

	private boolean isMicrosoftFoundry;

	private boolean isGitHubModels;

	private Duration timeout = AbstractOpenAiOptions.DEFAULT_TIMEOUT;

	private Integer maxRetries = AbstractOpenAiOptions.DEFAULT_MAX_RETRIES;

	private @Nullable Proxy proxy;

	private @Nullable Map<String, String> customHeaders;

	// Base ChatOptions / ToolCallingChatOptions
	private @Nullable Double temperature;

	private @Nullable Double topP;

	private @Nullable List<String> stop;

	private @Nullable List<ToolCallback> toolCallbacks;

	private @Nullable Map<String, Object> toolContext;

	// Responses API specific
	private @Nullable Integer maxCompletionTokens;

	private @Nullable String reasoningEffort;

	private @Nullable String reasoningSummary;

	private @Nullable Boolean parallelToolCalls;

	private @Nullable Boolean strictTools;

	private @Nullable String user;

	private @Nullable String serviceTier;

	private @Nullable Map<String, String> metadata;

	private @Nullable Boolean store;

	private @Nullable List<BuiltInTool> builtInTools;

	private @Nullable WebSearchOptions webSearchOptions;

	private @Nullable String previousResponseId;

	private @Nullable String instructions;

	private @Nullable String truncation;

	private @Nullable Object toolChoice;

	private @Nullable Boolean internalToolExecutionEnabled;

	OpenAiResponsesOptions() {
	}

	public @Nullable String getBaseUrl() {
		return this.baseUrl;
	}

	public @Nullable String getApiKey() {
		return this.apiKey;
	}

	public @Nullable Credential getCredential() {
		return this.credential;
	}

	@Override
	public @Nullable String getModel() {
		return this.model;
	}

	public @Nullable String getMicrosoftDeploymentName() {
		return this.microsoftDeploymentName;
	}

	/**
	 * Alias for {@link #getMicrosoftDeploymentName()}.
	 */
	public @Nullable String getDeploymentName() {
		return this.microsoftDeploymentName;
	}

	public @Nullable AzureOpenAIServiceVersion getMicrosoftFoundryServiceVersion() {
		return this.microsoftFoundryServiceVersion;
	}

	public @Nullable String getOrganizationId() {
		return this.organizationId;
	}

	public boolean isMicrosoftFoundry() {
		return this.isMicrosoftFoundry;
	}

	public boolean isGitHubModels() {
		return this.isGitHubModels;
	}

	public Duration getTimeout() {
		return this.timeout;
	}

	public Integer getMaxRetries() {
		return this.maxRetries;
	}

	public @Nullable Proxy getProxy() {
		return this.proxy;
	}

	public @Nullable Map<String, String> getCustomHeaders() {
		return this.customHeaders;
	}

	@Override
	public @Nullable Double getTemperature() {
		return this.temperature;
	}

	@Override
	public @Nullable Double getTopP() {
		return this.topP;
	}

	public @Nullable List<String> getStop() {
		return this.stop;
	}

	@Override
	public @Nullable List<String> getStopSequences() {
		return this.stop;
	}

	public @Nullable Integer getMaxCompletionTokens() {
		return this.maxCompletionTokens;
	}

	@Override
	public @Nullable Integer getMaxTokens() {
		return this.maxCompletionTokens;
	}

	public @Nullable String getReasoningEffort() {
		return this.reasoningEffort;
	}

	/**
	 * Reasoning summary visibility for the Responses API ({@code "auto"},
	 * {@code "concise"}, or {@code "detailed"}). When set, the API streams
	 * reasoning-summary deltas, which are the only reasoning content GPT-5.x exposes to
	 * clients. When {@code null}, no summary is emitted.
	 */
	public @Nullable String getReasoningSummary() {
		return this.reasoningSummary;
	}

	public @Nullable Boolean getParallelToolCalls() {
		return this.parallelToolCalls;
	}

	/**
	 * Whether OpenAI's strict-mode JSON schema validation is requested for function tool
	 * definitions. When {@code true}, every property in the schema must appear in the
	 * {@code required} array; this conflicts with {@code @ToolParam(required = false)} on
	 * MCP tools that legitimately expose optional parameters. When {@code null}, the
	 * implementation falls back to its built-in default ({@code true}).
	 */
	public @Nullable Boolean getStrictTools() {
		return this.strictTools;
	}

	public @Nullable String getUser() {
		return this.user;
	}

	public @Nullable String getServiceTier() {
		return this.serviceTier;
	}

	public @Nullable Map<String, String> getMetadata() {
		return this.metadata;
	}

	public @Nullable Boolean getStore() {
		return this.store;
	}

	public @Nullable List<BuiltInTool> getBuiltInTools() {
		return this.builtInTools;
	}

	public @Nullable WebSearchOptions getWebSearchOptions() {
		return this.webSearchOptions;
	}

	public @Nullable String getPreviousResponseId() {
		return this.previousResponseId;
	}

	public @Nullable String getInstructions() {
		return this.instructions;
	}

	public @Nullable String getTruncation() {
		return this.truncation;
	}

	public @Nullable Object getToolChoice() {
		return this.toolChoice;
	}

	public @Nullable Boolean getInternalToolExecutionEnabled() {
		return this.internalToolExecutionEnabled;
	}

	@Override
	public @Nullable List<ToolCallback> getToolCallbacks() {
		return this.toolCallbacks;
	}

	@Override
	public @Nullable Map<String, Object> getToolContext() {
		return this.toolContext;
	}

	@Override
	public @Nullable Integer getTopK() {
		return null;
	}

	@Override
	public @Nullable Double getFrequencyPenalty() {
		return null;
	}

	@Override
	public @Nullable Double getPresencePenalty() {
		return null;
	}

	public static Builder builder() {
		return new Builder();
	}

	@Override
	public Builder mutate() {
		return builder()
			// connection
			.baseUrl(this.baseUrl)
			.apiKey(this.apiKey)
			.credential(this.credential)
			.model(this.model)
			.deploymentName(this.microsoftDeploymentName)
			.microsoftFoundryServiceVersion(this.microsoftFoundryServiceVersion)
			.organizationId(this.organizationId)
			.microsoftFoundry(this.isMicrosoftFoundry)
			.gitHubModels(this.isGitHubModels)
			.timeout(this.timeout)
			.maxRetries(this.maxRetries)
			.proxy(this.proxy)
			.customHeaders(this.customHeaders)
			// base ChatOptions / ToolCallingChatOptions
			.temperature(this.temperature)
			.topP(this.topP)
			.stopSequences(this.stop)
			.toolCallbacks(this.toolCallbacks)
			.toolContext(this.toolContext)
			// Responses specific
			.maxCompletionTokens(this.maxCompletionTokens)
			.reasoningEffort(this.reasoningEffort)
			.reasoningSummary(this.reasoningSummary)
			.parallelToolCalls(this.parallelToolCalls)
			.strictTools(this.strictTools)
			.user(this.user)
			.serviceTier(this.serviceTier)
			.metadata(this.metadata)
			.store(this.store)
			.builtInTools(this.builtInTools)
			.webSearchOptions(this.webSearchOptions)
			.previousResponseId(this.previousResponseId)
			.instructions(this.instructions)
			.truncation(this.truncation)
			.toolChoice(this.toolChoice)
			.internalToolExecutionEnabled(this.internalToolExecutionEnabled);
	}

	@Override
	public boolean equals(@Nullable Object o) {
		if (this == o) {
			return true;
		}
		if (o == null || getClass() != o.getClass()) {
			return false;
		}
		OpenAiResponsesOptions that = (OpenAiResponsesOptions) o;
		return this.isMicrosoftFoundry == that.isMicrosoftFoundry && this.isGitHubModels == that.isGitHubModels
				&& Objects.equals(this.baseUrl, that.baseUrl) && Objects.equals(this.apiKey, that.apiKey)
				&& Objects.equals(this.credential, that.credential) && Objects.equals(this.model, that.model)
				&& Objects.equals(this.microsoftDeploymentName, that.microsoftDeploymentName)
				&& Objects.equals(this.microsoftFoundryServiceVersion, that.microsoftFoundryServiceVersion)
				&& Objects.equals(this.organizationId, that.organizationId)
				&& Objects.equals(this.timeout, that.timeout) && Objects.equals(this.maxRetries, that.maxRetries)
				&& Objects.equals(this.proxy, that.proxy) && Objects.equals(this.customHeaders, that.customHeaders)
				&& Objects.equals(this.temperature, that.temperature) && Objects.equals(this.topP, that.topP)
				&& Objects.equals(this.stop, that.stop) && Objects.equals(this.toolCallbacks, that.toolCallbacks)
				&& Objects.equals(this.toolContext, that.toolContext)
				&& Objects.equals(this.maxCompletionTokens, that.maxCompletionTokens)
				&& Objects.equals(this.reasoningEffort, that.reasoningEffort)
				&& Objects.equals(this.reasoningSummary, that.reasoningSummary)
				&& Objects.equals(this.parallelToolCalls, that.parallelToolCalls)
				&& Objects.equals(this.strictTools, that.strictTools) && Objects.equals(this.user, that.user)
				&& Objects.equals(this.serviceTier, that.serviceTier) && Objects.equals(this.metadata, that.metadata)
				&& Objects.equals(this.store, that.store) && Objects.equals(this.builtInTools, that.builtInTools)
				&& Objects.equals(this.webSearchOptions, that.webSearchOptions)
				&& Objects.equals(this.previousResponseId, that.previousResponseId)
				&& Objects.equals(this.instructions, that.instructions)
				&& Objects.equals(this.truncation, that.truncation) && Objects.equals(this.toolChoice, that.toolChoice)
				&& Objects.equals(this.internalToolExecutionEnabled, that.internalToolExecutionEnabled);
	}

	@Override
	public int hashCode() {
		return Objects.hash(this.baseUrl, this.apiKey, this.credential, this.model, this.microsoftDeploymentName,
				this.microsoftFoundryServiceVersion, this.organizationId, this.isMicrosoftFoundry, this.isGitHubModels,
				this.timeout, this.maxRetries, this.proxy, this.customHeaders, this.temperature, this.topP, this.stop,
				this.toolCallbacks, this.toolContext, this.maxCompletionTokens, this.reasoningEffort,
				this.reasoningSummary, this.parallelToolCalls, this.strictTools, this.user, this.serviceTier,
				this.metadata, this.store, this.builtInTools, this.webSearchOptions, this.previousResponseId,
				this.instructions, this.truncation, this.toolChoice, this.internalToolExecutionEnabled);
	}

	// The public Builder. Hides the noisy self-referential generics.
	public static class Builder extends AbstractBuilder<Builder> {

	}

	protected abstract static class AbstractBuilder<B extends AbstractBuilder<B>>
			extends DefaultToolCallingChatOptions.Builder<B> {

		protected @Nullable String baseUrl;

		protected @Nullable String apiKey;

		protected @Nullable Credential credential;

		protected @Nullable String microsoftDeploymentName;

		protected @Nullable AzureOpenAIServiceVersion microsoftFoundryServiceVersion;

		protected @Nullable String organizationId;

		protected @Nullable Boolean isMicrosoftFoundry;

		protected @Nullable Boolean isGitHubModels;

		protected @Nullable Duration timeout;

		protected @Nullable Integer maxRetries;

		protected @Nullable Proxy proxy;

		protected @Nullable Map<String, String> customHeaders;

		protected @Nullable Integer maxCompletionTokens;

		protected @Nullable String reasoningEffort;

		protected @Nullable String reasoningSummary;

		protected @Nullable Boolean parallelToolCalls;

		protected @Nullable Boolean strictTools;

		protected @Nullable String user;

		protected @Nullable String serviceTier;

		protected @Nullable Map<String, String> metadata;

		protected @Nullable Boolean store;

		protected @Nullable List<BuiltInTool> builtInTools;

		protected @Nullable WebSearchOptions webSearchOptions;

		protected @Nullable String previousResponseId;

		protected @Nullable String instructions;

		protected @Nullable String truncation;

		protected @Nullable Object toolChoice;

		protected @Nullable Boolean internalToolExecutionEnabled;

		public B baseUrl(@Nullable String baseUrl) {
			this.baseUrl = baseUrl;
			return self();
		}

		public B apiKey(@Nullable String apiKey) {
			this.apiKey = apiKey;
			return self();
		}

		public B credential(@Nullable Credential credential) {
			this.credential = credential;
			return self();
		}

		public B deploymentName(@Nullable String deploymentName) {
			this.microsoftDeploymentName = deploymentName;
			return self();
		}

		public B microsoftFoundryServiceVersion(@Nullable AzureOpenAIServiceVersion microsoftFoundryServiceVersion) {
			this.microsoftFoundryServiceVersion = microsoftFoundryServiceVersion;
			return self();
		}

		public B organizationId(@Nullable String organizationId) {
			this.organizationId = organizationId;
			return self();
		}

		public B microsoftFoundry(@Nullable Boolean microsoftFoundry) {
			this.isMicrosoftFoundry = microsoftFoundry;
			return self();
		}

		public B gitHubModels(@Nullable Boolean gitHubModels) {
			this.isGitHubModels = gitHubModels;
			return self();
		}

		public B timeout(@Nullable Duration timeout) {
			this.timeout = timeout;
			return self();
		}

		public B maxRetries(@Nullable Integer maxRetries) {
			this.maxRetries = maxRetries;
			return self();
		}

		public B proxy(@Nullable Proxy proxy) {
			this.proxy = proxy;
			return self();
		}

		public B customHeaders(@Nullable Map<String, String> customHeaders) {
			this.customHeaders = customHeaders;
			return self();
		}

		public B maxCompletionTokens(@Nullable Integer maxCompletionTokens) {
			this.maxCompletionTokens = maxCompletionTokens;
			return self();
		}

		public B stop(@Nullable List<String> stop) {
			return this.stopSequences(stop);
		}

		public B reasoningEffort(@Nullable String reasoningEffort) {
			this.reasoningEffort = reasoningEffort;
			return self();
		}

		public B reasoningSummary(@Nullable String reasoningSummary) {
			this.reasoningSummary = reasoningSummary;
			return self();
		}

		public B parallelToolCalls(@Nullable Boolean parallelToolCalls) {
			this.parallelToolCalls = parallelToolCalls;
			return self();
		}

		public B strictTools(@Nullable Boolean strictTools) {
			this.strictTools = strictTools;
			return self();
		}

		public B user(@Nullable String user) {
			this.user = user;
			return self();
		}

		public B serviceTier(@Nullable String serviceTier) {
			this.serviceTier = serviceTier;
			return self();
		}

		public B metadata(@Nullable Map<String, String> metadata) {
			this.metadata = metadata;
			return self();
		}

		public B store(@Nullable Boolean store) {
			this.store = store;
			return self();
		}

		public B builtInTools(@Nullable List<BuiltInTool> builtInTools) {
			this.builtInTools = builtInTools;
			return self();
		}

		public B builtInTools(BuiltInTool... builtInTools) {
			this.builtInTools = Arrays.asList(builtInTools);
			return self();
		}

		public B webSearchOptions(@Nullable WebSearchOptions webSearchOptions) {
			this.webSearchOptions = webSearchOptions;
			return self();
		}

		public B previousResponseId(@Nullable String previousResponseId) {
			this.previousResponseId = previousResponseId;
			return self();
		}

		public B instructions(@Nullable String instructions) {
			this.instructions = instructions;
			return self();
		}

		public B truncation(@Nullable String truncation) {
			this.truncation = truncation;
			return self();
		}

		public B toolChoice(@Nullable Object toolChoice) {
			this.toolChoice = toolChoice;
			return self();
		}

		public B internalToolExecutionEnabled(@Nullable Boolean internalToolExecutionEnabled) {
			this.internalToolExecutionEnabled = internalToolExecutionEnabled;
			return self();
		}

		@Override
		public B combineWith(ChatOptions.Builder<?> other) {
			super.combineWith(other);
			if (other instanceof AbstractBuilder<?> that) {
				if (that.baseUrl != null) {
					this.baseUrl = that.baseUrl;
				}
				if (that.apiKey != null) {
					this.apiKey = that.apiKey;
				}
				if (that.credential != null) {
					this.credential = that.credential;
				}
				if (that.microsoftDeploymentName != null) {
					this.microsoftDeploymentName = that.microsoftDeploymentName;
				}
				if (that.microsoftFoundryServiceVersion != null) {
					this.microsoftFoundryServiceVersion = that.microsoftFoundryServiceVersion;
				}
				if (that.organizationId != null) {
					this.organizationId = that.organizationId;
				}
				if (that.isMicrosoftFoundry != null) {
					this.isMicrosoftFoundry = that.isMicrosoftFoundry;
				}
				if (that.isGitHubModels != null) {
					this.isGitHubModels = that.isGitHubModels;
				}
				if (that.timeout != null) {
					this.timeout = that.timeout;
				}
				if (that.maxRetries != null) {
					this.maxRetries = that.maxRetries;
				}
				if (that.proxy != null) {
					this.proxy = that.proxy;
				}
				if (that.customHeaders != null) {
					this.customHeaders = that.customHeaders;
				}
				if (that.maxCompletionTokens != null) {
					this.maxCompletionTokens = that.maxCompletionTokens;
				}
				if (that.reasoningEffort != null) {
					this.reasoningEffort = that.reasoningEffort;
				}
				if (that.reasoningSummary != null) {
					this.reasoningSummary = that.reasoningSummary;
				}
				if (that.parallelToolCalls != null) {
					this.parallelToolCalls = that.parallelToolCalls;
				}
				if (that.strictTools != null) {
					this.strictTools = that.strictTools;
				}
				if (that.user != null) {
					this.user = that.user;
				}
				if (that.serviceTier != null) {
					this.serviceTier = that.serviceTier;
				}
				if (that.metadata != null) {
					this.metadata = that.metadata;
				}
				if (that.store != null) {
					this.store = that.store;
				}
				if (that.builtInTools != null) {
					this.builtInTools = that.builtInTools;
				}
				if (that.webSearchOptions != null) {
					this.webSearchOptions = that.webSearchOptions;
				}
				if (that.previousResponseId != null) {
					this.previousResponseId = that.previousResponseId;
				}
				if (that.instructions != null) {
					this.instructions = that.instructions;
				}
				if (that.truncation != null) {
					this.truncation = that.truncation;
				}
				if (that.toolChoice != null) {
					this.toolChoice = that.toolChoice;
				}
				if (that.internalToolExecutionEnabled != null) {
					this.internalToolExecutionEnabled = that.internalToolExecutionEnabled;
				}
			}
			return self();
		}

		@Override
		public OpenAiResponsesOptions build() {
			OpenAiResponsesOptions options = new OpenAiResponsesOptions();
			// connection
			options.baseUrl = this.baseUrl;
			options.apiKey = this.apiKey;
			options.credential = this.credential;
			options.model = this.model;
			options.microsoftDeploymentName = this.microsoftDeploymentName;
			options.microsoftFoundryServiceVersion = this.microsoftFoundryServiceVersion;
			options.organizationId = this.organizationId;
			options.isMicrosoftFoundry = this.isMicrosoftFoundry != null ? this.isMicrosoftFoundry : false;
			options.isGitHubModels = this.isGitHubModels != null ? this.isGitHubModels : false;
			options.timeout = this.timeout != null ? this.timeout : AbstractOpenAiOptions.DEFAULT_TIMEOUT;
			options.maxRetries = this.maxRetries != null ? this.maxRetries : AbstractOpenAiOptions.DEFAULT_MAX_RETRIES;
			options.proxy = this.proxy;
			options.customHeaders = this.customHeaders;
			// base ChatOptions / ToolCallingChatOptions
			options.temperature = this.temperature;
			options.topP = this.topP;
			options.stop = this.stopSequences;
			options.toolCallbacks = this.toolCallbacks;
			options.toolContext = this.toolContext;
			// Responses specific
			options.maxCompletionTokens = this.maxCompletionTokens;
			options.reasoningEffort = this.reasoningEffort;
			options.reasoningSummary = this.reasoningSummary;
			options.parallelToolCalls = this.parallelToolCalls;
			options.strictTools = this.strictTools;
			options.user = this.user;
			options.serviceTier = this.serviceTier;
			options.metadata = this.metadata;
			options.store = this.store;
			options.builtInTools = this.builtInTools;
			options.webSearchOptions = this.webSearchOptions;
			options.previousResponseId = this.previousResponseId;
			options.instructions = this.instructions;
			options.truncation = this.truncation;
			options.toolChoice = this.toolChoice;
			options.internalToolExecutionEnabled = this.internalToolExecutionEnabled;
			return options;
		}

	}

	/**
	 * Built-in tool types available in the OpenAI Responses API.
	 */
	public enum BuiltInTool {

		WEB_SEARCH, FILE_SEARCH, CODE_INTERPRETER

	}

	/**
	 * Configuration options for the {@code web_search} built-in tool.
	 */
	public record WebSearchOptions(@Nullable SearchContextSize searchContextSize, @Nullable UserLocation userLocation) {

		public static WebSearchOptions of(SearchContextSize searchContextSize) {
			return new WebSearchOptions(searchContextSize, null);
		}

		public static WebSearchOptions of(SearchContextSize searchContextSize, UserLocation userLocation) {
			return new WebSearchOptions(searchContextSize, userLocation);
		}

		public enum SearchContextSize {

			LOW, MEDIUM, HIGH

		}

		public record UserLocation(String type, @Nullable Approximate approximate) {

			public static UserLocation approximate(String city, String country, String region, String timezone) {
				return new UserLocation("approximate", new Approximate(city, country, region, timezone));
			}

			public record Approximate(String city, String country, String region, String timezone) {
			}
		}
	}

}
