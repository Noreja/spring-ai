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

package org.springframework.ai.openaisdk;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

import org.springframework.ai.model.tool.ToolCallingChatOptions;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.lang.Nullable;
import org.springframework.util.Assert;

/**
 * Configuration options for the OpenAI Responses API model implementation. This options
 * class supports built-in tools like web_search, file_search, and code_interpreter which
 * are only available through the Responses API.
 *
 * @author Noreja
 */
public class OpenAiSdkResponsesOptions extends AbstractOpenAiSdkOptions implements ToolCallingChatOptions {

	public static final String DEFAULT_MODEL = "gpt-4o";

	private Double temperature;

	private Double topP;

	private Integer maxCompletionTokens;

	private List<String> stop;

	private String reasoningEffort;

	private Boolean parallelToolCalls;

	private String user;

	private String serviceTier;

	private Map<String, String> metadata;

	private Boolean store;

	private List<BuiltInTool> builtInTools;

	private WebSearchOptions webSearchOptions;

	private String previousResponseId;

	private String instructions;

	private String truncation;

	private Object toolChoice;

	private List<ToolCallback> toolCallbacks = new ArrayList<>();

	private Set<String> toolNames = new HashSet<>();

	private Boolean internalToolExecutionEnabled;

	private Map<String, Object> toolContext = new HashMap<>();

	@Override
	public Double getTemperature() {
		return this.temperature;
	}

	public void setTemperature(Double temperature) {
		this.temperature = temperature;
	}

	@Override
	public Double getTopP() {
		return this.topP;
	}

	public void setTopP(Double topP) {
		this.topP = topP;
	}

	public Integer getMaxCompletionTokens() {
		return this.maxCompletionTokens;
	}

	public void setMaxCompletionTokens(Integer maxCompletionTokens) {
		this.maxCompletionTokens = maxCompletionTokens;
	}

	@Override
	public Integer getMaxTokens() {
		return this.maxCompletionTokens;
	}

	public List<String> getStop() {
		return this.stop;
	}

	public void setStop(List<String> stop) {
		this.stop = stop;
	}

	@Override
	public List<String> getStopSequences() {
		return getStop();
	}

	public void setStopSequences(List<String> stopSequences) {
		setStop(stopSequences);
	}

	public String getReasoningEffort() {
		return this.reasoningEffort;
	}

	public void setReasoningEffort(String reasoningEffort) {
		this.reasoningEffort = reasoningEffort;
	}

	public Boolean getParallelToolCalls() {
		return this.parallelToolCalls;
	}

	public void setParallelToolCalls(Boolean parallelToolCalls) {
		this.parallelToolCalls = parallelToolCalls;
	}

	public String getUser() {
		return this.user;
	}

	public void setUser(String user) {
		this.user = user;
	}

	public String getServiceTier() {
		return this.serviceTier;
	}

	public void setServiceTier(String serviceTier) {
		this.serviceTier = serviceTier;
	}

	public Map<String, String> getMetadata() {
		return this.metadata;
	}

	public void setMetadata(Map<String, String> metadata) {
		this.metadata = metadata;
	}

	public Boolean getStore() {
		return this.store;
	}

	public void setStore(Boolean store) {
		this.store = store;
	}

	public List<BuiltInTool> getBuiltInTools() {
		return this.builtInTools;
	}

	public void setBuiltInTools(List<BuiltInTool> builtInTools) {
		this.builtInTools = builtInTools;
	}

	public WebSearchOptions getWebSearchOptions() {
		return this.webSearchOptions;
	}

	public void setWebSearchOptions(WebSearchOptions webSearchOptions) {
		this.webSearchOptions = webSearchOptions;
	}

	public String getPreviousResponseId() {
		return this.previousResponseId;
	}

	public void setPreviousResponseId(String previousResponseId) {
		this.previousResponseId = previousResponseId;
	}

	public String getInstructions() {
		return this.instructions;
	}

	public void setInstructions(String instructions) {
		this.instructions = instructions;
	}

	public String getTruncation() {
		return this.truncation;
	}

	public void setTruncation(String truncation) {
		this.truncation = truncation;
	}

	public Object getToolChoice() {
		return this.toolChoice;
	}

	public void setToolChoice(Object toolChoice) {
		this.toolChoice = toolChoice;
	}

	@Override
	public List<ToolCallback> getToolCallbacks() {
		return this.toolCallbacks;
	}

	@Override
	public void setToolCallbacks(List<ToolCallback> toolCallbacks) {
		Assert.notNull(toolCallbacks, "toolCallbacks cannot be null");
		Assert.noNullElements(toolCallbacks, "toolCallbacks cannot contain null elements");
		this.toolCallbacks = toolCallbacks;
	}

	@Override
	public Set<String> getToolNames() {
		return this.toolNames;
	}

	@Override
	public void setToolNames(Set<String> toolNames) {
		Assert.notNull(toolNames, "toolNames cannot be null");
		Assert.noNullElements(toolNames, "toolNames cannot contain null elements");
		toolNames.forEach(tool -> Assert.hasText(tool, "toolNames cannot contain empty elements"));
		this.toolNames = toolNames;
	}

	@Override
	@Nullable
	public Boolean getInternalToolExecutionEnabled() {
		return this.internalToolExecutionEnabled;
	}

	@Override
	public void setInternalToolExecutionEnabled(@Nullable Boolean internalToolExecutionEnabled) {
		this.internalToolExecutionEnabled = internalToolExecutionEnabled;
	}

	@Override
	public Map<String, Object> getToolContext() {
		return this.toolContext;
	}

	@Override
	public void setToolContext(Map<String, Object> toolContext) {
		this.toolContext = toolContext;
	}

	@Override
	public Integer getTopK() {
		return null;
	}

	@Override
	public Double getFrequencyPenalty() {
		return null;
	}

	@Override
	public Double getPresencePenalty() {
		return null;
	}

	public static Builder builder() {
		return new Builder();
	}

	@Override
	public OpenAiSdkResponsesOptions copy() {
		return builder().from(this).build();
	}

	@Override
	public boolean equals(Object o) {
		if (o == null || getClass() != o.getClass()) {
			return false;
		}
		OpenAiSdkResponsesOptions that = (OpenAiSdkResponsesOptions) o;
		return Objects.equals(this.getModel(), that.getModel()) && Objects.equals(this.temperature, that.temperature)
				&& Objects.equals(this.topP, that.topP)
				&& Objects.equals(this.maxCompletionTokens, that.maxCompletionTokens)
				&& Objects.equals(this.stop, that.stop) && Objects.equals(this.reasoningEffort, that.reasoningEffort)
				&& Objects.equals(this.parallelToolCalls, that.parallelToolCalls)
				&& Objects.equals(this.user, that.user) && Objects.equals(this.serviceTier, that.serviceTier)
				&& Objects.equals(this.metadata, that.metadata) && Objects.equals(this.store, that.store)
				&& Objects.equals(this.builtInTools, that.builtInTools)
				&& Objects.equals(this.webSearchOptions, that.webSearchOptions)
				&& Objects.equals(this.previousResponseId, that.previousResponseId)
				&& Objects.equals(this.instructions, that.instructions)
				&& Objects.equals(this.truncation, that.truncation) && Objects.equals(this.toolChoice, that.toolChoice)
				&& Objects.equals(this.toolCallbacks, that.toolCallbacks)
				&& Objects.equals(this.toolNames, that.toolNames)
				&& Objects.equals(this.internalToolExecutionEnabled, that.internalToolExecutionEnabled)
				&& Objects.equals(this.toolContext, that.toolContext);
	}

	@Override
	public int hashCode() {
		return Objects.hash(this.getModel(), this.temperature, this.topP, this.maxCompletionTokens, this.stop,
				this.reasoningEffort, this.parallelToolCalls, this.user, this.serviceTier, this.metadata, this.store,
				this.builtInTools, this.webSearchOptions, this.previousResponseId, this.instructions, this.truncation,
				this.toolChoice, this.toolCallbacks, this.toolNames, this.internalToolExecutionEnabled,
				this.toolContext);
	}

	public static final class Builder {

		private final OpenAiSdkResponsesOptions options = new OpenAiSdkResponsesOptions();

		public Builder from(OpenAiSdkResponsesOptions fromOptions) {
			this.options.setBaseUrl(fromOptions.getBaseUrl());
			this.options.setApiKey(fromOptions.getApiKey());
			this.options.setCredential(fromOptions.getCredential());
			this.options.setModel(fromOptions.getModel());
			this.options.setDeploymentName(fromOptions.getDeploymentName());
			this.options.setMicrosoftFoundryServiceVersion(fromOptions.getMicrosoftFoundryServiceVersion());
			this.options.setOrganizationId(fromOptions.getOrganizationId());
			this.options.setMicrosoftFoundry(fromOptions.isMicrosoftFoundry());
			this.options.setGitHubModels(fromOptions.isGitHubModels());
			this.options.setTimeout(fromOptions.getTimeout());
			this.options.setMaxRetries(fromOptions.getMaxRetries());
			this.options.setProxy(fromOptions.getProxy());
			this.options.setCustomHeaders(
					fromOptions.getCustomHeaders() != null ? new HashMap<>(fromOptions.getCustomHeaders()) : null);
			this.options.setTemperature(fromOptions.getTemperature());
			this.options.setTopP(fromOptions.getTopP());
			this.options.setMaxCompletionTokens(fromOptions.getMaxCompletionTokens());
			this.options.setStop(fromOptions.getStop() != null ? new ArrayList<>(fromOptions.getStop()) : null);
			this.options.setReasoningEffort(fromOptions.getReasoningEffort());
			this.options.setParallelToolCalls(fromOptions.getParallelToolCalls());
			this.options.setUser(fromOptions.getUser());
			this.options.setServiceTier(fromOptions.getServiceTier());
			this.options.setMetadata(fromOptions.getMetadata());
			this.options.setStore(fromOptions.getStore());
			this.options.setBuiltInTools(
					fromOptions.getBuiltInTools() != null ? new ArrayList<>(fromOptions.getBuiltInTools()) : null);
			this.options.setWebSearchOptions(fromOptions.getWebSearchOptions());
			this.options.setPreviousResponseId(fromOptions.getPreviousResponseId());
			this.options.setInstructions(fromOptions.getInstructions());
			this.options.setTruncation(fromOptions.getTruncation());
			this.options.setToolChoice(fromOptions.getToolChoice());
			this.options.setToolCallbacks(new ArrayList<>(fromOptions.getToolCallbacks()));
			this.options.setToolNames(new HashSet<>(fromOptions.getToolNames()));
			this.options.setInternalToolExecutionEnabled(fromOptions.getInternalToolExecutionEnabled());
			this.options.setToolContext(new HashMap<>(fromOptions.getToolContext()));
			return this;
		}

		public Builder merge(OpenAiSdkResponsesOptions from) {
			if (from.getBaseUrl() != null) {
				this.options.setBaseUrl(from.getBaseUrl());
			}
			if (from.getApiKey() != null) {
				this.options.setApiKey(from.getApiKey());
			}
			if (from.getCredential() != null) {
				this.options.setCredential(from.getCredential());
			}
			if (from.getModel() != null) {
				this.options.setModel(from.getModel());
			}
			if (from.getDeploymentName() != null) {
				this.options.setDeploymentName(from.getDeploymentName());
			}
			if (from.getMicrosoftFoundryServiceVersion() != null) {
				this.options.setMicrosoftFoundryServiceVersion(from.getMicrosoftFoundryServiceVersion());
			}
			if (from.getOrganizationId() != null) {
				this.options.setOrganizationId(from.getOrganizationId());
			}
			this.options.setMicrosoftFoundry(from.isMicrosoftFoundry());
			this.options.setGitHubModels(from.isGitHubModels());
			if (from.getTimeout() != null) {
				this.options.setTimeout(from.getTimeout());
			}
			if (from.getMaxRetries() != null) {
				this.options.setMaxRetries(from.getMaxRetries());
			}
			if (from.getProxy() != null) {
				this.options.setProxy(from.getProxy());
			}
			if (from.getCustomHeaders() != null) {
				this.options.setCustomHeaders(from.getCustomHeaders());
			}
			if (from.getTemperature() != null) {
				this.options.setTemperature(from.getTemperature());
			}
			if (from.getTopP() != null) {
				this.options.setTopP(from.getTopP());
			}
			if (from.getMaxCompletionTokens() != null) {
				this.options.setMaxCompletionTokens(from.getMaxCompletionTokens());
			}
			if (from.getStop() != null) {
				this.options.setStop(new ArrayList<>(from.getStop()));
			}
			if (from.getReasoningEffort() != null) {
				this.options.setReasoningEffort(from.getReasoningEffort());
			}
			if (from.getParallelToolCalls() != null) {
				this.options.setParallelToolCalls(from.getParallelToolCalls());
			}
			if (from.getUser() != null) {
				this.options.setUser(from.getUser());
			}
			if (from.getServiceTier() != null) {
				this.options.setServiceTier(from.getServiceTier());
			}
			if (from.getMetadata() != null) {
				this.options.setMetadata(from.getMetadata());
			}
			if (from.getStore() != null) {
				this.options.setStore(from.getStore());
			}
			if (from.getBuiltInTools() != null) {
				this.options.setBuiltInTools(new ArrayList<>(from.getBuiltInTools()));
			}
			if (from.getWebSearchOptions() != null) {
				this.options.setWebSearchOptions(from.getWebSearchOptions());
			}
			if (from.getPreviousResponseId() != null) {
				this.options.setPreviousResponseId(from.getPreviousResponseId());
			}
			if (from.getInstructions() != null) {
				this.options.setInstructions(from.getInstructions());
			}
			if (from.getTruncation() != null) {
				this.options.setTruncation(from.getTruncation());
			}
			if (from.getToolChoice() != null) {
				this.options.setToolChoice(from.getToolChoice());
			}
			if (!from.getToolCallbacks().isEmpty()) {
				this.options.setToolCallbacks(new ArrayList<>(from.getToolCallbacks()));
			}
			if (!from.getToolNames().isEmpty()) {
				this.options.setToolNames(new HashSet<>(from.getToolNames()));
			}
			if (from.getInternalToolExecutionEnabled() != null) {
				this.options.setInternalToolExecutionEnabled(from.getInternalToolExecutionEnabled());
			}
			if (!from.getToolContext().isEmpty()) {
				this.options.setToolContext(new HashMap<>(from.getToolContext()));
			}
			return this;
		}

		public Builder model(String model) {
			this.options.setModel(model);
			return this;
		}

		public Builder deploymentName(String deploymentName) {
			this.options.setDeploymentName(deploymentName);
			return this;
		}

		public Builder baseUrl(String baseUrl) {
			this.options.setBaseUrl(baseUrl);
			return this;
		}

		public Builder apiKey(String apiKey) {
			this.options.setApiKey(apiKey);
			return this;
		}

		public Builder temperature(Double temperature) {
			this.options.setTemperature(temperature);
			return this;
		}

		public Builder topP(Double topP) {
			this.options.setTopP(topP);
			return this;
		}

		public Builder maxCompletionTokens(Integer maxCompletionTokens) {
			this.options.setMaxCompletionTokens(maxCompletionTokens);
			return this;
		}

		public Builder stop(List<String> stop) {
			this.options.setStop(stop);
			return this;
		}

		public Builder reasoningEffort(String reasoningEffort) {
			this.options.setReasoningEffort(reasoningEffort);
			return this;
		}

		public Builder parallelToolCalls(Boolean parallelToolCalls) {
			this.options.setParallelToolCalls(parallelToolCalls);
			return this;
		}

		public Builder user(String user) {
			this.options.setUser(user);
			return this;
		}

		public Builder serviceTier(String serviceTier) {
			this.options.setServiceTier(serviceTier);
			return this;
		}

		public Builder metadata(Map<String, String> metadata) {
			this.options.setMetadata(metadata);
			return this;
		}

		public Builder store(Boolean store) {
			this.options.setStore(store);
			return this;
		}

		public Builder builtInTools(List<BuiltInTool> builtInTools) {
			this.options.setBuiltInTools(builtInTools);
			return this;
		}

		public Builder builtInTools(BuiltInTool... builtInTools) {
			this.options.setBuiltInTools(Arrays.asList(builtInTools));
			return this;
		}

		public Builder webSearchOptions(WebSearchOptions webSearchOptions) {
			this.options.setWebSearchOptions(webSearchOptions);
			return this;
		}

		public Builder previousResponseId(String previousResponseId) {
			this.options.setPreviousResponseId(previousResponseId);
			return this;
		}

		public Builder instructions(String instructions) {
			this.options.setInstructions(instructions);
			return this;
		}

		public Builder truncation(String truncation) {
			this.options.setTruncation(truncation);
			return this;
		}

		public Builder toolChoice(Object toolChoice) {
			this.options.setToolChoice(toolChoice);
			return this;
		}

		public Builder toolCallbacks(List<ToolCallback> toolCallbacks) {
			this.options.setToolCallbacks(toolCallbacks);
			return this;
		}

		public Builder toolCallbacks(ToolCallback... toolCallbacks) {
			this.options.setToolCallbacks(Arrays.asList(toolCallbacks));
			return this;
		}

		public Builder toolNames(Set<String> toolNames) {
			Assert.notNull(toolNames, "toolNames cannot be null");
			this.options.setToolNames(toolNames);
			return this;
		}

		public Builder toolNames(String... toolNames) {
			Assert.notNull(toolNames, "toolNames cannot be null");
			this.options.setToolNames(new HashSet<>(Arrays.asList(toolNames)));
			return this;
		}

		public Builder internalToolExecutionEnabled(@Nullable Boolean internalToolExecutionEnabled) {
			this.options.setInternalToolExecutionEnabled(internalToolExecutionEnabled);
			return this;
		}

		public Builder toolContext(Map<String, Object> toolContext) {
			this.options.setToolContext(toolContext);
			return this;
		}

		public OpenAiSdkResponsesOptions build() {
			return this.options;
		}

	}

	/**
	 * Built-in tool types available in the OpenAI Responses API.
	 */
	public enum BuiltInTool {

		WEB_SEARCH, FILE_SEARCH, CODE_INTERPRETER

	}

	/**
	 * Configuration options for the web_search built-in tool.
	 */
	public record WebSearchOptions(SearchContextSize searchContextSize, UserLocation userLocation) {

		public static WebSearchOptions of(SearchContextSize searchContextSize) {
			return new WebSearchOptions(searchContextSize, null);
		}

		public static WebSearchOptions of(SearchContextSize searchContextSize, UserLocation userLocation) {
			return new WebSearchOptions(searchContextSize, userLocation);
		}

		public enum SearchContextSize {

			LOW, MEDIUM, HIGH

		}

		public record UserLocation(String type, Approximate approximate) {

			public static UserLocation approximate(String city, String country, String region, String timezone) {
				return new UserLocation("approximate", new Approximate(city, country, region, timezone));
			}

			public record Approximate(String city, String country, String region, String timezone) {
			}
		}
	}

}
