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
import java.util.Map;

import org.jspecify.annotations.Nullable;

import org.springframework.ai.openai.OpenAiResponsesOptions;
import org.springframework.ai.openai.OpenAiResponsesOptions.BuiltInTool;
import org.springframework.ai.openai.OpenAiResponsesOptions.WebSearchOptions;
import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Autoconfiguration properties for the OpenAI Responses API model.
 *
 * @author Noreja
 */
@ConfigurationProperties(OpenAiResponsesProperties.CONFIG_PREFIX)
public class OpenAiResponsesProperties extends AbstractOpenAiProperties {

	public static final String CONFIG_PREFIX = "spring.ai.openai.responses";

	private boolean enabled = false;

	private @Nullable Double temperature = 0.7;

	private @Nullable Double topP;

	private @Nullable Integer maxCompletionTokens;

	private @Nullable List<String> stop;

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

	public boolean isEnabled() {
		return this.enabled;
	}

	public void setEnabled(boolean enabled) {
		this.enabled = enabled;
	}

	public @Nullable Double getTemperature() {
		return this.temperature;
	}

	public void setTemperature(@Nullable Double temperature) {
		this.temperature = temperature;
	}

	public @Nullable Double getTopP() {
		return this.topP;
	}

	public void setTopP(@Nullable Double topP) {
		this.topP = topP;
	}

	public @Nullable Integer getMaxCompletionTokens() {
		return this.maxCompletionTokens;
	}

	public void setMaxCompletionTokens(@Nullable Integer maxCompletionTokens) {
		this.maxCompletionTokens = maxCompletionTokens;
	}

	public @Nullable List<String> getStop() {
		return this.stop;
	}

	public void setStop(@Nullable List<String> stop) {
		this.stop = stop;
	}

	public @Nullable String getReasoningEffort() {
		return this.reasoningEffort;
	}

	public void setReasoningEffort(@Nullable String reasoningEffort) {
		this.reasoningEffort = reasoningEffort;
	}

	public @Nullable String getReasoningSummary() {
		return this.reasoningSummary;
	}

	public void setReasoningSummary(@Nullable String reasoningSummary) {
		this.reasoningSummary = reasoningSummary;
	}

	public @Nullable Boolean getParallelToolCalls() {
		return this.parallelToolCalls;
	}

	public void setParallelToolCalls(@Nullable Boolean parallelToolCalls) {
		this.parallelToolCalls = parallelToolCalls;
	}

	public @Nullable Boolean getStrictTools() {
		return this.strictTools;
	}

	public void setStrictTools(@Nullable Boolean strictTools) {
		this.strictTools = strictTools;
	}

	public @Nullable String getUser() {
		return this.user;
	}

	public void setUser(@Nullable String user) {
		this.user = user;
	}

	public @Nullable String getServiceTier() {
		return this.serviceTier;
	}

	public void setServiceTier(@Nullable String serviceTier) {
		this.serviceTier = serviceTier;
	}

	public @Nullable Map<String, String> getMetadata() {
		return this.metadata;
	}

	public void setMetadata(@Nullable Map<String, String> metadata) {
		this.metadata = metadata;
	}

	public @Nullable Boolean getStore() {
		return this.store;
	}

	public void setStore(@Nullable Boolean store) {
		this.store = store;
	}

	public @Nullable List<BuiltInTool> getBuiltInTools() {
		return this.builtInTools;
	}

	public void setBuiltInTools(@Nullable List<BuiltInTool> builtInTools) {
		this.builtInTools = builtInTools;
	}

	public @Nullable WebSearchOptions getWebSearchOptions() {
		return this.webSearchOptions;
	}

	public void setWebSearchOptions(@Nullable WebSearchOptions webSearchOptions) {
		this.webSearchOptions = webSearchOptions;
	}

	public @Nullable String getPreviousResponseId() {
		return this.previousResponseId;
	}

	public void setPreviousResponseId(@Nullable String previousResponseId) {
		this.previousResponseId = previousResponseId;
	}

	public @Nullable String getInstructions() {
		return this.instructions;
	}

	public void setInstructions(@Nullable String instructions) {
		this.instructions = instructions;
	}

	public @Nullable String getTruncation() {
		return this.truncation;
	}

	public void setTruncation(@Nullable String truncation) {
		this.truncation = truncation;
	}

	public @Nullable Object getToolChoice() {
		return this.toolChoice;
	}

	public void setToolChoice(@Nullable Object toolChoice) {
		this.toolChoice = toolChoice;
	}

	public @Nullable Boolean getInternalToolExecutionEnabled() {
		return this.internalToolExecutionEnabled;
	}

	public void setInternalToolExecutionEnabled(@Nullable Boolean internalToolExecutionEnabled) {
		this.internalToolExecutionEnabled = internalToolExecutionEnabled;
	}

	public OpenAiResponsesOptions toOptions() {
		return OpenAiResponsesOptions.builder()
			.model(getModel() != null ? getModel() : OpenAiResponsesOptions.DEFAULT_MODEL)
			.temperature(this.temperature)
			.topP(this.topP)
			.maxCompletionTokens(this.maxCompletionTokens)
			.stopSequences(this.stop)
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
			.internalToolExecutionEnabled(this.internalToolExecutionEnabled)
			.build();
	}

}
