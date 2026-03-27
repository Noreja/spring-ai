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
import java.util.Base64;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.openai.client.OpenAIClient;
import com.openai.client.OpenAIClientAsync;
import com.openai.core.JsonValue;
import com.openai.models.Reasoning;
import com.openai.models.ReasoningEffort;
import com.openai.models.responses.FunctionTool;
import com.openai.models.responses.Response;
import com.openai.models.responses.ResponseCompletedEvent;
import com.openai.models.responses.ResponseCreateParams;
import com.openai.models.responses.ResponseFunctionCallArgumentsDeltaEvent;
import com.openai.models.responses.ResponseFunctionCallArgumentsDoneEvent;
import com.openai.models.responses.ResponseFunctionToolCall;
import com.openai.models.responses.ResponseFunctionWebSearch;
import com.openai.models.responses.ResponseInputContent;
import com.openai.models.responses.ResponseInputImage;
import com.openai.models.responses.ResponseInputItem;
import com.openai.models.responses.ResponseInputText;
import com.openai.models.responses.ResponseOutputItem;
import com.openai.models.responses.ResponseOutputMessage;
import com.openai.models.responses.ResponseOutputText;
import com.openai.models.responses.ResponseReasoningItem;
import com.openai.models.responses.ResponseStatus;
import com.openai.models.responses.ResponseTextDeltaEvent;
import com.openai.models.responses.ResponseUsage;
import com.openai.models.responses.Tool;
import com.openai.models.responses.WebSearchTool;
import io.micrometer.observation.Observation;
import io.micrometer.observation.ObservationRegistry;
import io.micrometer.observation.contextpropagation.ObservationThreadLocalAccessor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import reactor.core.publisher.Flux;
import reactor.core.scheduler.Schedulers;

import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.MessageType;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.metadata.ChatGenerationMetadata;
import org.springframework.ai.chat.metadata.ChatResponseMetadata;
import org.springframework.ai.chat.metadata.DefaultUsage;
import org.springframework.ai.chat.metadata.EmptyUsage;
import org.springframework.ai.chat.metadata.Usage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.observation.ChatModelObservationContext;
import org.springframework.ai.chat.observation.ChatModelObservationConvention;
import org.springframework.ai.chat.observation.ChatModelObservationDocumentation;
import org.springframework.ai.chat.observation.DefaultChatModelObservationConvention;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.content.Media;
import org.springframework.ai.model.ModelOptionsUtils;
import org.springframework.ai.model.tool.DefaultToolExecutionEligibilityPredicate;
import org.springframework.ai.model.tool.ToolCallingChatOptions;
import org.springframework.ai.model.tool.ToolCallingManager;
import org.springframework.ai.model.tool.ToolExecutionEligibilityPredicate;
import org.springframework.ai.model.tool.ToolExecutionResult;
import org.springframework.ai.model.tool.internal.ToolCallReactiveContextHolder;
import org.springframework.ai.observation.conventions.AiProvider;
import org.springframework.ai.openaisdk.setup.OpenAiSdkSetup;
import org.springframework.ai.support.UsageCalculator;
import org.springframework.ai.tool.definition.ToolDefinition;
import org.springframework.util.Assert;
import org.springframework.util.CollectionUtils;
import org.springframework.util.StringUtils;

/**
 * Chat Model implementation using the OpenAI Responses API via the OpenAI Java SDK. This
 * model supports built-in tools like web_search, file_search, and code_interpreter which
 * are only available through the Responses API endpoint.
 *
 * @author Noreja
 */
public class OpenAiSdkResponsesModel implements ChatModel {

	private static final ChatModelObservationConvention DEFAULT_OBSERVATION_CONVENTION = new DefaultChatModelObservationConvention();

	private static final ToolCallingManager DEFAULT_TOOL_CALLING_MANAGER = ToolCallingManager.builder().build();

	private final Logger logger = LoggerFactory.getLogger(OpenAiSdkResponsesModel.class);

	private final OpenAIClient openAiClient;

	private final OpenAIClientAsync openAiClientAsync;

	private final OpenAiSdkResponsesOptions options;

	private final ObservationRegistry observationRegistry;

	private final ToolCallingManager toolCallingManager;

	private final ToolExecutionEligibilityPredicate toolExecutionEligibilityPredicate;

	private ChatModelObservationConvention observationConvention = DEFAULT_OBSERVATION_CONVENTION;

	public OpenAiSdkResponsesModel() {
		this(null, null, null, null, null, null);
	}

	public OpenAiSdkResponsesModel(OpenAiSdkResponsesOptions options) {
		this(null, null, options, null, null, null);
	}

	public OpenAiSdkResponsesModel(OpenAiSdkResponsesOptions options, ToolCallingManager toolCallingManager,
			ObservationRegistry observationRegistry) {
		this(null, null, options, toolCallingManager, observationRegistry, null);
	}

	public OpenAiSdkResponsesModel(OpenAIClient openAiClient, OpenAIClientAsync openAiClientAsync,
			OpenAiSdkResponsesOptions options, ToolCallingManager toolCallingManager,
			ObservationRegistry observationRegistry,
			ToolExecutionEligibilityPredicate toolExecutionEligibilityPredicate) {

		if (options == null) {
			this.options = OpenAiSdkResponsesOptions.builder().model(OpenAiSdkResponsesOptions.DEFAULT_MODEL).build();
		}
		else {
			this.options = options;
		}
		this.openAiClient = Objects.requireNonNullElseGet(openAiClient,
				() -> OpenAiSdkSetup.setupSyncClient(this.options.getBaseUrl(), this.options.getApiKey(),
						this.options.getCredential(), this.options.getMicrosoftDeploymentName(),
						this.options.getMicrosoftFoundryServiceVersion(), this.options.getOrganizationId(),
						this.options.isMicrosoftFoundry(), this.options.isGitHubModels(), this.options.getModel(),
						this.options.getTimeout(), this.options.getMaxRetries(), this.options.getProxy(),
						this.options.getCustomHeaders()));

		this.openAiClientAsync = Objects.requireNonNullElseGet(openAiClientAsync,
				() -> OpenAiSdkSetup.setupAsyncClient(this.options.getBaseUrl(), this.options.getApiKey(),
						this.options.getCredential(), this.options.getMicrosoftDeploymentName(),
						this.options.getMicrosoftFoundryServiceVersion(), this.options.getOrganizationId(),
						this.options.isMicrosoftFoundry(), this.options.isGitHubModels(), this.options.getModel(),
						this.options.getTimeout(), this.options.getMaxRetries(), this.options.getProxy(),
						this.options.getCustomHeaders()));

		this.observationRegistry = Objects.requireNonNullElse(observationRegistry, ObservationRegistry.NOOP);
		this.toolCallingManager = Objects.requireNonNullElse(toolCallingManager, DEFAULT_TOOL_CALLING_MANAGER);
		this.toolExecutionEligibilityPredicate = Objects.requireNonNullElse(toolExecutionEligibilityPredicate,
				new DefaultToolExecutionEligibilityPredicate());
	}

	@Override
	public ChatResponse call(Prompt prompt) {
		Prompt requestPrompt = buildRequestPrompt(prompt);
		return this.internalCall(requestPrompt, null);
	}

	public ChatResponse internalCall(Prompt prompt, ChatResponse previousChatResponse) {

		ResponseCreateParams request = createRequest(prompt);

		ChatModelObservationContext observationContext = ChatModelObservationContext.builder()
			.prompt(prompt)
			.provider(AiProvider.OPENAI_SDK.value())
			.build();

		ChatResponse response = ChatModelObservationDocumentation.CHAT_MODEL_OPERATION
			.observation(this.observationConvention, DEFAULT_OBSERVATION_CONVENTION, () -> observationContext,
					this.observationRegistry)
			.observe(() -> {
				Response apiResponse = this.openAiClient.responses().create(request);
				ChatResponse chatResponse = processResponse(apiResponse, previousChatResponse);
				observationContext.setResponse(chatResponse);
				return chatResponse;
			});

		if (this.toolExecutionEligibilityPredicate.isToolExecutionRequired(prompt.getOptions(), response)) {
			var toolExecutionResult = this.toolCallingManager.executeToolCalls(prompt, response);
			if (toolExecutionResult.returnDirect()) {
				return ChatResponse.builder()
					.from(response)
					.generations(ToolExecutionResult.buildGenerations(toolExecutionResult))
					.build();
			}
			else {
				return this.internalCall(new Prompt(toolExecutionResult.conversationHistory(), prompt.getOptions()),
						response);
			}
		}

		return response;
	}

	@Override
	public Flux<ChatResponse> stream(Prompt prompt) {
		Prompt requestPrompt = buildRequestPrompt(prompt);
		return internalStream(requestPrompt, null);
	}

	public Flux<ChatResponse> internalStream(Prompt prompt, ChatResponse previousChatResponse) {
		return Flux.deferContextual(contextView -> {
			ResponseCreateParams request = createRequest(prompt);

			final ChatModelObservationContext observationContext = ChatModelObservationContext.builder()
				.prompt(prompt)
				.provider(AiProvider.OPENAI_SDK.value())
				.build();
			Observation observation = ChatModelObservationDocumentation.CHAT_MODEL_OPERATION.observation(
					this.observationConvention, DEFAULT_OBSERVATION_CONVENTION, () -> observationContext,
					this.observationRegistry);
			observation.parentObservation(contextView.getOrDefault(ObservationThreadLocalAccessor.KEY, null)).start();

			// Accumulate function call arguments across stream events
			ConcurrentHashMap<String, StringBuilder> functionCallArgs = new ConcurrentHashMap<>();
			ConcurrentHashMap<String, String> functionCallNames = new ConcurrentHashMap<>();
			ConcurrentHashMap<String, String> functionCallIds = new ConcurrentHashMap<>();

			Flux<ChatResponse> chatResponses = Flux.<ChatResponse>create(sink -> {
				this.openAiClientAsync.responses().createStreaming(request).subscribe(event -> {
					try {
						if (event.isOutputTextDelta()) {
							ResponseTextDeltaEvent textDelta = event.asOutputTextDelta();
							String delta = textDelta.delta();
							AssistantMessage assistantMessage = AssistantMessage.builder().content(delta).build();
							Generation generation = new Generation(assistantMessage,
									ChatGenerationMetadata.builder().build());
							sink.next(new ChatResponse(List.of(generation)));
						}
						else if (event.isFunctionCallArgumentsDelta()) {
							ResponseFunctionCallArgumentsDeltaEvent argsDelta = event.asFunctionCallArgumentsDelta();
							String itemId = argsDelta.itemId();
							functionCallArgs.computeIfAbsent(itemId, k -> new StringBuilder())
								.append(argsDelta.delta());
						}
						else if (event.isFunctionCallArgumentsDone()) {
							ResponseFunctionCallArgumentsDoneEvent argsDone = event.asFunctionCallArgumentsDone();
							String itemId = argsDone.itemId();
							functionCallNames.put(itemId, argsDone.name());
							functionCallIds.put(itemId, argsDone.itemId());
						}
						else if (event.isCompleted()) {
							ResponseCompletedEvent completed = event.asCompleted();
							Response apiResponse = completed.response();
							// Build a final response with only tool calls and
							// usage/metadata — text was already streamed via deltas
							ChatResponse fullResponse = processResponse(apiResponse, previousChatResponse);
							observationContext.setResponse(fullResponse);

							// Extract tool calls from the full response
							AssistantMessage fullMsg = fullResponse.getResult() != null
									? fullResponse.getResult().getOutput() : null;
							List<AssistantMessage.ToolCall> completedToolCalls = fullMsg != null
									? fullMsg.getToolCalls() : List.of();

							// Emit a metadata-only response (empty text) with usage and
							// any tool calls
							AssistantMessage metaMessage = AssistantMessage.builder()
								.content("")
								.toolCalls(completedToolCalls)
								.properties(fullMsg != null ? fullMsg.getMetadata() : Map.of())
								.build();
							Generation metaGen = new Generation(metaMessage,
									fullResponse.getResult() != null ? fullResponse.getResult().getMetadata()
											: ChatGenerationMetadata.builder().build());
							sink.next(new ChatResponse(List.of(metaGen), fullResponse.getMetadata()));
						}
					}
					catch (Exception e) {
						logger.error("Error processing response stream event", e);
						sink.error(e);
					}
				}).onCompleteFuture().whenComplete((unused, throwable) -> {
					if (throwable != null) {
						sink.error(throwable);
					}
					else {
						sink.complete();
					}
				});
			});

			Flux<ChatResponse> flux = chatResponses
				.contextWrite(ctx -> ctx.put(ObservationThreadLocalAccessor.KEY, observation));

			return flux.collectList().flatMapMany(list -> {
				if (list.isEmpty()) {
					return Flux.empty();
				}

				// Check if any response in the stream contains function tool calls
				boolean hasToolCalls = list.stream()
					.map(this::safeAssistantMessage)
					.filter(Objects::nonNull)
					.anyMatch(am -> !CollectionUtils.isEmpty(am.getToolCalls()));

				if (hasToolCalls) {
					// Find the final complete response (last one with tool calls)
					ChatResponse aggregated = list.get(list.size() - 1);
					return Flux.deferContextual(ctx -> {
						ToolExecutionResult toolExecutionResult;
						try {
							ToolCallReactiveContextHolder.setContext(ctx);
							toolExecutionResult = this.toolCallingManager.executeToolCalls(prompt, aggregated);
						}
						finally {
							ToolCallReactiveContextHolder.clearContext();
						}
						if (toolExecutionResult.returnDirect()) {
							return Flux.just(ChatResponse.builder()
								.from(aggregated)
								.generations(ToolExecutionResult.buildGenerations(toolExecutionResult))
								.build());
						}
						return this.internalStream(
								new Prompt(toolExecutionResult.conversationHistory(), prompt.getOptions()), aggregated);
					}).subscribeOn(Schedulers.boundedElastic());
				}
				return Flux.fromIterable(list);
			}).doOnError(observation::error).doFinally(s -> observation.stop());
		});
	}

	private AssistantMessage safeAssistantMessage(ChatResponse response) {
		if (response == null) {
			return null;
		}
		Generation gen = response.getResult();
		if (gen == null) {
			return null;
		}
		return gen.getOutput();
	}

	Prompt buildRequestPrompt(Prompt prompt) {
		OpenAiSdkResponsesOptions runtimeOptions = null;
		if (prompt.getOptions() != null) {
			if (prompt.getOptions() instanceof ToolCallingChatOptions toolCallingChatOptions) {
				runtimeOptions = ModelOptionsUtils.copyToTarget(toolCallingChatOptions, ToolCallingChatOptions.class,
						OpenAiSdkResponsesOptions.class);
			}
			else {
				runtimeOptions = ModelOptionsUtils.copyToTarget(prompt.getOptions(), ChatOptions.class,
						OpenAiSdkResponsesOptions.class);
			}
		}

		OpenAiSdkResponsesOptions requestOptions = OpenAiSdkResponsesOptions.builder()
			.from(this.options)
			.merge(runtimeOptions != null ? runtimeOptions : OpenAiSdkResponsesOptions.builder().build())
			.build();

		if (runtimeOptions != null) {
			requestOptions.setInternalToolExecutionEnabled(runtimeOptions.getInternalToolExecutionEnabled() != null
					? runtimeOptions.getInternalToolExecutionEnabled()
					: this.options.getInternalToolExecutionEnabled());
			requestOptions.setToolNames(
					ToolCallingChatOptions.mergeToolNames(runtimeOptions.getToolNames(), this.options.getToolNames()));
			requestOptions.setToolCallbacks(ToolCallingChatOptions.mergeToolCallbacks(runtimeOptions.getToolCallbacks(),
					this.options.getToolCallbacks()));
			requestOptions.setToolContext(ToolCallingChatOptions.mergeToolContext(runtimeOptions.getToolContext(),
					this.options.getToolContext()));
		}
		else {
			requestOptions.setInternalToolExecutionEnabled(this.options.getInternalToolExecutionEnabled());
			requestOptions.setToolNames(this.options.getToolNames());
			requestOptions.setToolCallbacks(this.options.getToolCallbacks());
			requestOptions.setToolContext(this.options.getToolContext());
		}

		ToolCallingChatOptions.validateToolCallbacks(requestOptions.getToolCallbacks());

		return new Prompt(prompt.getInstructions(), requestOptions);
	}

	ResponseCreateParams createRequest(Prompt prompt) {

		OpenAiSdkResponsesOptions requestOptions = (OpenAiSdkResponsesOptions) prompt.getOptions();

		ResponseCreateParams.Builder builder = ResponseCreateParams.builder();

		// Set model
		if (requestOptions.getDeploymentName() != null) {
			builder.model(requestOptions.getDeploymentName());
		}
		else if (requestOptions.getModel() != null) {
			builder.model(requestOptions.getModel());
		}

		// Convert messages to input items
		List<ResponseInputItem> inputItems = new ArrayList<>();
		String systemInstructions = null;

		for (var message : prompt.getInstructions()) {
			if (message.getMessageType() == MessageType.SYSTEM) {
				// Responses API uses instructions field for system messages
				systemInstructions = message.getText();
			}
			else if (message.getMessageType() == MessageType.USER) {
				List<ResponseInputContent> contentParts = new ArrayList<>();

				if (message instanceof UserMessage userMessage && !CollectionUtils.isEmpty(userMessage.getMedia())) {
					if (StringUtils.hasText(message.getText())) {
						contentParts.add(ResponseInputContent
							.ofInputText(ResponseInputText.builder().text(message.getText()).build()));
					}
					for (Media media : userMessage.getMedia()) {
						String mimeType = media.getMimeType().toString();
						if (mimeType.startsWith("image/")) {
							String imageUrl;
							if (media.getData() instanceof java.net.URI uri) {
								imageUrl = uri.toString();
							}
							else if (media.getData() instanceof String text) {
								imageUrl = text;
							}
							else if (media.getData() instanceof byte[] bytes) {
								imageUrl = "data:" + mimeType + ";base64," + Base64.getEncoder().encodeToString(bytes);
							}
							else {
								logger.warn("Unsupported image data type: {}", media.getData().getClass());
								continue;
							}
							contentParts.add(ResponseInputContent
								.ofInputImage(ResponseInputImage.builder().imageUrl(imageUrl).build()));
						}
						else {
							// For other media types, include as text
							contentParts.add(ResponseInputContent
								.ofInputText(ResponseInputText.builder().text(mediaToText(media)).build()));
						}
					}
				}
				else {
					contentParts.add(ResponseInputContent
						.ofInputText(ResponseInputText.builder().text(message.getText()).build()));
				}

				inputItems.add(ResponseInputItem.ofMessage(ResponseInputItem.Message.builder()
					.role(ResponseInputItem.Message.Role.USER)
					.content(contentParts)
					.build()));
			}
			else if (message.getMessageType() == MessageType.ASSISTANT) {
				var assistantMessage = (AssistantMessage) message;

				// Add the assistant message text as an output message in the conversation
				// history
				if (StringUtils.hasText(assistantMessage.getText())) {
					// Use the original response ID if available from metadata, otherwise
					// generate one
					String msgId = assistantMessage.getMetadata() != null
							&& assistantMessage.getMetadata().containsKey("id")
									? assistantMessage.getMetadata().get("id").toString()
									: "msg_" + java.util.UUID.randomUUID().toString().replace("-", "");
					inputItems.add(ResponseInputItem.ofResponseOutputMessage(ResponseOutputMessage.builder()
						.id(msgId)
						.content(List.of(ResponseOutputMessage.Content.ofOutputText(ResponseOutputText.builder()
							.text(assistantMessage.getText())
							.annotations(List.of())
							.build())))
						.status(ResponseOutputMessage.Status.COMPLETED)
						.build()));
				}

				// Add function calls as separate input items
				if (!CollectionUtils.isEmpty(assistantMessage.getToolCalls())) {
					for (var toolCall : assistantMessage.getToolCalls()) {
						inputItems.add(ResponseInputItem.ofFunctionCall(ResponseFunctionToolCall.builder()
							.callId(toolCall.id())
							.name(toolCall.name())
							.arguments(toolCall.arguments())
							.build()));
					}
				}
			}
			else if (message.getMessageType() == MessageType.TOOL) {
				ToolResponseMessage toolMessage = (ToolResponseMessage) message;
				for (var toolResponse : toolMessage.getResponses()) {
					inputItems.add(ResponseInputItem.ofFunctionCallOutput(ResponseInputItem.FunctionCallOutput.builder()
						.callId(toolResponse.id())
						.output(ResponseInputItem.FunctionCallOutput.Output.ofString(toolResponse.responseData()))
						.build()));
				}
			}
		}

		builder.inputOfResponse(inputItems);

		// Set instructions (system message)
		if (requestOptions.getInstructions() != null) {
			builder.instructions(requestOptions.getInstructions());
		}
		else if (systemInstructions != null) {
			builder.instructions(systemInstructions);
		}

		// Set common parameters
		if (requestOptions.getTemperature() != null) {
			builder.temperature(requestOptions.getTemperature());
		}
		if (requestOptions.getTopP() != null) {
			builder.topP(requestOptions.getTopP());
		}
		if (requestOptions.getMaxCompletionTokens() != null) {
			builder.maxOutputTokens(requestOptions.getMaxCompletionTokens().longValue());
		}
		if (requestOptions.getUser() != null) {
			builder.user(requestOptions.getUser());
		}
		if (requestOptions.getParallelToolCalls() != null) {
			builder.parallelToolCalls(requestOptions.getParallelToolCalls());
		}
		if (requestOptions.getReasoningEffort() != null) {
			builder.reasoning(Reasoning.builder()
				.effort(ReasoningEffort.of(requestOptions.getReasoningEffort().toLowerCase()))
				.build());
		}
		if (requestOptions.getStore() != null) {
			builder.store(requestOptions.getStore());
		}
		if (requestOptions.getMetadata() != null && !requestOptions.getMetadata().isEmpty()) {
			builder.metadata(ResponseCreateParams.Metadata.builder()
				.putAllAdditionalProperties(requestOptions.getMetadata()
					.entrySet()
					.stream()
					.collect(HashMap::new, (m, e) -> m.put(e.getKey(), JsonValue.from(e.getValue())), HashMap::putAll))
				.build());
		}
		if (requestOptions.getServiceTier() != null) {
			builder.serviceTier(ResponseCreateParams.ServiceTier.of(requestOptions.getServiceTier()));
		}
		if (requestOptions.getPreviousResponseId() != null) {
			builder.previousResponseId(requestOptions.getPreviousResponseId());
		}
		if (requestOptions.getTruncation() != null) {
			builder.truncation(ResponseCreateParams.Truncation.of(requestOptions.getTruncation()));
		}

		// Add tools
		List<Tool> tools = new ArrayList<>();

		// Add built-in tools
		if (!CollectionUtils.isEmpty(requestOptions.getBuiltInTools())) {
			for (OpenAiSdkResponsesOptions.BuiltInTool builtInTool : requestOptions.getBuiltInTools()) {
				switch (builtInTool) {
					case WEB_SEARCH -> {
						WebSearchTool.Builder webSearchBuilder = WebSearchTool.builder()
							.type(WebSearchTool.Type.WEB_SEARCH);
						if (requestOptions.getWebSearchOptions() != null) {
							var wsOpts = requestOptions.getWebSearchOptions();
							if (wsOpts.searchContextSize() != null) {
								webSearchBuilder.searchContextSize(WebSearchTool.SearchContextSize
									.of(wsOpts.searchContextSize().name().toLowerCase()));
							}
							if (wsOpts.userLocation() != null) {
								var loc = wsOpts.userLocation();
								WebSearchTool.UserLocation.Builder locBuilder = WebSearchTool.UserLocation.builder();
								locBuilder.type(WebSearchTool.UserLocation.Type.of(loc.type()));
								if (loc.approximate() != null) {
									var approx = loc.approximate();
									locBuilder.putAdditionalProperty("city", JsonValue.from(approx.city()));
									locBuilder.putAdditionalProperty("country", JsonValue.from(approx.country()));
									locBuilder.putAdditionalProperty("region", JsonValue.from(approx.region()));
									locBuilder.putAdditionalProperty("timezone", JsonValue.from(approx.timezone()));
								}
								webSearchBuilder.userLocation(locBuilder.build());
							}
						}
						tools.add(Tool.ofWebSearch(webSearchBuilder.build()));
					}
					case FILE_SEARCH ->
						tools.add(Tool.ofFileSearch(com.openai.models.responses.FileSearchTool.builder().build()));
					case CODE_INTERPRETER -> tools.add(Tool.ofCodeInterpreter(Tool.CodeInterpreter.builder().build()));
				}
			}
		}

		// Add function tools from ToolCallingManager
		List<ToolDefinition> toolDefinitions = this.toolCallingManager.resolveToolDefinitions(requestOptions);
		if (!CollectionUtils.isEmpty(toolDefinitions)) {
			for (ToolDefinition toolDef : toolDefinitions) {
				FunctionTool.Builder functionToolBuilder = FunctionTool.builder()
					.name(toolDef.name())
					.description(toolDef.description())
					.strict(true);

				if (StringUtils.hasText(toolDef.inputSchema())) {
					try {
						ObjectMapper mapper = new ObjectMapper();
						@SuppressWarnings("unchecked")
						Map<String, Object> schemaMap = mapper.readValue(toolDef.inputSchema(), Map.class);
						FunctionTool.Parameters.Builder paramsBuilder = FunctionTool.Parameters.builder();
						schemaMap
							.forEach((key, value) -> paramsBuilder.putAdditionalProperty(key, JsonValue.from(value)));
						functionToolBuilder.parameters(paramsBuilder.build());
					}
					catch (Exception e) {
						logger.error("Failed to parse tool schema for {}", toolDef.name(), e);
					}
				}

				tools.add(Tool.ofFunction(functionToolBuilder.build()));
			}
		}

		if (!tools.isEmpty()) {
			builder.tools(tools);
		}

		return builder.build();
	}

	private ChatResponse processResponse(Response apiResponse, ChatResponse previousChatResponse) {
		List<AssistantMessage.ToolCall> toolCalls = new ArrayList<>();
		StringBuilder textContent = new StringBuilder();
		Map<String, Object> responseMetadata = new HashMap<>();
		List<String> webSearchResults = new ArrayList<>();
		List<String> reasoningTexts = new ArrayList<>();

		for (ResponseOutputItem item : apiResponse.output()) {
			if (item.isMessage()) {
				ResponseOutputMessage message = item.asMessage();
				for (ResponseOutputMessage.Content content : message.content()) {
					if (content.isOutputText()) {
						ResponseOutputText outputText = content.asOutputText();
						textContent.append(outputText.text());
						// Extract annotations (URL citations from web search)
						if (!outputText.annotations().isEmpty()) {
							responseMetadata.put("annotations", outputText.annotations().toString());
						}
					}
				}
			}
			else if (item.isFunctionCall()) {
				ResponseFunctionToolCall functionCall = item.asFunctionCall();
				toolCalls.add(new AssistantMessage.ToolCall(functionCall.callId(), "function", functionCall.name(),
						functionCall.arguments()));
			}
			else if (item.isWebSearchCall()) {
				ResponseFunctionWebSearch webSearch = item.asWebSearchCall();
				// Store as simple string to avoid nested Map issues with downstream
				// persistence (e.g. Neo4j properties)
				webSearchResults.add(webSearch.id() + ":" + webSearch.status().asString());
			}
			else if (item.isReasoning()) {
				ResponseReasoningItem reasoning = item.asReasoning();
				reasoningTexts.add(reasoning.id());
			}
		}

		if (!webSearchResults.isEmpty()) {
			responseMetadata.put("webSearchResults", String.join(",", webSearchResults));
		}
		if (!reasoningTexts.isEmpty()) {
			responseMetadata.put("reasoning", String.join(",", reasoningTexts));
		}

		AssistantMessage assistantMessage = AssistantMessage.builder()
			.content(textContent.toString())
			.toolCalls(toolCalls)
			.properties(responseMetadata)
			.build();

		String finishReason = apiResponse.status().map(ResponseStatus::asString).orElse("completed");
		Generation generation = new Generation(assistantMessage,
				ChatGenerationMetadata.builder().finishReason(finishReason).build());

		// Build usage
		Usage currentUsage = apiResponse.usage().<Usage>map(this::toUsage).orElse(new EmptyUsage());
		Usage accumulatedUsage = UsageCalculator.getCumulativeUsage(currentUsage, previousChatResponse);

		ChatResponseMetadata metadata = ChatResponseMetadata.builder()
			.id(apiResponse.id())
			.usage(accumulatedUsage)
			.model(apiResponse.model().asString())
			.build();

		return new ChatResponse(List.of(generation), metadata);
	}

	private DefaultUsage toUsage(ResponseUsage usage) {
		return new DefaultUsage(Math.toIntExact(usage.inputTokens()), Math.toIntExact(usage.outputTokens()),
				Math.toIntExact(usage.totalTokens()), usage);
	}

	private String mediaToText(Media media) {
		if (media.getData() instanceof byte[] bytes) {
			return "data:" + media.getMimeType().toString() + ";base64," + Base64.getEncoder().encodeToString(bytes);
		}
		else if (media.getData() instanceof String text) {
			return text;
		}
		throw new IllegalArgumentException("Unsupported media data type: " + media.getData().getClass());
	}

	@Override
	public ChatOptions getDefaultOptions() {
		return this.options.copy();
	}

	public void setObservationConvention(ChatModelObservationConvention observationConvention) {
		Assert.notNull(observationConvention, "observationConvention cannot be null");
		this.observationConvention = observationConvention;
	}

	public static Builder builder() {
		return new Builder();
	}

	public static final class Builder {

		private OpenAIClient openAiClient;

		private OpenAIClientAsync openAiClientAsync;

		private OpenAiSdkResponsesOptions defaultOptions = OpenAiSdkResponsesOptions.builder()
			.model(OpenAiSdkResponsesOptions.DEFAULT_MODEL)
			.build();

		private ToolCallingManager toolCallingManager;

		private ToolExecutionEligibilityPredicate toolExecutionEligibilityPredicate = new DefaultToolExecutionEligibilityPredicate();

		private ObservationRegistry observationRegistry = ObservationRegistry.NOOP;

		private Builder() {
		}

		public Builder openAiClient(OpenAIClient openAiClient) {
			this.openAiClient = openAiClient;
			return this;
		}

		public Builder openAiClientAsync(OpenAIClientAsync openAiClientAsync) {
			this.openAiClientAsync = openAiClientAsync;
			return this;
		}

		public Builder defaultOptions(OpenAiSdkResponsesOptions defaultOptions) {
			this.defaultOptions = defaultOptions;
			return this;
		}

		public Builder toolCallingManager(ToolCallingManager toolCallingManager) {
			this.toolCallingManager = toolCallingManager;
			return this;
		}

		public Builder toolExecutionEligibilityPredicate(
				ToolExecutionEligibilityPredicate toolExecutionEligibilityPredicate) {
			this.toolExecutionEligibilityPredicate = toolExecutionEligibilityPredicate;
			return this;
		}

		public Builder observationRegistry(ObservationRegistry observationRegistry) {
			this.observationRegistry = observationRegistry;
			return this;
		}

		public OpenAiSdkResponsesModel build() {
			return new OpenAiSdkResponsesModel(this.openAiClient, this.openAiClientAsync, this.defaultOptions,
					this.toolCallingManager != null ? this.toolCallingManager : DEFAULT_TOOL_CALLING_MANAGER,
					this.observationRegistry, this.toolExecutionEligibilityPredicate);
		}

	}

}
