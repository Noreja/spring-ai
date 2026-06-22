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

import java.util.ArrayList;
import java.util.Base64;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicReference;

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
import com.openai.models.responses.ResponseOutputItemAddedEvent;
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
import org.jspecify.annotations.Nullable;
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
import org.springframework.ai.model.tool.ToolCallingChatOptions;
import org.springframework.ai.model.tool.ToolCallingManager;
import org.springframework.ai.model.tool.ToolExecutionResult;
import org.springframework.ai.model.tool.internal.ToolCallReactiveContextHolder;
import org.springframework.ai.observation.conventions.AiProvider;
import org.springframework.ai.openai.setup.OpenAiSetup;
import org.springframework.ai.support.UsageCalculator;
import org.springframework.ai.tool.definition.ToolDefinition;
import org.springframework.util.Assert;
import org.springframework.util.CollectionUtils;
import org.springframework.util.StringUtils;

/**
 * Chat Model implementation using the OpenAI Responses API via the official OpenAI Java
 * SDK. The Responses endpoint supports built-in tools like {@code web_search},
 * {@code file_search}, and {@code code_interpreter}, which are not available through the
 * Chat Completions API used by {@link OpenAiChatModel}.
 *
 * <p>
 * Reasoning content (the Responses {@code reasoning} item summary, and reasoning/summary
 * stream deltas) is surfaced as a separate {@link Generation} whose
 * {@link AssistantMessage} carries {@code properties["reasoning"] = true}, consistent
 * with reasoning surfacing in {@link OpenAiChatModel} and {@code AnthropicChatModel}.
 *
 * <p>
 * Unlike {@link OpenAiChatModel} (whose internal tool execution is deprecated in favour
 * of {@code ToolCallingAdvisor}), this model retains a self-contained tool-execution loop
 * so built-in and function tools work when the model is invoked directly.
 *
 * @author Noreja
 */
public class OpenAiResponsesModel implements ChatModel {

	private static final ChatModelObservationConvention DEFAULT_OBSERVATION_CONVENTION = new DefaultChatModelObservationConvention();

	private static final ToolCallingManager DEFAULT_TOOL_CALLING_MANAGER = ToolCallingManager.builder().build();

	private final Logger logger = LoggerFactory.getLogger(OpenAiResponsesModel.class);

	private final OpenAIClient openAiClient;

	private final OpenAIClientAsync openAiClientAsync;

	private final OpenAiResponsesOptions options;

	private final ObservationRegistry observationRegistry;

	private final ToolCallingManager toolCallingManager;

	private ChatModelObservationConvention observationConvention = DEFAULT_OBSERVATION_CONVENTION;

	public OpenAiResponsesModel() {
		this(null, null, null, null, null);
	}

	public OpenAiResponsesModel(OpenAiResponsesOptions options) {
		this(null, null, options, null, null);
	}

	public OpenAiResponsesModel(OpenAiResponsesOptions options, ToolCallingManager toolCallingManager,
			ObservationRegistry observationRegistry) {
		this(null, null, options, toolCallingManager, observationRegistry);
	}

	public OpenAiResponsesModel(@Nullable OpenAIClient openAiClient, @Nullable OpenAIClientAsync openAiClientAsync,
			@Nullable OpenAiResponsesOptions options, @Nullable ToolCallingManager toolCallingManager,
			@Nullable ObservationRegistry observationRegistry) {

		if (options == null) {
			this.options = OpenAiResponsesOptions.builder().model(OpenAiResponsesOptions.DEFAULT_MODEL).build();
		}
		else {
			this.options = options;
		}
		this.observationRegistry = Objects.requireNonNullElse(observationRegistry, ObservationRegistry.NOOP);

		this.openAiClient = openAiClient != null ? openAiClient
				: OpenAiSetup.setupSyncClient(this.options.getBaseUrl(), this.options.getApiKey(),
						this.options.getCredential(), this.options.getMicrosoftDeploymentName(),
						this.options.getMicrosoftFoundryServiceVersion(), this.options.getOrganizationId(),
						this.options.isMicrosoftFoundry(), this.options.isGitHubModels(), this.options.getModel(),
						this.options.getTimeout(), this.options.getMaxRetries(), this.options.getProxy(),
						this.options.getCustomHeaders(), this.observationRegistry, null, List.of());

		this.openAiClientAsync = openAiClientAsync != null ? openAiClientAsync
				: OpenAiSetup.setupAsyncClient(this.options.getBaseUrl(), this.options.getApiKey(),
						this.options.getCredential(), this.options.getMicrosoftDeploymentName(),
						this.options.getMicrosoftFoundryServiceVersion(), this.options.getOrganizationId(),
						this.options.isMicrosoftFoundry(), this.options.isGitHubModels(), this.options.getModel(),
						this.options.getTimeout(), this.options.getMaxRetries(), this.options.getProxy(),
						this.options.getCustomHeaders(), this.observationRegistry, null, List.of());

		this.toolCallingManager = toolCallingManager != null ? toolCallingManager : DEFAULT_TOOL_CALLING_MANAGER;
	}

	@Override
	public OpenAiResponsesOptions getOptions() {
		return this.options;
	}

	/**
	 * Merge the model's default options with any runtime options from the prompt, always
	 * yielding an {@link OpenAiResponsesOptions}. Spring AI 2.0 removed the
	 * {@code ChatModel.buildRequestPrompt} default method, so this is implemented
	 * locally. The merge (rather than 2.0's plain "use prompt options as-is") preserves
	 * connection-level defaults such as the API key when callers pass partial runtime
	 * options to this directly-invoked model.
	 */
	private Prompt buildRequestPrompt(Prompt prompt) {
		OpenAiResponsesOptions.Builder builder = this.options.mutate();
		if (prompt.getOptions() != null) {
			builder.combineWith(prompt.getOptions().mutate());
		}
		OpenAiResponsesOptions requestOptions = builder.build();
		ToolCallingChatOptions.validateToolCallbacks(requestOptions.getToolCallbacks());
		return new Prompt(prompt.getInstructions(), requestOptions);
	}

	@Override
	public ChatResponse call(Prompt prompt) {
		Prompt requestPrompt = buildRequestPrompt(prompt);
		return this.internalCall(requestPrompt, null);
	}

	public ChatResponse internalCall(Prompt prompt, @Nullable ChatResponse previousChatResponse) {

		ResponseCreateParams request = createRequest(prompt);

		ChatModelObservationContext observationContext = ChatModelObservationContext.builder()
			.prompt(prompt)
			.provider(AiProvider.OPENAI.value())
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

		if (isToolExecutionRequired(prompt, response)) {
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

	public Flux<ChatResponse> internalStream(Prompt prompt, @Nullable ChatResponse previousChatResponse) {
		return Flux.deferContextual(contextView -> {
			ResponseCreateParams request = createRequest(prompt);

			final ChatModelObservationContext observationContext = ChatModelObservationContext.builder()
				.prompt(prompt)
				.provider(AiProvider.OPENAI.value())
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
							AssistantMessage assistantMessage = AssistantMessage.builder()
								.content(delta)
								.properties(Map.of("text", Boolean.TRUE))
								.build();
							Generation generation = new Generation(assistantMessage,
									ChatGenerationMetadata.builder().build());
							sink.next(new ChatResponse(List.of(generation)));
						}
						else if (event.isReasoningTextDelta() || event.isReasoningSummaryTextDelta()) {
							String delta = event.isReasoningTextDelta() ? event.asReasoningTextDelta().delta()
									: event.asReasoningSummaryTextDelta().delta();
							AssistantMessage reasoningMessage = AssistantMessage.builder()
								.content(delta)
								.properties(Map.of("reasoning", Boolean.TRUE))
								.build();
							Generation generation = new Generation(reasoningMessage,
									ChatGenerationMetadata.builder().build());
							sink.next(new ChatResponse(List.of(generation)));
						}
						else if (event.isFunctionCallArgumentsDelta()) {
							ResponseFunctionCallArgumentsDeltaEvent argsDelta = event.asFunctionCallArgumentsDelta();
							String itemId = argsDelta.itemId();
							functionCallArgs.computeIfAbsent(itemId, k -> new StringBuilder())
								.append(argsDelta.delta());
						}
						else if (event.isOutputItemAdded()) {
							ResponseOutputItemAddedEvent itemAdded = event.asOutputItemAdded();
							ResponseOutputItem item = itemAdded.item();
							if (item.isFunctionCall()) {
								ResponseFunctionToolCall fc = item.asFunctionCall();
								functionCallNames.put(fc.id().orElse(fc.callId()), fc.name());
								functionCallIds.put(fc.id().orElse(fc.callId()), fc.callId());
							}
						}
						else if (event.isFunctionCallArgumentsDone()) {
							ResponseFunctionCallArgumentsDoneEvent argsDone = event.asFunctionCallArgumentsDone();
							String itemId = argsDone.itemId();
							argsDone._name().asKnown().ifPresent(name -> functionCallNames.put(itemId, name));
							functionCallIds.putIfAbsent(itemId, argsDone.itemId());
						}
						else if (event.isCompleted()) {
							ResponseCompletedEvent completed = event.asCompleted();
							Response apiResponse = completed.response();
							// Build a final response with only tool calls and
							// usage/metadata — text was already streamed via deltas
							ChatResponse fullResponse = processResponse(apiResponse, previousChatResponse);
							observationContext.setResponse(fullResponse);

							// Extract tool calls from the full response
							Generation fullResult = fullResponse.getResult();
							AssistantMessage fullMsg = fullResult != null ? fullResult.getOutput() : null;
							List<AssistantMessage.ToolCall> completedToolCalls = fullMsg != null
									? fullMsg.getToolCalls() : List.of();
							Map<String, Object> metaProps = fullMsg != null && fullMsg.getMetadata() != null
									? fullMsg.getMetadata() : Map.of();

							// Emit a metadata-only response (empty text) with usage and
							// any tool calls
							AssistantMessage metaMessage = AssistantMessage.builder()
								.content("")
								.toolCalls(completedToolCalls)
								.properties(metaProps)
								.build();
							Generation metaGen = new Generation(metaMessage, fullResult != null
									? fullResult.getMetadata() : ChatGenerationMetadata.builder().build());
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

			// Stream text deltas through immediately. The final completed
			// event (last element) may contain tool calls that require
			// execution and a recursive internalStream call.
			AtomicReference<ChatResponse> lastResponse = new AtomicReference<>();
			return flux.doOnNext(lastResponse::set).concatWith(Flux.defer(() -> {
				ChatResponse completed = lastResponse.get();
				if (completed == null) {
					return Flux.empty();
				}
				if (!isToolExecutionRequired(prompt, completed)) {
					return Flux.empty();
				}
				// Tool calls detected — execute and continue the
				// conversation
				return Flux.deferContextual(ctx -> {
					ToolExecutionResult toolExecutionResult;
					try {
						ToolCallReactiveContextHolder.setContext(ctx);
						toolExecutionResult = this.toolCallingManager.executeToolCalls(prompt, completed);
					}
					finally {
						ToolCallReactiveContextHolder.clearContext();
					}
					if (toolExecutionResult.returnDirect()) {
						return Flux.just(ChatResponse.builder()
							.from(completed)
							.generations(ToolExecutionResult.buildGenerations(toolExecutionResult))
							.build());
					}
					return this.internalStream(
							new Prompt(toolExecutionResult.conversationHistory(), prompt.getOptions()), completed);
				}).subscribeOn(Schedulers.boundedElastic());
			})).doOnError(observation::error).doFinally(s -> observation.stop());
		});
	}

	/**
	 * Whether internal tool execution should run for the given response. Replaces the
	 * removed {@code ToolExecutionEligibilityPredicate}: tool execution is required when
	 * the response carries tool calls and internal execution is not explicitly disabled.
	 */
	private boolean isToolExecutionRequired(Prompt prompt, @Nullable ChatResponse response) {
		if (response == null || !response.hasToolCalls()) {
			return false;
		}
		if (prompt.getOptions() instanceof OpenAiResponsesOptions responsesOptions
				&& responsesOptions.getInternalToolExecutionEnabled() != null) {
			return responsesOptions.getInternalToolExecutionEnabled();
		}
		return true;
	}

	ResponseCreateParams createRequest(Prompt prompt) {

		OpenAiResponsesOptions requestOptions = (OpenAiResponsesOptions) prompt.getOptions();
		Assert.state(requestOptions != null, "ChatOptions must not be null");

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
					String text = message.getText();
					contentParts.add(ResponseInputContent
						.ofInputText(ResponseInputText.builder().text(text != null ? text : "").build()));
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
		if (requestOptions.getReasoningEffort() != null || requestOptions.getReasoningSummary() != null) {
			Reasoning.Builder reasoningBuilder = Reasoning.builder();
			if (requestOptions.getReasoningEffort() != null) {
				reasoningBuilder.effort(ReasoningEffort.of(requestOptions.getReasoningEffort().toLowerCase()));
			}
			if (requestOptions.getReasoningSummary() != null) {
				reasoningBuilder.summary(Reasoning.Summary.of(requestOptions.getReasoningSummary().toLowerCase()));
			}
			builder.reasoning(reasoningBuilder.build());
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
			for (OpenAiResponsesOptions.BuiltInTool builtInTool : requestOptions.getBuiltInTools()) {
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
			boolean strictTools = requestOptions.getStrictTools() == null || requestOptions.getStrictTools();
			for (ToolDefinition toolDef : toolDefinitions) {
				FunctionTool.Builder functionToolBuilder = FunctionTool.builder()
					.name(toolDef.name())
					.description(toolDef.description())
					.strict(strictTools);

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

	private ChatResponse processResponse(Response apiResponse, @Nullable ChatResponse previousChatResponse) {
		List<AssistantMessage.ToolCall> toolCalls = new ArrayList<>();
		StringBuilder textContent = new StringBuilder();
		Map<String, Object> responseMetadata = new HashMap<>();
		List<String> webSearchResults = new ArrayList<>();
		StringBuilder reasoningText = new StringBuilder();

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
				for (ResponseReasoningItem.Summary summary : reasoning.summary()) {
					if (reasoningText.length() > 0) {
						reasoningText.append("\n");
					}
					reasoningText.append(summary.text());
				}
			}
		}

		if (!webSearchResults.isEmpty()) {
			responseMetadata.put("webSearchResults", String.join(",", webSearchResults));
		}

		String finishReason = apiResponse.status().map(ResponseStatus::asString).orElse("completed");
		List<Generation> generations = new ArrayList<>(2);
		if (reasoningText.length() > 0) {
			AssistantMessage reasoningMessage = AssistantMessage.builder()
				.content(reasoningText.toString())
				.properties(Map.of("reasoning", Boolean.TRUE))
				.build();
			generations.add(new Generation(reasoningMessage,
					ChatGenerationMetadata.builder().finishReason(finishReason).build()));
		}

		AssistantMessage assistantMessage = AssistantMessage.builder()
			.content(textContent.toString())
			.toolCalls(toolCalls)
			.properties(responseMetadata)
			.build();
		generations
			.add(new Generation(assistantMessage, ChatGenerationMetadata.builder().finishReason(finishReason).build()));

		// Build usage
		Usage currentUsage = apiResponse.usage().<Usage>map(this::toUsage).orElse(new EmptyUsage());
		Usage accumulatedUsage = UsageCalculator.getCumulativeUsage(currentUsage, previousChatResponse);

		ChatResponseMetadata metadata = ChatResponseMetadata.builder()
			.id(apiResponse.id())
			.usage(accumulatedUsage)
			// ResponsesModel is a union (string | ChatModel enum | ResponsesOnlyModel);
			// asString()
			// throws when the response model resolves to a non-string variant. Use the
			// safe accessor.
			.model(apiResponse.model().string().orElseGet(() -> apiResponse.model().toString()))
			.build();

		return new ChatResponse(generations, metadata);
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

	public void setObservationConvention(ChatModelObservationConvention observationConvention) {
		Assert.notNull(observationConvention, "observationConvention cannot be null");
		this.observationConvention = observationConvention;
	}

	public static Builder builder() {
		return new Builder();
	}

	public static final class Builder {

		private @Nullable OpenAIClient openAiClient;

		private @Nullable OpenAIClientAsync openAiClientAsync;

		private OpenAiResponsesOptions defaultOptions = OpenAiResponsesOptions.builder()
			.model(OpenAiResponsesOptions.DEFAULT_MODEL)
			.build();

		private @Nullable ToolCallingManager toolCallingManager;

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

		public Builder defaultOptions(OpenAiResponsesOptions defaultOptions) {
			this.defaultOptions = defaultOptions;
			return this;
		}

		public Builder toolCallingManager(ToolCallingManager toolCallingManager) {
			this.toolCallingManager = toolCallingManager;
			return this;
		}

		public Builder observationRegistry(ObservationRegistry observationRegistry) {
			this.observationRegistry = observationRegistry;
			return this;
		}

		public OpenAiResponsesModel build() {
			return new OpenAiResponsesModel(this.openAiClient, this.openAiClientAsync, this.defaultOptions,
					this.toolCallingManager != null ? this.toolCallingManager : DEFAULT_TOOL_CALLING_MANAGER,
					this.observationRegistry);
		}

	}

}
