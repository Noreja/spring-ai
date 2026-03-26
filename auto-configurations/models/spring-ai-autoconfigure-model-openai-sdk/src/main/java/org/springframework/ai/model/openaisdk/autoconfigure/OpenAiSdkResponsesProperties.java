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

package org.springframework.ai.model.openaisdk.autoconfigure;

import org.springframework.ai.openaisdk.AbstractOpenAiSdkOptions;
import org.springframework.ai.openaisdk.OpenAiSdkResponsesOptions;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.NestedConfigurationProperty;

@ConfigurationProperties(OpenAiSdkResponsesProperties.CONFIG_PREFIX)
public class OpenAiSdkResponsesProperties extends AbstractOpenAiSdkOptions {

	public static final String CONFIG_PREFIX = "spring.ai.openai-sdk.responses";

	private boolean enabled = false;

	@NestedConfigurationProperty
	private final OpenAiSdkResponsesOptions options = OpenAiSdkResponsesOptions.builder()
		.model(OpenAiSdkResponsesOptions.DEFAULT_MODEL)
		.temperature(0.7)
		.build();

	public boolean isEnabled() {
		return this.enabled;
	}

	public void setEnabled(boolean enabled) {
		this.enabled = enabled;
	}

	public OpenAiSdkResponsesOptions getOptions() {
		return this.options;
	}

}
