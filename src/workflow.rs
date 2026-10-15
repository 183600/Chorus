use crate::config::{Config, ModelConfig, WorkflowModelTarget, WorkflowPlan, WorkflowWorker};
use crate::llm::{parse_temperature_from_response, ChatMessage, LLMClient};
use anyhow::{anyhow, Result};
use async_recursion::async_recursion;
use futures::stream::{self, StreamExt as _};
use serde::{Deserialize, Serialize, Serializer};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::Arc;
use tokio::sync::{mpsc::UnboundedSender, RwLock};
use url::Url;

const DEFAULT_TEMPERATURE: f32 = 1.4;
const DEFAULT_WORKER_CONCURRENCY: usize = 4;

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
struct LlmClientCacheKey {
    api_base: String,
    api_key: String,
    timeout_secs: u64,
}

impl LlmClientCacheKey {
    fn new(api_base: &str, api_key: &str, timeout_secs: u64) -> Self {
        Self {
            api_base: api_base.to_string(),
            api_key: api_key.to_string(),
            timeout_secs,
        }
    }
}

pub type StreamCallback = UnboundedSender<String>;

struct SelectedChoice {
    index: usize,
    worker_name: String,
    response: String,
    reasoning: Option<String>,
}

#[derive(Debug, Clone)]
struct ParsedSelection {
    index: usize,
    label: Option<String>,
    reasoning: Option<String>,
    selected_response: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowResult {
    pub final_response: String,
    pub execution_details: WorkflowExecutionDetails,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowExecutionDetails {
    pub analyzer: AnalyzerDetails,
    pub workers: Vec<WorkerDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selector: Option<SelectorDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub synthesizer: Option<SynthesizerDetails>,
}

fn serialize_f32_as_f64<S>(value: &f32, serializer: S) -> std::result::Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_f64(*value as f64)
}

fn serialize_optional_f32_as_f64<S>(
    value: &Option<f32>,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error>
where
    S: Serializer,
{
    match value {
        Some(inner) => serializer.serialize_some(&(*inner as f64)),
        None => serializer.serialize_none(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzerDetails {
    pub model: String,
    #[serde(serialize_with = "serialize_f32_as_f64")]
    pub temperature: f32,
    pub auto_temperature: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerDetails {
    pub name: String,
    #[serde(serialize_with = "serialize_optional_f32_as_f64")]
    pub temperature: Option<f32>,
    pub response: Option<String>,
    pub success: bool,
    pub error: Option<String>,
    pub nested: Option<Box<WorkflowExecutionDetails>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectorDetails {
    pub model: String,
    #[serde(serialize_with = "serialize_f32_as_f64")]
    pub temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_worker: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_response: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_output: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesizerDetails {
    pub model: String,
    #[serde(serialize_with = "serialize_f32_as_f64")]
    pub temperature: f32,
}

pub struct WorkflowEngine {
    config: Arc<Config>,
    model_configs: HashMap<String, usize>,
    api_timeouts: HashMap<String, crate::config::TimeoutConfig>,
    llm_clients: RwLock<Vec<(LlmClientCacheKey, Arc<LLMClient>)>>,
}

impl WorkflowEngine {
    #[cfg(test)]
    pub fn new(config: Config) -> Self {
        Self::from_shared(Arc::new(config))
    }

    pub fn from_shared(config: Arc<Config>) -> Self {
        let mut model_configs = HashMap::with_capacity(config.models.len());
        let mut api_timeouts = HashMap::with_capacity(config.models.len());
        for (index, model) in config.models.iter().enumerate() {
            model_configs.insert(model.name.clone(), index);
            api_timeouts
                .entry(model.api_base.clone())
                .or_insert_with(|| Self::resolve_timeouts(&config, &model.api_base));
        }
        Self {
            config,
            model_configs,
            api_timeouts,
            llm_clients: RwLock::new(Vec::new()),
        }
    }

    fn resolve_timeouts(config: &Config, api_base: &str) -> crate::config::TimeoutConfig {
        match Url::parse(api_base) {
            Ok(url) => config.effective_timeouts_for_domain(url.host_str()),
            Err(_) => config.effective_timeouts_for_domain(None),
        }
    }

    pub async fn process(&self, prompt: String) -> Result<String> {
        self.run_plan(&self.config.workflow_integration, &prompt, 0, None)
            .await
    }

    pub async fn process_with_details(&self, prompt: String) -> Result<WorkflowResult> {
        self.run_plan_with_details(&self.config.workflow_integration, &prompt, 0, None)
            .await
    }

    pub async fn process_with_stream(
        &self,
        prompt: String,
        stream: Option<StreamCallback>,
    ) -> Result<String> {
        self.run_plan(&self.config.workflow_integration, &prompt, 0, stream)
            .await
    }

    pub async fn process_with_details_stream(
        &self,
        prompt: String,
        stream: Option<StreamCallback>,
    ) -> Result<WorkflowResult> {
        self.run_plan_with_details(&self.config.workflow_integration, &prompt, 0, stream)
            .await
    }

    fn find_cached_client(
        clients: &[(LlmClientCacheKey, Arc<LLMClient>)],
        api_base: &str,
        api_key: &str,
        timeout_secs: u64,
    ) -> Option<Arc<LLMClient>> {
        clients
            .iter()
            .find(|(key, _)| {
                key.api_base == api_base
                    && key.api_key == api_key
                    && key.timeout_secs == timeout_secs
            })
            .map(|(_, client)| Arc::clone(client))
    }

    async fn get_llm_client(
        &self,
        api_base: &str,
        api_key: &str,
        timeout_secs: u64,
    ) -> Result<Arc<LLMClient>> {
        {
            let clients = self.llm_clients.read().await;
            if let Some(client) = Self::find_cached_client(&clients, api_base, api_key, timeout_secs)
            {
                return Ok(client);
            }
        }

        let new_client = Arc::new(LLMClient::new(
            api_base.to_string(),
            api_key.to_string(),
            timeout_secs,
        )?);

        let mut clients = self.llm_clients.write().await;
        if let Some(client) = Self::find_cached_client(&clients, api_base, api_key, timeout_secs) {
            return Ok(client);
        }
        clients.push((
            LlmClientCacheKey::new(api_base, api_key, timeout_secs),
            Arc::clone(&new_client),
        ));
        Ok(new_client)
    }

    fn effective_timeouts_for_api_base(&self, api_base: &str) -> crate::config::TimeoutConfig {
        match self.api_timeouts.get(api_base) {
            Some(timeouts) => timeouts.clone(),
            None => Self::resolve_timeouts(&self.config, api_base),
        }
    }

    #[async_recursion]
    async fn run_plan_with_details(
        &self,
        plan: &WorkflowPlan,
        prompt: &str,
        depth: usize,
        stream: Option<StreamCallback>,
    ) -> Result<WorkflowResult> {
        if depth == 0 {
            tracing::info!("Starting workflow processing with details");
        } else {
            tracing::debug!(
                "Starting nested workflow at depth {} ({})",
                depth,
                plan.label()
            );
        }

        let target = &plan.analyzer;
        let model_config = self.lookup_model(&target.model)?;

        let auto_temperature = target
            .auto_temperature
            .or(model_config.auto_temperature)
            .unwrap_or(false);

        let temperature = self
            .resolve_analyzer_temperature(plan, model_config, prompt, depth)
            .await?;

        let analyzer_details = AnalyzerDetails {
            model: target.model.clone(),
            temperature,
            auto_temperature,
        };

        if depth == 0 {
            tracing::info!("Step 1 completed - Temperature: {}", temperature);
        } else {
            tracing::debug!(
                "Nested workflow depth {} analyzer temperature resolved to {}",
                depth,
                temperature
            );
        }

        let worker_details = self
            .run_workers_with_details(plan, prompt, temperature, auto_temperature, depth)
            .await?;

        if depth == 0 {
            tracing::info!(
                "Step 2 completed - Collected {} worker responses",
                worker_details.iter().filter(|w| w.success).count()
            );
        } else {
            tracing::debug!(
                "Nested workflow depth {} collected {} worker responses",
                depth,
                worker_details.iter().filter(|w| w.success).count()
            );
        }

        let worker_responses: Vec<(&str, &str)> = worker_details
            .iter()
            .filter_map(|w| {
                if w.success {
                    w.response.as_deref().map(|r| (w.name.as_str(), r))
                } else {
                    None
                }
            })
            .collect();

        let (selector_details, selected_choice) =
            if let Some(selector_target) = plan.selector.as_ref() {
                let (details, choice) = self
                    .execute_selector(selector_target, prompt, &worker_responses, depth)
                    .await;
                (Some(details), choice)
            } else {
                (None, None)
            };

        let mut top_level_streamed = false;

        let (synthesizer_details, final_response) =
            if let Some(synthesizer_target) = plan.synthesizer.as_ref() {
                let synthesizer_model_config = self.lookup_model(&synthesizer_target.model)?;
                let synthesizer_temperature = self.resolve_synthesizer_temperature(
                    synthesizer_target,
                    synthesizer_model_config,
                    depth,
                );

                let synthesizer_details = SynthesizerDetails {
                    model: synthesizer_target.model.clone(),
                    temperature: synthesizer_temperature,
                };

                let stream_for_synth = if depth == 0 { stream.clone() } else { None };

                let (final_response, streamed) = self
                    .call_synthesizer(
                        synthesizer_target,
                        prompt,
                        &worker_responses,
                        selected_choice.as_ref(),
                        depth,
                        stream_for_synth,
                    )
                    .await?;

                if depth == 0 {
                    top_level_streamed = streamed;
                }

                (Some(synthesizer_details), final_response)
            } else {
                let final_response = self.resolve_final_response_without_synthesizer(
                    plan,
                    &worker_details,
                    selector_details.as_ref(),
                    selected_choice.as_ref(),
                    depth,
                )?;

                (None, final_response)
            };

        if depth == 0 {
            if let Some(sender) = stream.as_ref() {
                if !top_level_streamed {
                    let _ = sender.send(final_response.clone());
                }
            }
            tracing::info!("Step 3 completed - Final response generated");
        } else {
            tracing::debug!("Nested workflow depth {} produced final response", depth);
        }

        Ok(WorkflowResult {
            final_response,
            execution_details: WorkflowExecutionDetails {
                analyzer: analyzer_details,
                workers: worker_details,
                selector: selector_details,
                synthesizer: synthesizer_details,
            },
        })
    }

    async fn run_plan(
        &self,
        plan: &WorkflowPlan,
        prompt: &str,
        depth: usize,
        stream: Option<StreamCallback>,
    ) -> Result<String> {
        let result = self
            .run_plan_with_details(plan, prompt, depth, stream)
            .await?;
        Ok(result.final_response)
    }

    fn resolve_final_response_without_synthesizer(
        &self,
        plan: &WorkflowPlan,
        worker_details: &[WorkerDetails],
        selector_details: Option<&SelectorDetails>,
        selected_choice: Option<&SelectedChoice>,
        depth: usize,
    ) -> Result<String> {
        let _ = self;
        let plan_label = plan.label();

        if let Some(details) = selector_details {
            if let Some(selected_response) = details.selected_response.clone() {
                if depth == 0 {
                    tracing::info!(
                        plan = %plan_label,
                        "Using selector recommendation as final response"
                    );
                } else {
                    tracing::debug!(
                        plan = %plan_label,
                        depth,
                        "Using selector recommendation as final response"
                    );
                }
                return Ok(selected_response);
            }

            if let Some(choice) = selected_choice {
                if depth == 0 {
                    tracing::info!(
                        plan = %plan_label,
                        worker = %choice.worker_name,
                        "Selector chose worker response; using it as final output"
                    );
                } else {
                    tracing::debug!(
                        plan = %plan_label,
                        depth,
                        worker = %choice.worker_name,
                        "Selector chose worker response; using it as final output"
                    );
                }
                return Ok(choice.response.clone());
            }

            tracing::warn!(
                plan = %plan_label,
                depth,
                "Selector failed to provide a choice; falling back to worker responses"
            );
        } else {
            tracing::debug!(
                plan = %plan_label,
                depth,
                "No selector configured; using worker responses directly"
            );
        }

        let fallback = worker_details.iter().find_map(|worker| {
            if worker.success {
                worker.response.clone()
            } else {
                None
            }
        });

        if let Some(response) = fallback {
            if depth == 0 {
                tracing::info!(
                    plan = %plan_label,
                    "Using first successful worker response as final output"
                );
            } else {
                tracing::debug!(
                    plan = %plan_label,
                    depth,
                    "Using first successful worker response as final output"
                );
            }
            Ok(response)
        } else {
            Err(anyhow!(
                "Workflow plan {} could not determine a final response because the selector failed and no worker responses were available.",
                plan_label
            ))
        }
    }

    async fn resolve_analyzer_temperature(
        &self,
        plan: &WorkflowPlan,
        model_config: &ModelConfig,
        prompt: &str,
        depth: usize,
    ) -> Result<f32> {
        let target = &plan.analyzer;

        if let Some(explicit) = target.temperature.or(model_config.temperature) {
            if depth == 0 {
                tracing::info!("Using configured analyzer temperature: {}", explicit);
            } else {
                tracing::debug!(
                    "Depth {} using configured analyzer temperature {}",
                    depth,
                    explicit
                );
            }
            return Ok(explicit);
        }

        let auto = target
            .auto_temperature
            .or(model_config.auto_temperature)
            .unwrap_or(false);

        if !auto {
            if depth == 0 {
                tracing::info!(
                    "Analyzer auto temperature disabled, using default: {}",
                    DEFAULT_TEMPERATURE
                );
            } else {
                tracing::debug!(
                    "Depth {} analyzer auto temperature disabled, using default {}",
                    depth,
                    DEFAULT_TEMPERATURE
                );
            }
            return Ok(DEFAULT_TEMPERATURE);
        }

        if depth == 0 {
            tracing::info!(
                "Auto temperature enabled for analyzer {} at depth {}, analyzing prompt",
                target.model,
                depth
            );
        } else {
            tracing::debug!(
                "Auto temperature enabled for analyzer {} at depth {}, analyzing prompt",
                target.model,
                depth
            );
        }

        let timeouts = self.effective_timeouts_for_api_base(&model_config.api_base);
        let client = self
            .get_llm_client(
                &model_config.api_base,
                &model_config.api_key,
                timeouts.analyzer_timeout_secs,
            )
            .await?;

        let analysis_prompt = format!(
            r#"请分析以下用户提示，并为其推荐一个合适的temperature参数（0.0-2.0之间的浮点数）。
Temperature越低（接近0），输出越确定和保守；temperature越高（接近2），输出越有创造性和随机性。

用户提示: {}

请只返回一个JSON对象，格式如下：
{{
    "temperature": 0.7,
    "reasoning": "简短说明为什么选择这个temperature值"
}}
"#,
            prompt
        );

        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: analysis_prompt,
        }];

        let response = client
            .chat_completion(&target.model, messages, Some(0.3))
            .await?;

        let temperature = parse_temperature_from_response(&response);
        tracing::debug!(
            "Analyzer {} produced temperature {} (depth {}), response: {}",
            target.model,
            temperature,
            depth,
            response
        );

        Ok(temperature)
    }

    #[allow(dead_code)]
    async fn run_workers(
        &self,
        plan: &WorkflowPlan,
        prompt: &str,
        base_temperature: f32,
        depth: usize,
    ) -> Result<Vec<(String, String)>> {
        let analyzer_target = &plan.analyzer;
        let analyzer_model_config = self.lookup_model(&analyzer_target.model)?;
        let analyzer_auto = analyzer_target
            .auto_temperature
            .or(analyzer_model_config.auto_temperature)
            .unwrap_or(false);

        let details = self
            .run_workers_with_details(plan, prompt, base_temperature, analyzer_auto, depth)
            .await?;
        let responses = details
            .into_iter()
            .filter_map(|worker| {
                if worker.success {
                    worker.response.map(|resp| (worker.name, resp))
                } else {
                    None
                }
            })
            .collect();
        Ok(responses)
    }

    #[async_recursion]
    async fn run_workers_with_details(
        &self,
        plan: &WorkflowPlan,
        prompt: &str,
        base_temperature: f32,
        analyzer_auto: bool,
        depth: usize,
    ) -> Result<Vec<WorkerDetails>> {
        let plan_label = plan.label();

        if plan.workers.is_empty() {
            tracing::error!(
                plan = %plan_label,
                depth,
                "Workflow plan is missing worker nodes"
            );
            return Err(anyhow!(
                "Workflow plan {} has no worker nodes configured at depth {}. Please define at least one worker in the workflow configuration.",
                plan_label,
                depth
            ));
        }

        let concurrency = self.worker_concurrency().min(plan.workers.len()).max(1);

        let worker_details: Vec<WorkerDetails> = if concurrency == 1 || plan.workers.len() <= 1 {
            let mut details = Vec::with_capacity(plan.workers.len());
            for worker in &plan.workers {
                details.push(
                    self.run_single_worker(worker, prompt, base_temperature, analyzer_auto, depth)
                        .await,
                );
            }
            details
        } else {
            let pending: Vec<_> = plan
                .workers
                .iter()
                .map(|worker| {
                    self.run_single_worker(worker, prompt, base_temperature, analyzer_auto, depth)
                })
                .collect();
            stream::iter(pending).buffered(concurrency).collect().await
        };

        if worker_details.iter().filter(|w| w.success).count() == 0 {
            let worker_errors: Vec<String> = worker_details
                .iter()
                .filter_map(|w| w.error.as_ref().map(|e| format!("{}: {}", w.name, e)))
                .collect();

            let mut message = format!(
                "All worker nodes failed at depth {} for plan {}",
                depth, plan_label
            );
            if !worker_errors.is_empty() {
                message.push_str(". Worker errors: ");
                message.push_str(&worker_errors.join(" | "));
            } else {
                let labels = plan.worker_labels();
                if !labels.is_empty() {
                    message.push_str(". Configured workers: ");
                    message.push_str(&labels.join(", "));
                }
            }
            return Err(anyhow!(message));
        }

        Ok(worker_details)
    }

    fn worker_concurrency(&self) -> usize {
        self.config
            .workflow
            .worker_concurrency
            .unwrap_or(DEFAULT_WORKER_CONCURRENCY)
            .max(1)
    }

    async fn run_single_worker(
        &self,
        worker: &WorkflowWorker,
        prompt: &str,
        base_temperature: f32,
        analyzer_auto: bool,
        depth: usize,
    ) -> WorkerDetails {
        match worker {
            WorkflowWorker::Model(target) => {
                if depth == 0 {
                    tracing::info!("Calling worker model: {}", target.model);
                } else {
                    tracing::debug!("Calling worker model {} at depth {}", target.model, depth);
                }

                let model_config = match self.lookup_model(&target.model) {
                    Ok(model_config) => model_config,
                    Err(err) => {
                        let err_display = err.to_string();
                        tracing::warn!(
                            worker = %target.model,
                            depth,
                            error = %err_display,
                            "Worker lookup failed"
                        );
                        return WorkerDetails {
                            name: target.model.clone(),
                            temperature: None,
                            response: None,
                            success: false,
                            error: Some(err_display),
                            nested: None,
                        };
                    }
                };

                let temperature = self.resolve_worker_temperature(
                    target,
                    model_config,
                    base_temperature,
                    analyzer_auto,
                );

                match self
                    .call_worker_model(target, model_config, temperature, prompt, depth)
                    .await
                {
                    Ok(response) => {
                        tracing::debug!("Worker {} succeeded at depth {}", target.model, depth);
                        WorkerDetails {
                            name: target.model.clone(),
                            temperature: Some(temperature),
                            response: Some(response),
                            success: true,
                            error: None,
                            nested: None,
                        }
                    }
                    Err(err) => {
                        let err_display = err.to_string();
                        tracing::warn!(
                            worker = %target.model,
                            depth,
                            error = %err_display,
                            "Worker call failed"
                        );
                        WorkerDetails {
                            name: target.model.clone(),
                            temperature: Some(temperature),
                            response: None,
                            success: false,
                            error: Some(err_display),
                            nested: None,
                        }
                    }
                }
            }
            WorkflowWorker::Workflow(sub_plan) => {
                let label = sub_plan.label();
                if depth == 0 {
                    tracing::info!("Executing nested workflow worker: {}", label);
                } else {
                    tracing::debug!(
                        "Executing nested workflow worker {} at depth {}",
                        label,
                        depth
                    );
                }

                match self
                    .run_plan_with_details(sub_plan, prompt, depth + 1, None)
                    .await
                {
                    Ok(result) => {
                        tracing::debug!("Nested workflow {} succeeded at depth {}", label, depth);
                        WorkerDetails {
                            name: label.clone(),
                            temperature: None,
                            response: Some(result.final_response),
                            success: true,
                            error: None,
                            nested: Some(Box::new(result.execution_details)),
                        }
                    }
                    Err(err) => {
                        let err_display = err.to_string();
                        tracing::warn!(
                            workflow = %label,
                            depth,
                            error = %err_display,
                            "Nested workflow failed"
                        );
                        WorkerDetails {
                            name: label,
                            temperature: None,
                            response: None,
                            success: false,
                            error: Some(err_display),
                            nested: None,
                        }
                    }
                }
            }
        }
    }

    async fn call_worker_model(
        &self,
        target: &WorkflowModelTarget,
        model_config: &ModelConfig,
        temperature: f32,
        prompt: &str,
        depth: usize,
    ) -> Result<String> {
        let timeouts = self.effective_timeouts_for_api_base(&model_config.api_base);
        let client = self
            .get_llm_client(
                &model_config.api_base,
                &model_config.api_key,
                timeouts.worker_timeout_secs,
            )
            .await?;

        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }];

        tracing::debug!(
            "Using temperature {} for worker {} at depth {}",
            temperature,
            target.model,
            depth
        );

        let response = client
            .chat_completion(&target.model, messages, Some(temperature))
            .await?;

        tracing::debug!(
            "Worker {} returned response at depth {}",
            target.model,
            depth
        );

        Ok(response)
    }

    async fn execute_selector(
        &self,
        target: &WorkflowModelTarget,
        original_prompt: &str,
        worker_responses: &[(&str, &str)],
        depth: usize,
    ) -> (SelectorDetails, Option<SelectedChoice>) {
        if worker_responses.is_empty() {
            tracing::warn!(
                selector = %target.model,
                depth,
                "Selector skipped because no worker responses are available"
            );
            return (
                SelectorDetails {
                    model: target.model.clone(),
                    temperature: DEFAULT_TEMPERATURE,
                    selected_index: None,
                    selected_worker: None,
                    selected_response: None,
                    reasoning: None,
                    success: false,
                    error: Some("No worker responses available for selector".to_string()),
                    raw_output: None,
                },
                None,
            );
        }

        let model_config = match self.lookup_model(&target.model) {
            Ok(config) => config,
            Err(err) => {
                let message = err.to_string();
                tracing::warn!(
                    selector = %target.model,
                    depth,
                    error = %message,
                    "Selector lookup failed"
                );
                return (
                    SelectorDetails {
                        model: target.model.clone(),
                        temperature: DEFAULT_TEMPERATURE,
                        selected_index: None,
                        selected_worker: None,
                        selected_response: None,
                        reasoning: None,
                        success: false,
                        error: Some(message),
                        raw_output: None,
                    },
                    None,
                );
            }
        };

        let temperature = self.resolve_selector_temperature(target, model_config, depth);

        let timeouts = self.effective_timeouts_for_api_base(&model_config.api_base);
        let client = match self
            .get_llm_client(
                &model_config.api_base,
                &model_config.api_key,
                timeouts.synthesizer_timeout_secs,
            )
            .await
        {
            Ok(client) => client,
            Err(err) => {
                let message = err.to_string();
                tracing::warn!(
                    selector = %target.model,
                    depth,
                    error = %message,
                    "Selector client construction failed"
                );
                return (
                    SelectorDetails {
                        model: target.model.clone(),
                        temperature,
                        selected_index: None,
                        selected_worker: None,
                        selected_response: None,
                        reasoning: None,
                        success: false,
                        error: Some(message),
                        raw_output: None,
                    },
                    None,
                );
            }
        };

        let prompt_capacity = original_prompt.len()
            + 160
            + worker_responses
                .iter()
                .map(|(label, response)| label.len() + response.len() + 24)
                .sum::<usize>();
        let mut selector_prompt = String::with_capacity(prompt_capacity);
        selector_prompt.push_str("原始用户问题：\n");
        selector_prompt.push_str(original_prompt);
        selector_prompt.push_str("\n\n以下是多个模型给出的回答，请选出质量最高的一条。\n\n");

        for (i, (label, response)) in worker_responses.iter().enumerate() {
            let _ = write!(
                selector_prompt,
                "【回答{}：{}】\n{}\n\n",
                i + 1,
                label,
                response
            );
        }

        selector_prompt.push_str(
            "请仅返回一个 JSON 对象，格式如下：\n\
            {\n  \"selected_index\": 1,\n  \"selected_worker\": \"模型名称\",\n  \"selected_response\": \"可选：直接粘贴所选回答\",\n  \"reasoning\": \"简要说明\"\n}\n\
            要求：\n\
            - selected_index 使用上面编号（从 1 开始）\n\
            - reasoning 简洁说明选择理由，如有不足请指出\n\
            - 如所有回答都存在问题，请选出相对最佳的一条并说明原因\n\
            只需输出 JSON 对象。\n",
        );

        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: selector_prompt,
        }];

        let raw_output = match client
            .chat_completion(&target.model, messages, Some(temperature))
            .await
        {
            Ok(content) => content,
            Err(err) => {
                let message = err.to_string();
                tracing::warn!(
                    selector = %target.model,
                    depth,
                    error = %message,
                    "Selector call failed"
                );
                return (
                    SelectorDetails {
                        model: target.model.clone(),
                        temperature,
                        selected_index: None,
                        selected_worker: None,
                        selected_response: None,
                        reasoning: None,
                        success: false,
                        error: Some(message),
                        raw_output: None,
                    },
                    None,
                );
            }
        };

        match parse_selector_choice(&raw_output, worker_responses.len()) {
            Ok(parsed) => {
                let worker_entry = &worker_responses[parsed.index - 1];
                let worker_name = worker_entry.0.to_string();
                let worker_response = worker_entry.1.to_string();
                let reasoning = parsed.reasoning;
                let selected_response = parsed
                    .selected_response
                    .unwrap_or_else(|| worker_response.clone());

                if depth == 0 {
                    tracing::info!(
                        selector = %target.model,
                        chosen_index = parsed.index,
                        chosen_worker = %worker_name,
                        "Selector chose worker response at top-level"
                    );
                } else {
                    tracing::debug!(
                        selector = %target.model,
                        depth,
                        chosen_index = parsed.index,
                        chosen_worker = %worker_name,
                        "Selector chose worker response"
                    );
                }

                let details = SelectorDetails {
                    model: target.model.clone(),
                    temperature,
                    selected_index: Some(parsed.index),
                    selected_worker: Some(worker_name.clone()),
                    selected_response: Some(selected_response),
                    reasoning: reasoning.clone(),
                    success: true,
                    error: None,
                    raw_output: Some(raw_output),
                };

                let choice = SelectedChoice {
                    index: parsed.index,
                    worker_name,
                    response: worker_response,
                    reasoning,
                };

                (details, Some(choice))
            }
            Err(err) => {
                let message = err.to_string();
                tracing::warn!(
                    selector = %target.model,
                    depth,
                    error = %message,
                    "Selector response parsing failed"
                );
                (
                    SelectorDetails {
                        model: target.model.clone(),
                        temperature,
                        selected_index: None,
                        selected_worker: None,
                        selected_response: None,
                        reasoning: None,
                        success: false,
                        error: Some(message),
                        raw_output: Some(raw_output),
                    },
                    None,
                )
            }
        }
    }

    async fn call_synthesizer(
        &self,
        target: &WorkflowModelTarget,
        original_prompt: &str,
        worker_responses: &[(&str, &str)],
        selected_choice: Option<&SelectedChoice>,
        depth: usize,
        stream: Option<StreamCallback>,
    ) -> Result<(String, bool)> {
        let model_config = self.lookup_model(&target.model)?;

        let timeouts = self.effective_timeouts_for_api_base(&model_config.api_base);
        let client = self
            .get_llm_client(
                &model_config.api_base,
                &model_config.api_key,
                timeouts.synthesizer_timeout_secs,
            )
            .await?;

        let prompt_capacity = original_prompt.len()
            + 240
            + worker_responses
                .iter()
                .map(|(label, response)| label.len() + response.len() + 24)
                .sum::<usize>()
            + selected_choice.map_or(0, |choice| {
                choice.worker_name.len()
                    + choice.response.len()
                    + choice.reasoning.as_ref().map_or(0, String::len)
                    + 64
            });
        let mut synthesis_prompt = String::with_capacity(prompt_capacity);
        synthesis_prompt.push_str("原始用户问题：\n");
        synthesis_prompt.push_str(original_prompt);
        synthesis_prompt.push_str("\n\n以下是多个AI模型对该问题的回答：\n\n");

        for (i, (label, response)) in worker_responses.iter().enumerate() {
            let _ = write!(synthesis_prompt, "【模型{}：{}】\n{}\n\n", i + 1, label, response);
        }

        if let Some(choice) = selected_choice {
            let _ = write!(
                synthesis_prompt,
                "选择器推荐的最佳回答（编号 {} / 模型 {}）：\n{}\n\n",
                choice.index, choice.worker_name, choice.response
            );
            if let Some(reasoning) = &choice.reasoning {
                synthesis_prompt.push_str("推荐理由：\n");
                synthesis_prompt.push_str(reasoning);
                synthesis_prompt.push_str("\n\n");
            }
            synthesis_prompt.push_str(
                "请基于推荐回答进行优化，必要时参考其他回答补充信息，生成一个高质量的最终答案。要求：\n\
                1. 保留或提升推荐回答中的核心信息\n\
                2. 结合其他回答弥补遗漏或纠正错误\n\
                3. 保持逻辑清晰、结构合理\n\
                4. 直接给出最终答案，不要提及\"综合以上回答\"等元信息\n",
            );
        } else {
            synthesis_prompt.push_str(
                "请综合以上所有回答，生成一个高质量的最终答案。要求：\n\
                1. 综合各个模型的优点\n\
                2. 确保答案准确、完整\n\
                3. 保持清晰的逻辑和结构\n\
                4. 直接给出最终答案，不要提及\"综合以上回答\"等元信息\n",
            );
        }

        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: synthesis_prompt,
        }];

        let temperature = self.resolve_synthesizer_temperature(target, model_config, depth);

        tracing::debug!(
            "Using temperature {} for synthesizer {} at depth {}",
            temperature,
            target.model,
            depth
        );

        let completion = client
            .chat_completion_with_stream(&target.model, messages, Some(temperature), stream)
            .await?;

        Ok((completion.content, completion.streamed))
    }

    fn resolve_worker_temperature(
        &self,
        target: &WorkflowModelTarget,
        model_config: &ModelConfig,
        base_temperature: f32,
        analyzer_auto: bool,
    ) -> f32 {
        if let Some(t) = target.temperature {
            return t;
        }
        if let Some(t) = model_config.temperature {
            return t;
        }

        let auto_enabled = target
            .auto_temperature
            .or(model_config.auto_temperature)
            .unwrap_or(analyzer_auto);

        if auto_enabled {
            tracing::debug!(
                "Worker {} auto temperature enabled, using analyzer recommendation {}",
                target.model,
                base_temperature
            );
            base_temperature
        } else {
            tracing::debug!(
                "Worker {} auto temperature disabled, using default {}",
                target.model,
                DEFAULT_TEMPERATURE
            );
            DEFAULT_TEMPERATURE
        }
    }

    fn resolve_selector_temperature(
        &self,
        target: &WorkflowModelTarget,
        model_config: &ModelConfig,
        depth: usize,
    ) -> f32 {
        self.resolve_secondary_temperature(target, model_config, depth, "Selector")
    }

    fn resolve_synthesizer_temperature(
        &self,
        target: &WorkflowModelTarget,
        model_config: &ModelConfig,
        depth: usize,
    ) -> f32 {
        self.resolve_secondary_temperature(target, model_config, depth, "Synthesizer")
    }

    fn resolve_secondary_temperature(
        &self,
        target: &WorkflowModelTarget,
        model_config: &ModelConfig,
        depth: usize,
        role: &str,
    ) -> f32 {
        if let Some(t) = target.temperature {
            return t;
        }
        if let Some(t) = model_config.temperature {
            return t;
        }
        if matches!(target.auto_temperature, Some(true))
            || matches!(model_config.auto_temperature, Some(true))
        {
            tracing::debug!(
                "{} {} requested auto temperature at depth {}, defaulting to {}",
                role,
                target.model,
                depth,
                DEFAULT_TEMPERATURE
            );
        }
        DEFAULT_TEMPERATURE
    }

    fn lookup_model(&self, name: &str) -> Result<&ModelConfig> {
        let index = *self.model_configs.get(name).ok_or_else(|| {
            anyhow!(
                "Model '{}' not found in configuration. Did you define it under [[model]]?",
                name
            )
        })?;
        Ok(&self.config.models[index])
    }
}

fn parse_selector_choice(response: &str, worker_count: usize) -> Result<ParsedSelection> {
    if worker_count == 0 {
        return Err(anyhow!(
            "Selector cannot choose from an empty set of worker responses"
        ));
    }

    if let Some(json_str) = extract_first_json_object(response) {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(json_str) {
            if let Some(index_value) = find_value_in_json(
                &value,
                &[
                    "selected_index",
                    "index",
                    "choice",
                    "selected",
                    "best_index",
                    "best",
                ],
            ) {
                if let Some(index) = value_to_usize(index_value) {
                    if (1..=worker_count).contains(&index) {
                        let label = find_value_in_json(
                            &value,
                            &[
                                "selected_worker",
                                "selected_label",
                                "model",
                                "name",
                                "label",
                            ],
                        )
                        .and_then(value_to_string);
                        let reasoning = find_value_in_json(
                            &value,
                            &["reasoning", "explanation", "why", "comment", "analysis"],
                        )
                        .and_then(value_to_string);
                        let selected_response = find_value_in_json(
                            &value,
                            &["selected_response", "response", "content", "answer"],
                        )
                        .and_then(value_to_string);

                        return Ok(ParsedSelection {
                            index,
                            label,
                            reasoning,
                            selected_response,
                        });
                    }
                }
            }
        }
    }

    if let Some(index) = find_first_index_in_text(response, worker_count) {
        let reasoning_text = response.trim();
        let reasoning = if reasoning_text.is_empty() {
            None
        } else {
            Some(reasoning_text.to_string())
        };
        return Ok(ParsedSelection {
            index,
            label: None,
            reasoning,
            selected_response: None,
        });
    }

    Err(anyhow!(
        "Selector response did not include a valid selected_index"
    ))
}

fn extract_first_json_object(input: &str) -> Option<&str> {
    let mut depth = 0;
    let mut start = None;
    let mut in_string = false;
    let mut escape = false;

    for (idx, ch) in input.char_indices() {
        if in_string {
            if escape {
                escape = false;
                continue;
            }
            if ch == '\\' {
                escape = true;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => {
                if depth == 0 {
                    start = Some(idx);
                }
                depth += 1;
            }
            '}' => {
                if depth > 0 {
                    depth -= 1;
                    if depth == 0 {
                        if let Some(begin) = start {
                            return input.get(begin..=idx);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    None
}

fn find_value_in_json<'a>(
    value: &'a serde_json::Value,
    keys: &[&str],
) -> Option<&'a serde_json::Value> {
    match value {
        serde_json::Value::Object(map) => {
            for key in keys {
                if let Some(found) = map.get(*key) {
                    return Some(found);
                }
            }
            for val in map.values() {
                if let Some(found) = find_value_in_json(val, keys) {
                    return Some(found);
                }
            }
            None
        }
        serde_json::Value::Array(items) => {
            items.iter().find_map(|item| find_value_in_json(item, keys))
        }
        _ => None,
    }
}

fn value_to_usize(value: &serde_json::Value) -> Option<usize> {
    match value {
        serde_json::Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                Some(u as usize)
            } else if let Some(f) = n.as_f64() {
                let rounded = f.round();
                if (rounded - f).abs() < f32::EPSILON as f64 {
                    let val = rounded as i64;
                    if val > 0 {
                        Some(val as usize)
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        }
        serde_json::Value::String(s) => s.trim().parse::<usize>().ok(),
        _ => None,
    }
}

fn value_to_string(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::Null => None,
        serde_json::Value::String(s) => {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        }
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        serde_json::Value::Array(items) => {
            let mut parts = Vec::new();
            for item in items {
                if let Some(text) = value_to_string(item) {
                    let trimmed = text.trim();
                    if !trimmed.is_empty() {
                        parts.push(trimmed.to_string());
                    }
                }
            }
            if parts.is_empty() {
                None
            } else {
                Some(parts.join(" "))
            }
        }
        serde_json::Value::Object(_) => None,
    }
}

fn find_first_index_in_text(text: &str, worker_count: usize) -> Option<usize> {
    let mut current = String::new();

    for ch in text.chars() {
        if ch.is_ascii_digit() {
            current.push(ch);
        } else {
            if let Some(index) = parse_candidate_index(&current, worker_count) {
                return Some(index);
            }
            current.clear();
        }
    }

    if let Some(index) = parse_candidate_index(&current, worker_count) {
        return Some(index);
    }

    None
}

fn parse_candidate_index(fragment: &str, worker_count: usize) -> Option<usize> {
    if fragment.is_empty() {
        return None;
    }
    if let Ok(value) = fragment.parse::<usize>() {
        if value >= 1 && value <= worker_count {
            return Some(value);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        Config, ModelConfig, ServerConfig, TimeoutConfig, WorkflowConfig, WorkflowModelTarget,
        WorkflowPlan, WorkflowWorker,
    };
    use axum::{extract::State, routing::post, Json, Router};
    use serde_json::{json, Value};
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::Barrier;

    fn build_test_config_with_workers(workers: Vec<WorkflowWorker>) -> Config {
        Config {
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 11435,
            },
            models: vec![ModelConfig {
                name: "primary".to_string(),
                api_base: "http://localhost".to_string(),
                api_key: "sk-test".to_string(),
                temperature: Some(0.2),
                auto_temperature: None,
            }],
            workflow_integration: WorkflowPlan {
                analyzer: WorkflowModelTarget {
                    model: "primary".to_string(),
                    temperature: Some(0.2),
                    auto_temperature: None,
                },
                workers,
                synthesizer: Some(WorkflowModelTarget {
                    model: "primary".to_string(),
                    temperature: Some(0.2),
                    auto_temperature: None,
                }),
                selector: None,
                nested_worker_depth: None,
            },
            workflow: WorkflowConfig {
                timeouts: TimeoutConfig {
                    analyzer_timeout_secs: 30,
                    worker_timeout_secs: 30,
                    synthesizer_timeout_secs: 30,
                },
                domains: HashMap::new(),
                worker_concurrency: None,
            },
        }
    }

    #[test]
    fn worker_auto_temperature_disabled_falls_back_to_default() {
        let mut config = build_test_config_with_workers(Vec::new());
        config.models[0].temperature = None;
        config.models[0].auto_temperature = None;

        let engine = WorkflowEngine::new(config);
        let target = WorkflowModelTarget {
            model: "primary".to_string(),
            temperature: None,
            auto_temperature: None,
        };

        let resolved = {
            let model_config = engine.lookup_model(&target.model).unwrap();
            engine.resolve_worker_temperature(&target, model_config, 0.9, false)
        };

        assert!(
            (resolved - DEFAULT_TEMPERATURE).abs() < f32::EPSILON,
            "expected default temperature {}, got {}",
            DEFAULT_TEMPERATURE,
            resolved
        );
    }

    #[test]
    fn worker_auto_temperature_flag_enables_analyzer_reuse() {
        let mut config = build_test_config_with_workers(Vec::new());
        config.models[0].temperature = None;
        config.models[0].auto_temperature = None;

        let engine = WorkflowEngine::new(config);
        let target = WorkflowModelTarget {
            model: "primary".to_string(),
            temperature: None,
            auto_temperature: Some(true),
        };
        let base = 0.42;

        let resolved = {
            let model_config = engine.lookup_model(&target.model).unwrap();
            engine.resolve_worker_temperature(&target, model_config, base, false)
        };

        assert!(
            (resolved - base).abs() < f32::EPSILON,
            "expected analyzer temperature {}, got {}",
            base,
            resolved
        );
    }

    #[test]
    fn worker_inherits_analyzer_auto_when_unspecified() {
        let mut config = build_test_config_with_workers(Vec::new());
        config.models[0].temperature = None;
        config.models[0].auto_temperature = None;

        let engine = WorkflowEngine::new(config);
        let target = WorkflowModelTarget {
            model: "primary".to_string(),
            temperature: None,
            auto_temperature: None,
        };
        let base = 0.73;

        let resolved = {
            let model_config = engine.lookup_model(&target.model).unwrap();
            engine.resolve_worker_temperature(&target, model_config, base, true)
        };

        assert!(
            (resolved - base).abs() < f32::EPSILON,
            "expected analyzer temperature {}, got {}",
            base,
            resolved
        );
    }

    #[test]
    fn worker_explicit_auto_false_overrides_analyzer_auto() {
        let mut config = build_test_config_with_workers(Vec::new());
        config.models[0].temperature = None;
        config.models[0].auto_temperature = None;

        let engine = WorkflowEngine::new(config);
        let target = WorkflowModelTarget {
            model: "primary".to_string(),
            temperature: None,
            auto_temperature: Some(false),
        };

        let resolved = {
            let model_config = engine.lookup_model(&target.model).unwrap();
            engine.resolve_worker_temperature(&target, model_config, 0.81, true)
        };

        assert!(
            (resolved - DEFAULT_TEMPERATURE).abs() < f32::EPSILON,
            "expected default temperature {}, got {}",
            DEFAULT_TEMPERATURE,
            resolved
        );
    }

    #[test]
    fn worker_uses_model_config_auto_flag() {
        let mut config = build_test_config_with_workers(Vec::new());
        config.models[0].temperature = None;
        config.models[0].auto_temperature = Some(true);

        let engine = WorkflowEngine::new(config);
        let target = WorkflowModelTarget {
            model: "primary".to_string(),
            temperature: None,
            auto_temperature: None,
        };
        let base = 0.37;

        let resolved = {
            let model_config = engine.lookup_model(&target.model).unwrap();
            engine.resolve_worker_temperature(&target, model_config, base, false)
        };

        assert!(
            (resolved - base).abs() < f32::EPSILON,
            "expected analyzer temperature {}, got {}",
            base,
            resolved
        );
    }

    #[test]
    fn selector_parser_handles_json_payload() {
        let payload = r#"{
  "selected_index": 2,
  "selected_worker": "model-b",
  "reasoning": "更详细",
  "selected_response": "Answer B"
}"#;
        let parsed = parse_selector_choice(payload, 3).expect("should parse");
        assert_eq!(parsed.index, 2);
        assert_eq!(parsed.label.as_deref(), Some("model-b"));
        assert_eq!(parsed.selected_response.as_deref(), Some("Answer B"));
        assert_eq!(parsed.reasoning.as_deref(), Some("更详细"));
    }

    #[test]
    fn selector_parser_falls_back_to_text_index() {
        let payload = "我认为第1个回答最好，因为涵盖了所有要点。";
        let parsed = parse_selector_choice(payload, 3).expect("should parse");
        assert_eq!(parsed.index, 1);
        assert!(parsed
            .reasoning
            .as_ref()
            .expect("has reasoning")
            .contains("回答"));
    }

    #[tokio::test]
    async fn reports_missing_workers_in_error() {
        let config = build_test_config_with_workers(Vec::new());
        let engine = WorkflowEngine::new(config);
        let err = engine
            .process("hello world".to_string())
            .await
            .expect_err("expected failure when no workers configured");
        let message = err.to_string();
        assert!(
            message.contains("workflow:primary"),
            "message did not include plan label: {}",
            message
        );
        assert!(
            message.contains("no worker nodes configured"),
            "message did not explain worker misconfiguration: {}",
            message
        );
    }

    #[test]
    fn execution_detail_temperatures_serialize_like_value_path() {
        let legacy_temperature = serde_json::json!({ "temperature": 1.4f32 })["temperature"].to_string();

        let analyzer = AnalyzerDetails {
            model: "m".to_string(),
            temperature: 1.4,
            auto_temperature: true,
        };
        let serialized = serde_json::to_string(&analyzer).unwrap();
        assert!(
            serialized.contains(&format!("\"temperature\":{}", legacy_temperature)),
            "analyzer temperature serialized as {} (expected legacy {})",
            serialized,
            legacy_temperature
        );

        let synthesizer = SynthesizerDetails {
            model: "m".to_string(),
            temperature: 1.4,
        };
        let serialized = serde_json::to_string(&synthesizer).unwrap();
        assert!(
            serialized.contains(&format!("\"temperature\":{}", legacy_temperature)),
            "synthesizer temperature serialized as {} (expected legacy {})",
            serialized,
            legacy_temperature
        );

        let selector = SelectorDetails {
            model: "m".to_string(),
            temperature: 1.4,
            selected_index: None,
            selected_worker: None,
            selected_response: None,
            reasoning: None,
            success: true,
            error: None,
            raw_output: None,
        };
        let serialized = serde_json::to_string(&selector).unwrap();
        assert!(
            serialized.contains(&format!("\"temperature\":{}", legacy_temperature)),
            "selector temperature serialized as {} (expected legacy {})",
            serialized,
            legacy_temperature
        );

        let legacy_worker_temperature =
            serde_json::json!({ "temperature": 0.3f32 })["temperature"].to_string();
        let worker = WorkerDetails {
            name: "w".to_string(),
            temperature: Some(0.3),
            response: None,
            success: true,
            error: None,
            nested: None,
        };
        let serialized = serde_json::to_string(&worker).unwrap();
        assert!(
            serialized.contains(&format!("\"temperature\":{}", legacy_worker_temperature)),
            "worker temperature serialized as {} (expected legacy {})",
            serialized,
            legacy_worker_temperature
        );
    }

    #[tokio::test]
    async fn includes_worker_failure_details() {
        let workers = vec![WorkflowWorker::Model(WorkflowModelTarget {
            model: "missing".to_string(),
            temperature: None,
            auto_temperature: None,
        })];
        let config = build_test_config_with_workers(workers);
        let engine = WorkflowEngine::new(config);
        let err = engine
            .process("hello world".to_string())
            .await
            .expect_err("expected failure when worker model missing");
        let message = err.to_string();
        assert!(
            message.contains("missing"),
            "message did not include worker label: {}",
            message
        );
        assert!(
            message.contains("Model 'missing' not found in configuration"),
            "message did not include underlying error: {}",
            message
        );
        assert!(
            message.contains("Worker errors"),
            "message did not include worker errors section: {}",
            message
        );
    }

    struct ConcurrencyProbe {
        active: AtomicUsize,
        max_active: AtomicUsize,
        calls: AtomicUsize,
        delay: Duration,
        barrier: Option<Arc<Barrier>>,
    }

    impl ConcurrencyProbe {
        fn new(delay: Duration, barrier: Option<Arc<Barrier>>) -> Self {
            Self {
                active: AtomicUsize::new(0),
                max_active: AtomicUsize::new(0),
                calls: AtomicUsize::new(0),
                delay,
                barrier,
            }
        }
    }

    async fn concurrency_probe_handler(
        State(probe): State<Arc<ConcurrencyProbe>>,
        Json(body): Json<Value>,
    ) -> Json<Value> {
        let current = probe.active.fetch_add(1, Ordering::SeqCst) + 1;
        probe.max_active.fetch_max(current, Ordering::SeqCst);
        probe.calls.fetch_add(1, Ordering::SeqCst);

        if let Some(barrier) = &probe.barrier {
            barrier.wait().await;
        }
        if !probe.delay.is_zero() {
            tokio::time::sleep(probe.delay).await;
        }

        let model = body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        let content = format!("echo:{}", model);

        probe.active.fetch_sub(1, Ordering::SeqCst);

        Json(json!({
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }]
        }))
    }

    async fn spawn_concurrency_probe(probe: Arc<ConcurrencyProbe>) -> String {
        let app = Router::new()
            .route("/v1/chat/completions", post(concurrency_probe_handler))
            .with_state(probe);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        format!("http://127.0.0.1:{}/v1", port)
    }

    fn build_network_test_config(
        api_base: &str,
        model_names: &[&str],
        concurrency: Option<usize>,
    ) -> Config {
        let models = model_names
            .iter()
            .map(|name| ModelConfig {
                name: (*name).to_string(),
                api_base: api_base.to_string(),
                api_key: "sk-test".to_string(),
                temperature: None,
                auto_temperature: None,
            })
            .collect();
        let workers = model_names
            .iter()
            .map(|name| {
                WorkflowWorker::Model(WorkflowModelTarget {
                    model: (*name).to_string(),
                    temperature: None,
                    auto_temperature: None,
                })
            })
            .collect();

        Config {
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 0,
            },
            models,
            workflow_integration: WorkflowPlan {
                analyzer: WorkflowModelTarget {
                    model: model_names[0].to_string(),
                    temperature: None,
                    auto_temperature: None,
                },
                workers,
                synthesizer: Some(WorkflowModelTarget {
                    model: model_names[0].to_string(),
                    temperature: None,
                    auto_temperature: None,
                }),
                selector: None,
                nested_worker_depth: None,
            },
            workflow: WorkflowConfig {
                timeouts: TimeoutConfig {
                    analyzer_timeout_secs: 30,
                    worker_timeout_secs: 30,
                    synthesizer_timeout_secs: 30,
                },
                domains: HashMap::new(),
                worker_concurrency: concurrency,
            },
        }
    }

    #[test]
    fn worker_concurrency_defaults_and_bounds() {
        let mut config = build_test_config_with_workers(Vec::new());

        let engine = WorkflowEngine::new(config.clone());
        assert_eq!(engine.worker_concurrency(), DEFAULT_WORKER_CONCURRENCY);

        config.workflow.worker_concurrency = Some(2);
        let engine = WorkflowEngine::new(config.clone());
        assert_eq!(engine.worker_concurrency(), 2);

        config.workflow.worker_concurrency = Some(0);
        let engine = WorkflowEngine::new(config);
        assert_eq!(engine.worker_concurrency(), 1);
    }

    #[tokio::test]
    async fn workers_dispatch_concurrently_and_preserve_configuration_order() {
        let barrier = Arc::new(Barrier::new(3));
        let probe = Arc::new(ConcurrencyProbe::new(Duration::ZERO, Some(barrier)));
        let api_base = spawn_concurrency_probe(Arc::clone(&probe)).await;
        let config = build_network_test_config(&api_base, &["alpha", "beta", "gamma"], None);
        let engine = WorkflowEngine::new(config);

        let details = tokio::time::timeout(
            Duration::from_secs(10),
            engine.run_workers_with_details(
                &engine.config.workflow_integration,
                "ping",
                1.0,
                false,
                0,
            ),
        )
        .await
        .expect("workers should reach the barrier concurrently instead of running sequentially")
        .expect("workers should succeed");

        assert_eq!(probe.calls.load(Ordering::SeqCst), 3);
        let names: Vec<&str> = details.iter().map(|d| d.name.as_str()).collect();
        assert_eq!(names, ["alpha", "beta", "gamma"]);
        for details_entry in &details {
            assert!(
                details_entry.success,
                "worker {} failed",
                details_entry.name
            );
            assert_eq!(
                details_entry.response.as_deref(),
                Some(format!("echo:{}", details_entry.name).as_str()),
                "worker responses must stay associated with their model"
            );
        }
    }

    #[tokio::test]
    async fn workers_respect_configured_concurrency_limit() {
        let probe = Arc::new(ConcurrencyProbe::new(Duration::from_millis(200), None));
        let api_base = spawn_concurrency_probe(Arc::clone(&probe)).await;
        let config = build_network_test_config(&api_base, &["a", "b", "c"], Some(2));
        let engine = WorkflowEngine::new(config);

        let details = engine
            .run_workers_with_details(&engine.config.workflow_integration, "ping", 1.0, false, 0)
            .await
            .expect("workers should succeed");

        assert_eq!(details.len(), 3);
        assert_eq!(probe.calls.load(Ordering::SeqCst), 3);
        assert!(
            probe.max_active.load(Ordering::SeqCst) >= 2,
            "expected at least two overlapping worker calls"
        );
        assert!(details.iter().all(|d| d.success));
    }

    #[tokio::test]
    async fn worker_concurrency_one_keeps_calls_sequential() {
        let probe = Arc::new(ConcurrencyProbe::new(Duration::from_millis(50), None));
        let api_base = spawn_concurrency_probe(Arc::clone(&probe)).await;
        let config = build_network_test_config(&api_base, &["a", "b", "c"], Some(1));
        let engine = WorkflowEngine::new(config);

        let details = engine
            .run_workers_with_details(&engine.config.workflow_integration, "ping", 1.0, false, 0)
            .await
            .expect("workers should succeed");

        assert_eq!(details.len(), 3);
        assert_eq!(probe.calls.load(Ordering::SeqCst), 3);
        assert_eq!(
            probe.max_active.load(Ordering::SeqCst),
            1,
            "concurrency limit of one must not overlap worker calls"
        );
    }
}
