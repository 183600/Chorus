use crate::config::Config;
use crate::workflow::{StreamCallback, WorkflowEngine, WorkflowExecutionDetails};
use anyhow::Result;
use axum::{
    extract::{Request, State},
    http::{header, HeaderMap, HeaderValue, Method, StatusCode},
    middleware::{self, Next},
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use bytes::Bytes;
use futures::{stream, StreamExt};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::UnboundedReceiverStream;

use std::convert::Infallible;
use tower_http::cors::CorsLayer;

type SharedState = Arc<AppState>;

const STREAM_CHUNK_SIZE: usize = 120;

pub struct AppState {
    config: Arc<Config>,
    workflow_engine: WorkflowEngine,
    health_body: Bytes,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub model: Option<String>,
    pub prompt: String,
    pub stream: Option<bool>,
    pub include_workflow: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: Option<String>,
    pub messages: Vec<Message>,
    pub stream: Option<bool>,
    pub include_workflow: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum PromptInput {
    Single(String),
    Multiple(Vec<String>),
}

impl PromptInput {
    fn into_prompt(self) -> String {
        match self {
            PromptInput::Single(s) => s,
            PromptInput::Multiple(items) => items.join("\n"),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub prompt: PromptInput,
    pub stream: Option<bool>,
    pub include_workflow: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workflow: Option<WorkflowExecutionDetails>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatResponse {
    pub model: String,
    pub created_at: String,
    pub message: Message,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workflow: Option<WorkflowExecutionDetails>,
}

#[cfg(test)]
fn for_each_chunk(content: &str, max_len: usize, mut emit: impl FnMut(&str)) {
    if content.is_empty() {
        emit("");
        return;
    }

    let mut start = 0usize;
    let mut current_len = 0usize;

    for (idx, ch) in content.char_indices() {
        current_len += ch.len_utf8();

        if ch == '\n' || current_len >= max_len {
            let end = idx + ch.len_utf8();
            emit(&content[start..end]);
            start = end;
            current_len = 0;
        }
    }

    if start < content.len() {
        emit(&content[start..]);
    }
}

#[cfg(test)]
fn chunk_text(content: &str, max_len: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    for_each_chunk(content, max_len, |piece| chunks.push(piece.to_string()));
    chunks
}

/// Lazily splits an owned stream segment into protocol-sized events.
struct ChunkEvents<F> {
    content: String,
    max_len: usize,
    pos: usize,
    finished: bool,
    make_event: F,
}

impl<F> ChunkEvents<F>
where
    F: FnMut(&str) -> Result<Event, Infallible>,
{
    fn new(content: String, max_len: usize, make_event: F) -> Self {
        Self {
            content,
            max_len,
            pos: 0,
            finished: false,
            make_event,
        }
    }
}

impl<F> Iterator for ChunkEvents<F>
where
    F: FnMut(&str) -> Result<Event, Infallible>,
{
    type Item = Result<Event, Infallible>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let content_len = self.content.len();
        if content_len == 0 {
            self.finished = true;
            return Some((self.make_event)(""));
        }
        if self.pos >= content_len {
            self.finished = true;
            return None;
        }

        let start = self.pos;
        let limit = start
            .saturating_add(self.max_len.max(1))
            .min(content_len);
        let mut end = limit;
        while end < content_len && !self.content.is_char_boundary(end) {
            end += 1;
        }

        let end = match self.content[start..end].find('\n') {
            Some(offset) => start + offset + 1,
            None => end,
        };

        self.pos = end;
        if end >= content_len {
            self.finished = true;
        }

        let Self {
            content,
            make_event,
            ..
        } = self;
        Some(make_event(&content[start..end]))
    }
}

fn chunk_events<F>(segment: String, make_event: F) -> ChunkEvents<F>
where
    F: FnMut(&str) -> Result<Event, Infallible>,
{
    ChunkEvents::new(segment, STREAM_CHUNK_SIZE, make_event)
}

fn build_prompt_from_messages(messages: &[Message]) -> String {
    let capacity = messages
        .iter()
        .map(|message| message.role.len() + message.content.len() + 3)
        .sum();
    let mut prompt = String::with_capacity(capacity);
    for (index, message) in messages.iter().enumerate() {
        if index > 0 {
            prompt.push('\n');
        }
        prompt.push_str(&message.role);
        prompt.push_str(": ");
        prompt.push_str(&message.content);
    }
    prompt
}

#[derive(Serialize)]
struct GenerateStreamChunk<'a> {
    created_at: &'a str,
    done: bool,
    model: &'a str,
    response: &'a str,
}

#[derive(Serialize)]
struct ChatStreamMessage<'a> {
    content: &'a str,
    role: &'static str,
}

#[derive(Serialize)]
struct ChatStreamChunk<'a> {
    created_at: &'a str,
    done: bool,
    message: ChatStreamMessage<'a>,
    model: &'a str,
}

#[derive(Serialize)]
struct OpenAiDeltaContent<'a> {
    content: &'a str,
}

#[derive(Serialize)]
struct OpenAiChatChoice<'a> {
    delta: OpenAiDeltaContent<'a>,
    finish_reason: Option<&'static str>,
    index: u32,
}

#[derive(Serialize)]
struct OpenAiChatStreamChunk<'a> {
    choices: [OpenAiChatChoice<'a>; 1],
    created: i64,
    id: &'a str,
    model: &'a str,
    object: &'static str,
}

#[derive(Serialize)]
struct OpenAiCompletionChoice<'a> {
    finish_reason: Option<&'static str>,
    index: u32,
    logprobs: Option<()>,
    text: &'a str,
}

#[derive(Serialize)]
struct OpenAiCompletionStreamChunk<'a> {
    choices: [OpenAiCompletionChoice<'a>; 1],
    created: i64,
    id: &'a str,
    model: &'a str,
    object: &'static str,
}

#[derive(Serialize)]
struct ResponsesDelta<'a> {
    text: &'a str,
    #[serde(rename = "type")]
    kind: &'static str,
}

#[derive(Serialize)]
struct ResponsesDeltaChunk<'a> {
    content_index: usize,
    created: i64,
    delta: ResponsesDelta<'a>,
    id: &'a str,
    model: &'a str,
    object: &'static str,
    output_index: u32,
}

#[derive(Serialize)]
struct OpenAiChatCompletionMessage<'a> {
    content: &'a str,
    role: &'static str,
}

#[derive(Serialize)]
struct OpenAiChatCompletionChoice<'a> {
    finish_reason: &'static str,
    index: u32,
    message: OpenAiChatCompletionMessage<'a>,
}

#[derive(Serialize)]
struct OpenAiChatCompletionResponse<'a> {
    choices: [OpenAiChatCompletionChoice<'a>; 1],
    created: i64,
    id: &'a str,
    model: &'a str,
    object: &'static str,
    workflow: Option<WorkflowExecutionDetails>,
}

#[derive(Serialize)]
struct OpenAiCompletionResponseChoice<'a> {
    finish_reason: &'static str,
    index: u32,
    logprobs: Option<()>,
    text: &'a str,
}

#[derive(Serialize)]
struct OpenAiCompletionResponse<'a> {
    choices: [OpenAiCompletionResponseChoice<'a>; 1],
    created: i64,
    id: &'a str,
    model: &'a str,
    object: &'static str,
    workflow: Option<WorkflowExecutionDetails>,
}

#[derive(Serialize)]
struct ResponsesOutputContent<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    text: &'a str,
}

#[derive(Serialize)]
struct ResponsesOutputMessage<'a> {
    id: &'a str,
    #[serde(rename = "type")]
    kind: &'static str,
    role: &'static str,
    content: [ResponsesOutputContent<'a>; 1],
}

#[derive(Serialize)]
struct ResponsesResponse<'a> {
    id: &'a str,
    object: &'static str,
    created: i64,
    model: &'a str,
    status: &'static str,
    output: [ResponsesOutputMessage<'a>; 1],
    output_text: &'a str,
    workflow: Option<WorkflowExecutionDetails>,
}

async fn execute_workflow(
    state: &AppState,
    prompt: String,
    include_workflow: bool,
    stream: Option<StreamCallback>,
) -> Result<(String, Option<WorkflowExecutionDetails>), AppError> {
    if include_workflow {
        let result = state
            .workflow_engine
            .process_with_details_stream(prompt, stream)
            .await?;
        Ok((result.final_response, Some(result.execution_details)))
    } else {
        let response = state
            .workflow_engine
            .process_with_stream(prompt, stream)
            .await?;
        Ok((response, None))
    }
}

type WorkflowTaskResult = Result<(String, Option<WorkflowExecutionDetails>), AppError>;

/// Runs the workflow for a streaming response.
///
/// The workflow is cancelled as soon as the SSE response body is dropped
/// (e.g. the client disconnected or hit stop) so in-flight upstream LLM calls
/// stop consuming quota instead of running to completion unnoticed.
fn spawn_streaming_workflow(
    state: SharedState,
    prompt: String,
    include_workflow: bool,
) -> (
    mpsc::UnboundedReceiver<String>,
    oneshot::Receiver<WorkflowTaskResult>,
) {
    let (chunk_tx, chunk_rx) = mpsc::unbounded_channel::<String>();
    let (result_tx, result_rx) = oneshot::channel();
    let disconnect = chunk_tx.clone();

    tokio::spawn(async move {
        let result = tokio::select! {
            result = execute_workflow(&state, prompt, include_workflow, Some(chunk_tx)) => result,
            _ = disconnect.closed() => {
                tracing::info!("Client disconnected; cancelling the in-flight workflow");
                Err(AppError::new(
                    StatusCode::from_u16(499).expect("499 is a valid status code"),
                    anyhow::anyhow!("client disconnected before the workflow completed"),
                ))
            }
        };
        let _ = result_tx.send(result);
    });

    (chunk_rx, result_rx)
}

fn insert_workflow_field(
    payload: &mut serde_json::Value,
    details: &Option<WorkflowExecutionDetails>,
) {
    if let Some(details) = details {
        if let serde_json::Value::Object(map) = payload {
            if let Ok(value) = serde_json::to_value(details) {
                map.insert("workflow".to_string(), value);
            }
        }
    }
}

pub async fn start_server(config: Arc<Config>) -> Result<()> {
    let workflow_engine = WorkflowEngine::from_shared(Arc::clone(&config));
    let addr = format!("{}:{}", config.server.host, config.server.port);

    let health_body = Bytes::from(serde_json::to_vec(&serde_json::json!({
        "status": "ok",
        "service": "Chorus",
        "version": "0.1.0"
    }))?);

    if config.server.api_key.is_some() {
        tracing::info!("API key authentication is enabled for /api and /v1 endpoints");
    }

    let state = Arc::new(AppState {
        config,
        workflow_engine,
        health_body,
    });

    let app = build_router(state);

    let listener = tokio::net::TcpListener::bind(&addr).await?;

    tracing::info!("Chorus server listening on http://{}", addr);

    axum::serve(listener, app).await?;

    Ok(())
}

fn build_router(state: SharedState) -> Router {
    Router::new()
        .route("/", get(health_check))
        // API v0 style
        .route("/api/generate", post(generate))
        .route("/api/chat", post(chat))
        .route("/api/tags", get(list_models))
        // API v1 alias (same handlers)
        .route("/v1/generate", post(generate))
        .route("/v1/chat", post(chat))
        // OpenAI-compatible endpoints (for Cherry Studio and similar clients)
        .route("/v1/chat/completions", post(openai_chat_completions))
        .route("/v1/completions", post(openai_completions))
        .route("/v1/models", get(list_models_openai))
        .route("/v1/tags", get(list_models))
        .route("/v1/responses", post(responses))
        .layer(middleware::from_fn_with_state(
            Arc::clone(&state),
            require_api_key,
        ))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// Rejects requests without a valid key when `[server].api_key` (or the
/// `CHORUS_API_KEY` override) is set. The health probe at `/` and CORS
/// preflight requests stay public.
async fn require_api_key(
    State(state): State<SharedState>,
    request: Request,
    next: Next,
) -> Response {
    let Some(expected) = state.config.server.api_key.as_deref() else {
        return next.run(request).await;
    };

    if request.uri().path() == "/" || request.method() == Method::OPTIONS {
        return next.run(request).await;
    }

    let authorized = extract_api_key(request.headers())
        .map(|provided| constant_time_eq(provided.as_bytes(), expected.as_bytes()))
        .unwrap_or(false);

    if authorized {
        next.run(request).await
    } else {
        unauthorized_response()
    }
}

fn extract_api_key(headers: &HeaderMap) -> Option<&str> {
    if let Some(value) = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
    {
        if let Some(token) = bearer_token(value) {
            return Some(token);
        }
    }

    headers
        .get("x-api-key")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

fn bearer_token(value: &str) -> Option<&str> {
    let (scheme, token) = value.trim().split_once(' ')?;
    if !scheme.eq_ignore_ascii_case("bearer") {
        return None;
    }
    let token = token.trim();
    if token.is_empty() {
        None
    } else {
        Some(token)
    }
}

fn constant_time_eq(provided: &[u8], expected: &[u8]) -> bool {
    if provided.len() != expected.len() {
        return false;
    }
    let mut diff = 0u8;
    for (a, b) in provided.iter().zip(expected.iter()) {
        diff |= a ^ b;
    }
    diff == 0
}

fn unauthorized_response() -> Response {
    let mut response = (
        StatusCode::UNAUTHORIZED,
        Json(serde_json::json!({
            "error": "Unauthorized: supply the configured API key via `Authorization: Bearer <key>` or the `x-api-key` header."
        })),
    )
        .into_response();
    response.headers_mut().insert(
        header::WWW_AUTHENTICATE,
        HeaderValue::from_static("Bearer"),
    );
    response
}

async fn health_check(State(state): State<SharedState>) -> Response {
    let mut response = Response::new(axum::body::Body::from(state.health_body.clone()));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    response
}

async fn generate(
    State(state): State<SharedState>,
    Json(req): Json<GenerateRequest>,
) -> Result<Response, AppError> {
    let GenerateRequest {
        model,
        prompt,
        stream,
        include_workflow,
    } = req;

    tracing::info!(
        "Received generate request, stream: {:?}, include_workflow: {:?}",
        stream,
        include_workflow
    );

    let model_name = model.unwrap_or_else(|| "chorus".to_string());
    let stream_enabled = stream.unwrap_or(false);
    let include_workflow_details = include_workflow.unwrap_or(false);

    if stream_enabled {
        let created_at = chrono::Utc::now().to_rfc3339();
        let (chunk_rx, result_rx) =
            spawn_streaming_workflow(state.clone(), prompt, include_workflow_details);

        let chunk_stream =
            UnboundedReceiverStream::new(chunk_rx).flat_map({
                let model_name: Arc<str> = Arc::from(model_name.as_str());
                let created_at: Arc<str> = Arc::from(created_at.as_str());
                move |segment| {
                    let model_name = Arc::clone(&model_name);
                    let created_at = Arc::clone(&created_at);
                    stream::iter(chunk_events(segment, move |piece| {
                        Ok(Event::default()
                            .json_data(GenerateStreamChunk {
                                created_at: &created_at,
                                done: false,
                                model: &model_name,
                                response: piece,
                            })
                            .unwrap())
                    }))
                }
            });

        let completion_stream = futures::stream::once({
            let model_name = model_name.clone();
            let created_at = created_at.clone();
            async move {
                match result_rx.await {
                    Ok(Ok((_, workflow_details))) => {
                        let mut payload = serde_json::json!({
                            "model": model_name,
                            "created_at": created_at,
                            "response": "",
                            "done": true,
                        });
                        insert_workflow_field(&mut payload, &workflow_details);
                        Ok::<Event, Infallible>(Event::default().json_data(payload).unwrap())
                    }
                    Ok(Err(err)) => {
                        let error_message = err.error.to_string();
                        let payload = serde_json::json!({
                            "model": model_name,
                            "created_at": created_at,
                            "done": true,
                            "error": error_message,
                        });
                        Ok::<Event, Infallible>(Event::default().json_data(payload).unwrap())
                    }
                    Err(_) => {
                        let payload = serde_json::json!({
                            "model": model_name,
                            "created_at": created_at,
                            "done": true,
                            "error": "stream cancelled",
                        });
                        Ok::<Event, Infallible>(Event::default().json_data(payload).unwrap())
                    }
                }
            }
        });

        let sse_stream = chunk_stream.chain(completion_stream);
        return Ok(Sse::new(sse_stream).into_response());
    }

    let (response_text, workflow_details) =
        execute_workflow(&state, prompt, include_workflow_details, None).await?;

    Ok(Json(GenerateResponse {
        model: model_name,
        created_at: chrono::Utc::now().to_rfc3339(),
        response: response_text,
        done: true,
        workflow: workflow_details,
    })
    .into_response())
}

async fn chat(
    State(state): State<SharedState>,
    Json(req): Json<ChatRequest>,
) -> Result<Response, AppError> {
    tracing::info!(
        "Received chat request with {} messages, stream: {:?}, include_workflow: {:?}",
        req.messages.len(),
        req.stream,
        req.include_workflow
    );

    let prompt = build_prompt_from_messages(&req.messages);

    let model_name = req.model.unwrap_or_else(|| "chorus".to_string());
    let stream_enabled = req.stream.unwrap_or(false);
    let include_workflow_details = req.include_workflow.unwrap_or(false);

    if stream_enabled {
        let created_at = chrono::Utc::now().to_rfc3339();
        let (chunk_rx, result_rx) =
            spawn_streaming_workflow(state.clone(), prompt, include_workflow_details);

        let chunk_stream =
            UnboundedReceiverStream::new(chunk_rx).flat_map({
                let model_name: Arc<str> = Arc::from(model_name.as_str());
                let created_at: Arc<str> = Arc::from(created_at.as_str());
                move |segment| {
                    let model_name = Arc::clone(&model_name);
                    let created_at = Arc::clone(&created_at);
                    stream::iter(chunk_events(segment, move |piece| {
                        Ok(Event::default()
                            .json_data(ChatStreamChunk {
                                created_at: &created_at,
                                done: false,
                                message: ChatStreamMessage {
                                    content: piece,
                                    role: "assistant",
                                },
                                model: &model_name,
                            })
                            .unwrap())
                    }))
                }
            });

        let completion_stream = futures::stream::once({
            let model_name = model_name.clone();
            let created_at = created_at.clone();
            async move {
                match result_rx.await {
                    Ok(Ok((_, workflow_details))) => {
                        let mut payload = serde_json::json!({
                            "model": model_name,
                            "created_at": created_at,
                            "message": {
                                "role": "assistant",
                                "content": "",
                            },
                            "done": true,
                        });
                        insert_workflow_field(&mut payload, &workflow_details);
                        Ok::<Event, Infallible>(Event::default().json_data(payload).unwrap())
                    }
                    Ok(Err(err)) => {
                        let error_message = err.error.to_string();
                        let payload = serde_json::json!({
                            "model": model_name,
                            "created_at": created_at,
                            "message": {
                                "role": "assistant",
                                "content": "",
                            },
                            "done": true,
                            "error": error_message,
                        });
                        Ok::<Event, Infallible>(Event::default().json_data(payload).unwrap())
                    }
                    Err(_) => {
                        let payload = serde_json::json!({
                            "model": model_name,
                            "created_at": created_at,
                            "message": {
                                "role": "assistant",
                                "content": "",
                            },
                            "done": true,
                            "error": "stream cancelled",
                        });
                        Ok::<Event, Infallible>(Event::default().json_data(payload).unwrap())
                    }
                }
            }
        });

        let sse_stream = chunk_stream.chain(completion_stream);
        return Ok(Sse::new(sse_stream).into_response());
    }

    let (response_text, workflow_details) =
        execute_workflow(&state, prompt, include_workflow_details, None).await?;

    Ok(Json(ChatResponse {
        model: model_name,
        created_at: chrono::Utc::now().to_rfc3339(),
        message: Message {
            role: "assistant".to_string(),
            content: response_text,
        },
        done: true,
        workflow: workflow_details,
    })
    .into_response())
}

// OpenAI Chat Completions compatible endpoint
async fn openai_chat_completions(
    State(state): State<SharedState>,
    Json(req): Json<ChatRequest>,
) -> Result<Response, AppError> {
    tracing::info!(
        "Received OpenAI chat.completions request with {} messages, stream: {:?}",
        req.messages.len(),
        req.stream
    );

    let prompt = build_prompt_from_messages(&req.messages);

    let model_name = req.model.unwrap_or_else(|| "chorus".to_string());
    let stream_enabled = req.stream.unwrap_or(false);
    let include_workflow_details = req.include_workflow.unwrap_or(false);

    if stream_enabled {
        let now = chrono::Utc::now();
        let created = now.timestamp();
        let id = format!("chatcmpl_{}", now.timestamp_millis());
        let (chunk_rx, result_rx) =
            spawn_streaming_workflow(state.clone(), prompt, include_workflow_details);

        let initial_stream = futures::stream::once({
            let id = id.clone();
            let model_name = model_name.clone();
            async move {
                let payload = serde_json::json!({
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [ {
                        "index": 0,
                        "delta": { "role": "assistant" },
                        "finish_reason": serde_json::Value::Null
                    } ],
                });
                Ok::<Event, Infallible>(Event::default().json_data(payload).unwrap())
            }
        });

        let chunk_stream =
            UnboundedReceiverStream::new(chunk_rx).flat_map({
                let id: Arc<str> = Arc::from(id.as_str());
                let model_name: Arc<str> = Arc::from(model_name.as_str());
                move |segment| {
                    let id = Arc::clone(&id);
                    let model_name = Arc::clone(&model_name);
                    stream::iter(chunk_events(segment, move |piece| {
                        Ok(Event::default()
                            .json_data(OpenAiChatStreamChunk {
                                choices: [OpenAiChatChoice {
                                    delta: OpenAiDeltaContent { content: piece },
                                    finish_reason: None,
                                    index: 0,
                                }],
                                created,
                                id: &id,
                                model: &model_name,
                                object: "chat.completion.chunk",
                            })
                            .unwrap())
                    }))
                }
            });

        let completion_stream = futures::stream::once({
            let id = id.clone();
            let model_name = model_name.clone();
            async move {
                match result_rx.await {
                    Ok(Ok((_, workflow_details))) => {
                        let mut payload = serde_json::json!({
                            "id": id.clone(),
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [ {
                                "index": 0,
                                "delta": serde_json::json!({}),
                                "finish_reason": "stop"
                            } ]
                        });
                        insert_workflow_field(&mut payload, &workflow_details);
                        Ok::<Event, Infallible>(Event::default().json_data(payload).unwrap())
                    }
                    Ok(Err(err)) => {
                        let error_message = err.error.to_string();
                        let payload = serde_json::json!({
                            "id": id.clone(),
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [ {
                                "index": 0,
                                "delta": serde_json::json!({}),
                                "finish_reason": "error"
                            } ],
                            "error": error_message,
                        });
                        Ok::<Event, Infallible>(Event::default().json_data(payload).unwrap())
                    }
                    Err(_) => {
                        let payload = serde_json::json!({
                            "id": id.clone(),
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [ {
                                "index": 0,
                                "delta": serde_json::json!({}),
                                "finish_reason": "error"
                            } ],
                            "error": "stream cancelled",
                        });
                        Ok::<Event, Infallible>(Event::default().json_data(payload).unwrap())
                    }
                }
            }
        });

        let done_stream = futures::stream::once(async {
            Ok::<Event, Infallible>(Event::default().data("[DONE]"))
        });

        let sse_stream = initial_stream
            .chain(chunk_stream)
            .chain(completion_stream)
            .chain(done_stream);

        return Ok(Sse::new(sse_stream).into_response());
    }

    let (response_text, workflow_details) =
        execute_workflow(&state, prompt, include_workflow_details, None).await?;
    let now = chrono::Utc::now();
    let created = now.timestamp();
    let id = format!("chatcmpl_{}", now.timestamp_millis());

    Ok(Json(OpenAiChatCompletionResponse {
        choices: [OpenAiChatCompletionChoice {
            finish_reason: "stop",
            index: 0,
            message: OpenAiChatCompletionMessage {
                content: &response_text,
                role: "assistant",
            },
        }],
        created,
        id: &id,
        model: &model_name,
        object: "chat.completion",
        workflow: workflow_details,
    })
    .into_response())
}

async fn openai_completions(
    State(state): State<SharedState>,
    Json(req): Json<CompletionRequest>,
) -> Result<Response, AppError> {
    tracing::info!(
        "Received OpenAI completions request, stream: {:?}",
        req.stream
    );

    let prompt = req.prompt.into_prompt();
    let model_name = req.model.unwrap_or_else(|| "chorus".to_string());
    let stream_enabled = req.stream.unwrap_or(false);
    let include_workflow_details = req.include_workflow.unwrap_or(false);

    if stream_enabled {
        let now = chrono::Utc::now();
        let created = now.timestamp();
        let id = format!("cmpl_{}", now.timestamp_millis());
        let (chunk_rx, result_rx) =
            spawn_streaming_workflow(state.clone(), prompt, include_workflow_details);

        let chunk_stream =
            UnboundedReceiverStream::new(chunk_rx).flat_map({
                let id: Arc<str> = Arc::from(id.as_str());
                let model_name: Arc<str> = Arc::from(model_name.as_str());
                move |segment| {
                    let id = Arc::clone(&id);
                    let model_name = Arc::clone(&model_name);
                    stream::iter(chunk_events(segment, move |piece| {
                        Ok(Event::default()
                            .json_data(OpenAiCompletionStreamChunk {
                                choices: [OpenAiCompletionChoice {
                                    finish_reason: None,
                                    index: 0,
                                    logprobs: None,
                                    text: piece,
                                }],
                                created,
                                id: &id,
                                model: &model_name,
                                object: "text_completion",
                            })
                            .unwrap())
                    }))
                }
            });

        let completion_stream = futures::stream::once({
            let id = id.clone();
            let model_name = model_name.clone();
            async move {
                match result_rx.await {
                    Ok(Ok((_, workflow_details))) => {
                        let mut payload = serde_json::json!({
                            "id": id.clone(),
                            "object": "text_completion",
                            "created": created,
                            "model": model_name,
                            "choices": [ {
                                "text": "",
                                "index": 0,
                                "logprobs": serde_json::Value::Null,
                                "finish_reason": "stop"
                            } ]
                        });
                        insert_workflow_field(&mut payload, &workflow_details);
                        Ok::<Event, Infallible>(Event::default().json_data(payload).unwrap())
                    }
                    Ok(Err(err)) => {
                        let error_message = err.error.to_string();
                        let payload = serde_json::json!({
                            "id": id.clone(),
                            "object": "text_completion",
                            "created": created,
                            "model": model_name,
                            "choices": [ {
                                "text": "",
                                "index": 0,
                                "logprobs": serde_json::Value::Null,
                                "finish_reason": "error"
                            } ],
                            "error": error_message,
                        });
                        Ok::<Event, Infallible>(Event::default().json_data(payload).unwrap())
                    }
                    Err(_) => {
                        let payload = serde_json::json!({
                            "id": id.clone(),
                            "object": "text_completion",
                            "created": created,
                            "model": model_name,
                            "choices": [ {
                                "text": "",
                                "index": 0,
                                "logprobs": serde_json::Value::Null,
                                "finish_reason": "error"
                            } ],
                            "error": "stream cancelled",
                        });
                        Ok::<Event, Infallible>(Event::default().json_data(payload).unwrap())
                    }
                }
            }
        });

        let done_stream = futures::stream::once(async {
            Ok::<Event, Infallible>(Event::default().data("[DONE]"))
        });

        let sse_stream = chunk_stream.chain(completion_stream).chain(done_stream);
        return Ok(Sse::new(sse_stream).into_response());
    }

    let (response_text, workflow_details) =
        execute_workflow(&state, prompt, include_workflow_details, None).await?;

    let now = chrono::Utc::now();
    let created = now.timestamp();
    let id = format!("cmpl_{}", now.timestamp_millis());

    Ok(Json(OpenAiCompletionResponse {
        choices: [OpenAiCompletionResponseChoice {
            finish_reason: "stop",
            index: 0,
            logprobs: None,
            text: &response_text,
        }],
        created,
        id: &id,
        model: &model_name,
        object: "text_completion",
        workflow: workflow_details,
    })
    .into_response())
}

fn extract_text_value(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::Bool(b) => Some(b.to_string()),
        Value::Number(n) => Some(n.to_string()),
        Value::String(s) => {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        }
        Value::Array(items) => {
            let mut parts = Vec::new();
            for item in items {
                if let Some(text) = extract_text_value(item) {
                    if !text.is_empty() {
                        parts.push(text);
                    }
                }
            }
            if parts.is_empty() {
                None
            } else {
                Some(parts.join("\n"))
            }
        }
        Value::Object(map) => {
            for key in ["text", "input_text", "value", "output_text"] {
                if let Some(Value::String(s)) = map.get(key) {
                    let trimmed = s.trim();
                    if !trimmed.is_empty() {
                        return Some(trimmed.to_string());
                    }
                }
            }
            if let Some(content) = map.get("content") {
                if let Some(text) = extract_text_value(content) {
                    if !text.is_empty() {
                        return Some(text);
                    }
                }
            }
            if let Some(parts) = map.get("parts") {
                if let Some(text) = extract_text_value(parts) {
                    if !text.is_empty() {
                        return Some(text);
                    }
                }
            }
            if let Some(messages) = map.get("messages") {
                if let Some(text) = extract_text_value(messages) {
                    if !text.is_empty() {
                        return Some(text);
                    }
                }
            }
            None
        }
    }
}

fn extract_message_text(value: &Value) -> Option<String> {
    if let Value::Object(map) = value {
        let role = match map.get("role").and_then(|v| v.as_str()) {
            Some(role) => role,
            None => return None,
        };
        if let Some(content) = map.get("content") {
            if let Some(text) = extract_text_value(content) {
                if text.is_empty() {
                    return None;
                }
                return Some(format!("{}: {}", role, text));
            }
        }
        if let Some(Value::String(text)) = map.get("text") {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                return None;
            }
            return Some(format!("{}: {}", role, trimmed));
        }
    }
    None
}

fn extract_prompt_from_responses_body(payload: &Value) -> Option<String> {
    let mut segments: Vec<String> = Vec::new();

    if let Some(Value::String(instr)) = payload.get("instructions") {
        let trimmed = instr.trim();
        if !trimmed.is_empty() {
            segments.push(format!("system: {}", trimmed));
        }
    }

    if let Some(Value::Array(messages)) = payload.get("messages") {
        for msg in messages {
            if let Some(text) = extract_message_text(msg) {
                segments.push(text);
            } else if let Some(text) = extract_text_value(msg) {
                if !text.is_empty() {
                    segments.push(text);
                }
            }
        }
    }

    if let Some(input) = payload.get("input") {
        match input {
            Value::String(s) => {
                let trimmed = s.trim();
                if !trimmed.is_empty() {
                    segments.push(trimmed.to_string());
                }
            }
            Value::Array(items) => {
                for item in items {
                    if let Some(text) = extract_message_text(item) {
                        segments.push(text);
                    } else if let Some(text) = extract_text_value(item) {
                        if !text.is_empty() {
                            segments.push(text);
                        }
                    }
                }
            }
            Value::Object(_) => {
                if let Some(text) = extract_message_text(input) {
                    segments.push(text);
                } else if let Some(text) = extract_text_value(input) {
                    if !text.is_empty() {
                        segments.push(text);
                    }
                }
            }
            _ => {}
        }
    }

    for key in ["prompt", "input_text"] {
        if let Some(value) = payload.get(key) {
            if let Some(text) = extract_text_value(value) {
                if !text.is_empty() {
                    segments.push(text);
                }
            }
        }
    }

    if segments.is_empty() {
        None
    } else {
        Some(segments.join("\n"))
    }
}

async fn responses(
    State(state): State<SharedState>,
    Json(req): Json<Value>,
) -> Result<Response, AppError> {
    let model_name = req
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("chorus")
        .to_string();

    let stream_requested = matches!(
        req.get("stream"),
        Some(Value::Bool(true)) | Some(Value::Object(_))
    );

    let include_workflow_details = req
        .get("include_workflow")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    tracing::info!(
        "Received v1/responses request for model {}, stream: {}, include_workflow: {}",
        model_name.as_str(),
        stream_requested,
        include_workflow_details
    );

    let prompt = extract_prompt_from_responses_body(&req).ok_or_else(|| {
        AppError::bad_request(anyhow::anyhow!(
            "invalid request: missing input/messages/prompt/instructions"
        ))
    })?;

    let prompt_len = prompt.len();

    if stream_requested {
        let now = chrono::Utc::now();
        let created = now.timestamp();
        let resp_id = format!("resp_{}", now.timestamp_millis());
        let msg_id = format!("msg_{}", now.timestamp_millis());
        let (chunk_rx, result_rx) =
            spawn_streaming_workflow(state.clone(), prompt, include_workflow_details);

        let initial_stream = futures::stream::once({
            let resp_id = resp_id.clone();
            let msg_id = msg_id.clone();
            let model_name = model_name.clone();
            let created = created;
            async move {
                let payload = serde_json::json!({
                    "id": resp_id,
                    "object": "response",
                    "created": created,
                    "model": model_name,
                    "status": "in_progress",
                    "output": [ {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "content": Vec::<serde_json::Value>::new()
                    } ],
                    "output_text": "",
                });
                Ok::<Event, Infallible>(
                    Event::default()
                        .event("response.created")
                        .json_data(payload)
                        .unwrap(),
                )
            }
        });

        let content_index = Arc::new(AtomicUsize::new(0));

        let chunk_stream =
            UnboundedReceiverStream::new(chunk_rx).flat_map({
                let resp_id: Arc<str> = Arc::from(resp_id.as_str());
                let model_name: Arc<str> = Arc::from(model_name.as_str());
                let content_index = Arc::clone(&content_index);
                move |segment| {
                    let resp_id = Arc::clone(&resp_id);
                    let model_name = Arc::clone(&model_name);
                    let content_index = Arc::clone(&content_index);
                    stream::iter(chunk_events(segment, move |piece| {
                        let index = content_index.fetch_add(1, Ordering::Relaxed);
                        Ok(Event::default()
                            .event("response.output_text.delta")
                            .json_data(ResponsesDeltaChunk {
                                content_index: index,
                                created,
                                delta: ResponsesDelta {
                                    text: piece,
                                    kind: "output_text.delta",
                                },
                                id: &resp_id,
                                model: &model_name,
                                object: "response.output_text.delta",
                                output_index: 0,
                            })
                            .unwrap())
                    }))
                }
            });

        let completion_stream = futures::stream::once({
            let resp_id = resp_id.clone();
            let msg_id = msg_id.clone();
            let model_name = model_name.clone();
            let prompt_len = prompt_len;
            let created = created;
            async move {
                match result_rx.await {
                    Ok(Ok((response_text, workflow_details))) => {
                        tracing::debug!(
                            "Generated responses stream payload (prompt {} bytes, response {} bytes)",
                            prompt_len,
                            response_text.len()
                        );
                        let mut payload = serde_json::json!({
                            "id": resp_id,
                            "object": "response",
                            "created": created,
                            "model": model_name,
                            "status": "completed",
                            "output": [ {
                                "id": msg_id,
                                "type": "message",
                                "role": "assistant",
                                "content": [ { "type": "output_text", "text": response_text } ]
                            } ],
                            "output_text": response_text,
                        });
                        insert_workflow_field(&mut payload, &workflow_details);
                        Ok::<Event, Infallible>(
                            Event::default()
                                .event("response.completed")
                                .json_data(payload)
                                .unwrap(),
                        )
                    }
                    Ok(Err(err)) => {
                        let error_message = err.error.to_string();
                        tracing::error!(
                            "Responses stream failed after prompt {} bytes: {}",
                            prompt_len,
                            error_message
                        );
                        let payload = serde_json::json!({
                            "id": resp_id,
                            "object": "response.error",
                            "created": created,
                            "model": model_name,
                            "status": "failed",
                            "error": error_message,
                        });
                        Ok::<Event, Infallible>(
                            Event::default()
                                .event("response.error")
                                .json_data(payload)
                                .unwrap(),
                        )
                    }
                    Err(_) => {
                        tracing::warn!(
                            "Responses stream cancelled before completion (prompt {} bytes)",
                            prompt_len
                        );
                        let payload = serde_json::json!({
                            "id": resp_id,
                            "object": "response.error",
                            "created": created,
                            "model": model_name,
                            "status": "failed",
                            "error": "stream cancelled",
                        });
                        Ok::<Event, Infallible>(
                            Event::default()
                                .event("response.error")
                                .json_data(payload)
                                .unwrap(),
                        )
                    }
                }
            }
        });

        let done_stream = futures::stream::once(async {
            Ok::<Event, Infallible>(Event::default().data("[DONE]"))
        });

        let sse_stream = initial_stream
            .chain(chunk_stream)
            .chain(completion_stream)
            .chain(done_stream);

        return Ok(Sse::new(sse_stream).into_response());
    }

    let (response_text, workflow_details) =
        execute_workflow(&state, prompt, include_workflow_details, None).await?;

    tracing::debug!(
        "Generated responses payload (prompt {} bytes, response {} bytes)",
        prompt_len,
        response_text.len()
    );

    let now = chrono::Utc::now();
    let resp_id = format!("resp_{}", now.timestamp_millis());
    let msg_id = format!("msg_{}", now.timestamp_millis());
    let created = now.timestamp();

    Ok(Json(ResponsesResponse {
        id: &resp_id,
        object: "response",
        created,
        model: &model_name,
        status: "completed",
        output: [ResponsesOutputMessage {
            id: &msg_id,
            kind: "message",
            role: "assistant",
            content: [ResponsesOutputContent {
                kind: "output_text",
                text: &response_text,
            }],
        }],
        output_text: &response_text,
        workflow: workflow_details,
    })
    .into_response())
}

#[derive(Serialize)]
struct OpenAiModelEntry<'a> {
    created: i64,
    id: &'a str,
    object: &'static str,
    owned_by: &'static str,
    permission: [Value; 0],
}

#[derive(Serialize)]
struct OpenAiModelList<'a> {
    data: Vec<OpenAiModelEntry<'a>>,
    object: &'static str,
}

#[derive(Serialize)]
struct ModelEntry<'a> {
    model: &'a str,
    modified_at: &'a str,
    name: &'a str,
}

#[derive(Serialize)]
struct ModelList<'a> {
    models: Vec<ModelEntry<'a>>,
}

async fn list_models_openai(State(state): State<SharedState>) -> Response {
    let created = chrono::Utc::now().timestamp();
    let data = state
        .config
        .models
        .iter()
        .map(|m| OpenAiModelEntry {
            created,
            id: &m.name,
            object: "model",
            owned_by: "chorus",
            permission: [],
        })
        .collect();

    Json(OpenAiModelList {
        data,
        object: "list",
    })
    .into_response()
}

async fn list_models(State(state): State<SharedState>) -> Response {
    let modified_at = chrono::Utc::now().to_rfc3339();
    let models = state
        .config
        .models
        .iter()
        .map(|m| ModelEntry {
            model: &m.name,
            modified_at: &modified_at,
            name: &m.name,
        })
        .collect();

    Json(ModelList { models }).into_response()
}

#[cfg(test)]
mod responses_tests {
    use super::extract_prompt_from_responses_body;
    use serde_json::json;

    #[test]
    fn extract_prompt_prefers_instructions_and_input() {
        let payload = json!({
            "instructions": "Be helpful",
            "input": "Say hello"
        });
        assert_eq!(
            extract_prompt_from_responses_body(&payload).unwrap(),
            "system: Be helpful\nSay hello"
        );
    }

    #[test]
    fn extract_prompt_handles_message_arrays() {
        let payload = json!({
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hi there"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Hello!"}
                    ]
                }
            ]
        });
        assert_eq!(
            extract_prompt_from_responses_body(&payload).unwrap(),
            "user: Hi there\nassistant: Hello!"
        );
    }

    #[test]
    fn extract_prompt_handles_messages_field_with_string_content() {
        let payload = json!({
            "messages": [
                {"role": "user", "content": "ping"}
            ]
        });
        assert_eq!(
            extract_prompt_from_responses_body(&payload).unwrap(),
            "user: ping"
        );
    }

    #[test]
    fn extract_prompt_handles_text_blocks() {
        let payload = json!({
            "input": [
                {"type": "text", "text": "First"},
                {"type": "input_text", "text": "Second"}
            ]
        });
        assert_eq!(
            extract_prompt_from_responses_body(&payload).unwrap(),
            "First\nSecond"
        );
    }

    #[test]
    fn extract_prompt_handles_prompt_array() {
        let payload = json!({
            "prompt": [
                "First",
                { "text": "Second" },
                { "content": [{ "text": "Third" }] }
            ]
        });
        assert_eq!(
            extract_prompt_from_responses_body(&payload).unwrap(),
            "First\nSecond\nThird"
        );
    }

    #[test]
    fn extract_prompt_returns_none_when_empty() {
        assert!(extract_prompt_from_responses_body(&json!({})).is_none());
    }

    #[test]
    fn app_error_bad_request_uses_400_status() {
        use axum::http::StatusCode;
        use axum::response::IntoResponse;

        let response = super::AppError::bad_request(anyhow::anyhow!("bad request")).into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}

#[cfg(test)]
mod stream_payload_tests {
    use super::*;

    fn reference_chunk_text(content: &str, max_len: usize) -> Vec<String> {
        if content.is_empty() {
            return vec![String::new()];
        }

        let mut chunks = Vec::new();
        let mut current = String::new();
        let mut current_len = 0usize;

        for ch in content.chars() {
            current_len += ch.len_utf8();
            current.push(ch);

            if ch == '\n' || current_len >= max_len {
                chunks.push(current);
                current = String::new();
                current_len = 0;
            }
        }

        if !current.is_empty() {
            chunks.push(current);
        }

        chunks
    }

    #[test]
    fn chunk_text_matches_reference_implementation() {
        let cases: Vec<String> = vec![
            String::new(),
            "hello world".to_string(),
            "line1\nline2\n".to_string(),
            "\n\n".to_string(),
            "中文测试内容比较长一些".to_string(),
            "x".repeat(500),
            "a\nbbb\nccccc".to_string(),
            "emoji 😀😀😀 tail".to_string(),
            "a".repeat(STREAM_CHUNK_SIZE * 2 + 1),
        ];

        for case in &cases {
            let case = case.as_str();
            for max_len in [1usize, 3, 7, STREAM_CHUNK_SIZE] {
                assert_eq!(
                    chunk_text(case, max_len),
                    reference_chunk_text(case, max_len),
                    "case {:?} with max_len {}",
                    case,
                    max_len
                );
            }
        }
    }

    #[test]
    fn lazy_chunk_events_match_reference_implementation() {
        use std::sync::Mutex;

        let cases: Vec<String> = vec![
            String::new(),
            "hello world".to_string(),
            "line1\nline2\n".to_string(),
            "\n\n".to_string(),
            "中文测试内容比较长一些".to_string(),
            "x".repeat(500),
            "a\nbbb\nccccc".to_string(),
            "emoji 😀😀😀 tail".to_string(),
            "a".repeat(STREAM_CHUNK_SIZE * 2 + 1),
        ];

        for case in &cases {
            for max_len in [1usize, 3, 7, STREAM_CHUNK_SIZE] {
                let collected: Mutex<Vec<String>> = Mutex::new(Vec::new());
                let events = ChunkEvents::new(case.clone(), max_len, |piece: &str| {
                    collected
                        .lock()
                        .expect("collector lock")
                        .push(piece.to_string());
                    Ok(Event::default().data(piece))
                });

                let mut produced = 0usize;
                for event in events {
                    let _ = event.expect("chunk events are infallible");
                    produced += 1;
                }

                let collected = collected.into_inner().expect("collector lock");
                assert_eq!(
                    collected,
                    reference_chunk_text(case, max_len),
                    "case {:?} with max_len {}",
                    case,
                    max_len
                );
                assert_eq!(produced, collected.len());
            }
        }
    }

    #[test]
    fn prompt_building_matches_reference_implementation() {
        let messages = vec![
            Message {
                role: "user".to_string(),
                content: "hi".to_string(),
            },
            Message {
                role: "assistant".to_string(),
                content: "hello\nworld".to_string(),
            },
            Message {
                role: "tool".to_string(),
                content: String::new(),
            },
        ];

        let expected = messages
            .iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n");

        assert_eq!(build_prompt_from_messages(&messages), expected);
        assert_eq!(build_prompt_from_messages(&[]), String::new());
    }

    #[test]
    fn stream_payloads_match_legacy_json_macro_output() {
        let created = 1234i64;

        let chunk = GenerateStreamChunk {
            created_at: "t",
            done: false,
            model: "m",
            response: "r",
        };
        let legacy = serde_json::json!({
            "model": "m",
            "created_at": "t",
            "response": "r",
            "done": false,
        });
        assert_eq!(serde_json::to_string(&chunk).unwrap(), legacy.to_string());

        let chat = ChatStreamChunk {
            created_at: "t",
            done: false,
            message: ChatStreamMessage {
                content: "r",
                role: "assistant",
            },
            model: "m",
        };
        let legacy = serde_json::json!({
            "model": "m",
            "created_at": "t",
            "message": { "role": "assistant", "content": "r" },
            "done": false,
        });
        assert_eq!(serde_json::to_string(&chat).unwrap(), legacy.to_string());

        let openai_chat = OpenAiChatStreamChunk {
            choices: [OpenAiChatChoice {
                delta: OpenAiDeltaContent { content: "r" },
                finish_reason: None,
                index: 0,
            }],
            created,
            id: "id",
            model: "m",
            object: "chat.completion.chunk",
        };
        let legacy = serde_json::json!({
            "id": "id",
            "object": "chat.completion.chunk",
            "created": created,
            "model": "m",
            "choices": [ {
                "index": 0,
                "delta": { "content": "r" },
                "finish_reason": serde_json::Value::Null
            } ]
        });
        assert_eq!(serde_json::to_string(&openai_chat).unwrap(), legacy.to_string());

        let openai_completion = OpenAiCompletionStreamChunk {
            choices: [OpenAiCompletionChoice {
                finish_reason: None,
                index: 0,
                logprobs: None,
                text: "r",
            }],
            created,
            id: "id",
            model: "m",
            object: "text_completion",
        };
        let legacy = serde_json::json!({
            "id": "id",
            "object": "text_completion",
            "created": created,
            "model": "m",
            "choices": [ {
                "text": "r",
                "index": 0,
                "logprobs": serde_json::Value::Null,
                "finish_reason": serde_json::Value::Null
            } ]
        });
        assert_eq!(
            serde_json::to_string(&openai_completion).unwrap(),
            legacy.to_string()
        );

        let responses = ResponsesDeltaChunk {
            content_index: 2,
            created,
            delta: ResponsesDelta {
                text: "r",
                kind: "output_text.delta",
            },
            id: "id",
            model: "m",
            object: "response.output_text.delta",
            output_index: 0,
        };
        let legacy = serde_json::json!({
            "id": "id",
            "object": "response.output_text.delta",
            "created": created,
            "model": "m",
            "output_index": 0,
            "content_index": 2,
            "delta": {
                "type": "output_text.delta",
                "text": "r"
            },
        });
        assert_eq!(serde_json::to_string(&responses).unwrap(), legacy.to_string());
    }

    #[test]
    fn non_stream_payloads_match_legacy_json_macro_output() {
        let created = 1234i64;

        let chat = OpenAiChatCompletionResponse {
            choices: [OpenAiChatCompletionChoice {
                finish_reason: "stop",
                index: 0,
                message: OpenAiChatCompletionMessage {
                    content: "r",
                    role: "assistant",
                },
            }],
            created,
            id: "id",
            model: "m",
            object: "chat.completion",
            workflow: None,
        };
        let legacy = serde_json::json!({
            "id": "id",
            "object": "chat.completion",
            "created": created,
            "model": "m",
            "choices": [ {
                "index": 0,
                "message": { "role": "assistant", "content": "r" },
                "finish_reason": "stop"
            } ],
            "workflow": serde_json::Value::Null,
        });
        assert_eq!(serde_json::to_string(&chat).unwrap(), legacy.to_string());

        let completion = OpenAiCompletionResponse {
            choices: [OpenAiCompletionResponseChoice {
                finish_reason: "stop",
                index: 0,
                logprobs: None,
                text: "r",
            }],
            created,
            id: "id",
            model: "m",
            object: "text_completion",
            workflow: None,
        };
        let legacy = serde_json::json!({
            "id": "id",
            "object": "text_completion",
            "created": created,
            "model": "m",
            "choices": [ {
                "text": "r",
                "index": 0,
                "logprobs": serde_json::Value::Null,
                "finish_reason": "stop"
            } ],
            "workflow": serde_json::Value::Null,
        });
        assert_eq!(serde_json::to_string(&completion).unwrap(), legacy.to_string());

        let responses = ResponsesResponse {
            id: "resp_1",
            object: "response",
            created,
            model: "m",
            status: "completed",
            output: [ResponsesOutputMessage {
                id: "msg_1",
                kind: "message",
                role: "assistant",
                content: [ResponsesOutputContent {
                    kind: "output_text",
                    text: "r",
                }],
            }],
            output_text: "r",
            workflow: None,
        };
        let legacy = serde_json::json!({
            "id": "resp_1",
            "object": "response",
            "created": created,
            "model": "m",
            "status": "completed",
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [ { "type": "output_text", "text": "r" } ]
                }
            ],
            "output_text": "r",
            "workflow": serde_json::Value::Null,
        });
        assert_eq!(
            serde_json::to_value(&responses).unwrap(),
            serde_json::to_value(&legacy).unwrap()
        );

        let details = WorkflowExecutionDetails {
            analyzer: crate::workflow::AnalyzerDetails {
                model: "m".to_string(),
                temperature: 1.0,
                auto_temperature: false,
            },
            workers: Vec::new(),
            selector: None,
            synthesizer: None,
        };
        let with_workflow = OpenAiChatCompletionResponse {
            choices: [OpenAiChatCompletionChoice {
                finish_reason: "stop",
                index: 0,
                message: OpenAiChatCompletionMessage {
                    content: "r",
                    role: "assistant",
                },
            }],
            created,
            id: "id",
            model: "m",
            object: "chat.completion",
            workflow: Some(details),
        };
        let legacy = serde_json::json!({
            "id": "id",
            "object": "chat.completion",
            "created": created,
            "model": "m",
            "choices": [ {
                "index": 0,
                "message": { "role": "assistant", "content": "r" },
                "finish_reason": "stop"
            } ],
            "workflow": {
                "analyzer": { "model": "m", "temperature": 1.0, "auto_temperature": false },
                "workers": [],
            },
        });
        assert_eq!(
            serde_json::to_value(with_workflow).unwrap(),
            serde_json::to_value(&legacy).unwrap()
        );
    }
}

// 错误处理
pub struct AppError {
    status: StatusCode,
    error: anyhow::Error,
}

impl AppError {
    pub fn new(status: StatusCode, err: impl Into<anyhow::Error>) -> Self {
        Self {
            status,
            error: err.into(),
        }
    }

    pub fn bad_request(err: impl Into<anyhow::Error>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, err)
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        tracing::error!(
            status = %self.status,
            error = %self.error,
            "Application error"
        );

        (
            self.status,
            Json(serde_json::json!({
                "error": self.error.to_string()
            })),
        )
            .into_response()
    }
}

impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, err)
    }
}

#[cfg(test)]
mod auth_tests {
    use super::*;
    use crate::config::{
        ModelConfig, ServerConfig, TimeoutConfig, WorkflowConfig, WorkflowModelTarget,
        WorkflowPlan, WorkflowWorker,
    };
    use std::collections::HashMap;

    fn test_config(api_key: Option<&str>) -> Config {
        let target = |name: &str| WorkflowModelTarget {
            model: name.to_string(),
            temperature: None,
            auto_temperature: None,
        };

        Config {
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 0,
                api_key: api_key.map(str::to_string),
            },
            models: vec![ModelConfig {
                name: "m1".to_string(),
                api_base: "http://127.0.0.1:1/v1".to_string(),
                api_key: "upstream-key".to_string(),
                temperature: None,
                auto_temperature: None,
            }],
            workflow_integration: WorkflowPlan {
                analyzer: target("m1"),
                workers: vec![WorkflowWorker::Model(target("m1"))],
                synthesizer: Some(target("m1")),
                selector: None,
                nested_worker_depth: None,
            },
            workflow: WorkflowConfig {
                timeouts: TimeoutConfig {
                    analyzer_timeout_secs: 5,
                    worker_timeout_secs: 5,
                    synthesizer_timeout_secs: 5,
                },
                domains: HashMap::new(),
                worker_concurrency: None,
                retry: Default::default(),
            },
        }
    }

    async fn spawn_test_server(api_key: Option<&str>) -> String {
        let config = Arc::new(test_config(api_key));
        let state = Arc::new(AppState {
            workflow_engine: WorkflowEngine::from_shared(Arc::clone(&config)),
            config,
            health_body: Bytes::from_static(b"{\"status\":\"ok\"}"),
        });
        let router = build_router(state);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, router).await;
        });

        format!("http://{}", addr)
    }

    fn client() -> reqwest::Client {
        reqwest::Client::new()
    }

    #[tokio::test]
    async fn requests_are_allowed_when_no_api_key_is_configured() {
        let base = spawn_test_server(None).await;

        let tags = client()
            .get(format!("{}/api/tags", base))
            .send()
            .await
            .unwrap();
        assert_eq!(tags.status(), 200);

        let models = client()
            .get(format!("{}/v1/models", base))
            .send()
            .await
            .unwrap();
        assert_eq!(models.status(), 200);
    }

    #[tokio::test]
    async fn api_routes_require_the_configured_key() {
        let base = spawn_test_server(Some("top-secret")).await;
        let url = format!("{}/api/tags", base);

        let missing = client().get(&url).send().await.unwrap();
        assert_eq!(missing.status(), 401);
        assert_eq!(
            missing
                .headers()
                .get("www-authenticate")
                .expect("missing WWW-Authenticate header"),
            "Bearer"
        );

        let wrong = client()
            .get(&url)
            .header("Authorization", "Bearer nope")
            .send()
            .await
            .unwrap();
        assert_eq!(wrong.status(), 401);

        let bearer = client()
            .get(&url)
            .header("Authorization", "Bearer top-secret")
            .send()
            .await
            .unwrap();
        assert_eq!(bearer.status(), 200);

        let lowercase_scheme = client()
            .get(&url)
            .header("Authorization", "bearer top-secret")
            .send()
            .await
            .unwrap();
        assert_eq!(lowercase_scheme.status(), 200);

        let x_api_key = client()
            .get(&url)
            .header("x-api-key", "top-secret")
            .send()
            .await
            .unwrap();
        assert_eq!(x_api_key.status(), 200);
    }

    #[tokio::test]
    async fn health_endpoint_stays_public_when_auth_is_enabled() {
        let base = spawn_test_server(Some("top-secret")).await;

        let response = client().get(format!("{}/", base)).send().await.unwrap();
        assert_eq!(response.status(), 200);
    }

    #[tokio::test]
    async fn cors_preflight_is_not_blocked_by_auth() {
        let base = spawn_test_server(Some("top-secret")).await;

        let response = client()
            .request(reqwest::Method::OPTIONS, format!("{}/api/chat", base))
            .header("Origin", "http://localhost:3000")
            .header("Access-Control-Request-Method", "POST")
            .header("Access-Control-Request-Headers", "authorization,content-type")
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), 200);
        assert!(response
            .headers()
            .contains_key("access-control-allow-origin"));
    }

    #[tokio::test]
    async fn invalid_credentials_are_rejected_before_running_the_workflow() {
        let base = spawn_test_server(Some("top-secret")).await;

        let response = client()
            .post(format!("{}/api/generate", base))
            .json(&serde_json::json!({"model": "chorus", "prompt": "hi"}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 401);

        let body: serde_json::Value = response.json().await.unwrap();
        assert!(body.get("error").is_some());
    }

    #[test]
    fn constant_time_equality_matches_plain_comparison() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(constant_time_eq(b"", b""));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"ab"));
        assert!(!constant_time_eq(b"", b"a"));
    }

    #[test]
    fn bearer_token_parsing_is_case_insensitive_and_strict() {
        assert_eq!(bearer_token("Bearer secret"), Some("secret"));
        assert_eq!(bearer_token("bearer secret"), Some("secret"));
        assert_eq!(bearer_token("BEARER  secret "), Some("secret"));
        assert_eq!(bearer_token("Basic secret"), None);
        assert_eq!(bearer_token("Bearer"), None);
        assert_eq!(bearer_token("Bearer   "), None);
    }
}

#[cfg(test)]
mod disconnect_tests {
    use super::*;
    use crate::config::{
        ModelConfig, ServerConfig, TimeoutConfig, WorkflowConfig, WorkflowModelTarget,
        WorkflowPlan, WorkflowWorker,
    };
    use axum::response::Response as AxumResponse;
    use std::collections::HashMap;
    use std::time::Duration;
    use tokio::sync::mpsc as tokio_mpsc;

    struct UpstreamDropSignal(Option<tokio_mpsc::UnboundedSender<()>>);

    impl Drop for UpstreamDropSignal {
        fn drop(&mut self) {
            if let Some(tx) = self.0.take() {
                let _ = tx.send(());
            }
        }
    }

    #[derive(Clone)]
    struct UpstreamState {
        dropped: tokio_mpsc::UnboundedSender<()>,
        infinite: bool,
    }

    async fn upstream_handler(
        State(state): State<UpstreamState>,
        Json(body): Json<Value>,
    ) -> AxumResponse {
        let streaming = body.get("stream").and_then(Value::as_bool).unwrap_or(false);

        if !streaming {
            return Json(serde_json::json!({
                "choices": [{
                    "index": 0,
                    "message": { "role": "assistant", "content": "worker answer" },
                    "finish_reason": "stop"
                }]
            }))
            .into_response();
        }

        let infinite = state.infinite;
        let frames = futures::stream::unfold(
            (UpstreamDropSignal(Some(state.dropped)), 0u64),
            move |(signal, tick)| async move {
                if !infinite {
                    if tick == 3 {
                        return Some((
                            Ok::<_, Infallible>(Bytes::from_static(b"data: [DONE]\n\n")),
                            (signal, tick + 1),
                        ));
                    }
                    if tick >= 4 {
                        return None;
                    }
                }

                tokio::time::sleep(Duration::from_millis(25)).await;
                let payload = serde_json::json!({
                    "choices": [{ "index": 0, "delta": { "content": "x" } }]
                });
                let frame = format!("data: {}\n\n", payload);
                Some((Ok::<_, Infallible>(Bytes::from(frame)), (signal, tick + 1)))
            },
        );

        let mut response = Response::new(axum::body::Body::from_stream(frames));
        response.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream"),
        );
        response
    }

    async fn spawn_upstream(infinite: bool) -> (String, tokio_mpsc::UnboundedReceiver<()>) {
        let (dropped_tx, dropped_rx) = tokio_mpsc::unbounded_channel();
        let state = UpstreamState {
            dropped: dropped_tx,
            infinite,
        };
        let router = Router::new()
            .route("/v1/chat/completions", post(upstream_handler))
            .with_state(state);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, router).await;
        });

        (format!("http://{}", addr), dropped_rx)
    }

    fn test_config(api_base: &str) -> Config {
        let target = |name: &str| WorkflowModelTarget {
            model: name.to_string(),
            temperature: None,
            auto_temperature: None,
        };

        Config {
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 0,
                api_key: None,
            },
            models: vec![ModelConfig {
                name: "m1".to_string(),
                api_base: api_base.to_string(),
                api_key: "sk-test".to_string(),
                temperature: Some(0.5),
                auto_temperature: None,
            }],
            workflow_integration: WorkflowPlan {
                analyzer: target("m1"),
                workers: vec![WorkflowWorker::Model(target("m1"))],
                synthesizer: Some(target("m1")),
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
                retry: Default::default(),
            },
        }
    }

    async fn spawn_chorus(config: Config) -> String {
        let config = Arc::new(config);
        let state = Arc::new(AppState {
            workflow_engine: WorkflowEngine::from_shared(Arc::clone(&config)),
            config,
            health_body: Bytes::from_static(b"{\"status\":\"ok\"}"),
        });
        let router = build_router(state);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, router).await;
        });

        format!("http://{}", addr)
    }

    fn chat_request(chorus: &str) -> reqwest::RequestBuilder {
        reqwest::Client::new()
            .post(format!("{}/api/chat", chorus))
            .json(&serde_json::json!({
                "model": "chorus",
                "messages": [{ "role": "user", "content": "hi" }],
                "stream": true
            }))
    }

    #[tokio::test]
    async fn disconnecting_the_sse_client_cancels_upstream_work() {
        let (upstream, mut dropped) = spawn_upstream(true).await;
        let chorus = spawn_chorus(test_config(&format!("{}/v1", upstream))).await;

        let response = chat_request(&chorus).send().await.unwrap();
        assert_eq!(response.status(), 200);

        let mut stream = response.bytes_stream();
        for _ in 0..3 {
            tokio::time::timeout(Duration::from_secs(5), stream.next())
                .await
                .expect("Chorus should stream tokens")
                .expect("Chorus stream should stay open")
                .expect("stream chunk should be readable");
        }

        drop(stream);

        tokio::time::timeout(Duration::from_secs(5), dropped.recv())
            .await
            .expect("upstream request should be cancelled after the client disconnects")
            .expect("upstream drop signal channel closed unexpectedly");
    }

    #[tokio::test]
    async fn fully_consumed_streams_still_complete() {
        let (upstream, _dropped) = spawn_upstream(false).await;
        let chorus = spawn_chorus(test_config(&format!("{}/v1", upstream))).await;

        let response = chat_request(&chorus).send().await.unwrap();
        assert_eq!(response.status(), 200);

        let body = tokio::time::timeout(Duration::from_secs(5), response.text())
            .await
            .expect("stream should finish")
            .unwrap();

        assert!(
            body.contains("\"done\":true"),
            "final event missing done flag: {}",
            body
        );
        assert!(
            body.contains("\"content\":\"x\""),
            "streamed tokens missing from response: {}",
            body
        );
        assert!(!body.contains("\"error\""), "unexpected error: {}", body);
    }
}
