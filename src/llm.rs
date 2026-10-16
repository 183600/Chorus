use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::fmt::Write as _;
use std::time::Duration;
use tokio::sync::mpsc::UnboundedSender;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub content: String,
    pub role: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ChatResponse {
    pub id: Option<String>,
    pub object: Option<String>,
    pub created: Option<i64>,
    pub model: Option<String>,
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct Choice {
    pub index: i32,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct Usage {
    pub prompt_tokens: Option<i32>,
    pub completion_tokens: Option<i32>,
    pub total_tokens: Option<i32>,
}

#[derive(Debug)]
pub struct CompletionResult {
    pub content: String,
    pub streamed: bool,
}

/// Error raised while producing a completion.
///
/// Carries whether any content had already been forwarded to the caller so the
/// retry loop can decide if the request may be safely retried without
/// duplicating partial output.
#[derive(Debug)]
pub(crate) struct StreamFailure {
    pub(crate) error: anyhow::Error,
    pub(crate) streamed: bool,
}

impl StreamFailure {
    fn is_retryable(&self) -> bool {
        self.error
            .downcast_ref::<reqwest::Error>()
            .map(is_retryable_reqwest_error)
            .unwrap_or(false)
    }
}

impl From<reqwest::Error> for StreamFailure {
    fn from(error: reqwest::Error) -> Self {
        Self {
            error: error.into(),
            streamed: false,
        }
    }
}

impl From<serde_json::Error> for StreamFailure {
    fn from(error: serde_json::Error) -> Self {
        Self {
            error: error.into(),
            streamed: false,
        }
    }
}

impl From<anyhow::Error> for StreamFailure {
    fn from(error: anyhow::Error) -> Self {
        Self {
            error,
            streamed: false,
        }
    }
}

#[derive(Serialize)]
#[allow(dead_code)]
struct ChatCompletionRequest<'a> {
    messages: &'a [ChatMessage],
    model: &'a str,
    stream: bool,
    temperature: Option<f64>,
}

#[derive(Serialize)]
struct BorrowedChatMessage<'a> {
    content: &'a str,
    role: &'a str,
}

#[derive(Serialize)]
struct BorrowedChatCompletionRequest<'a> {
    messages: &'a [BorrowedChatMessage<'a>],
    model: &'a str,
    stream: bool,
    temperature: Option<f64>,
}

/// Policy controlling automatic retries for transient upstream failures.
///
/// Retries only apply to failures that happen before any response content has
/// been consumed: connection/timeout errors, HTTP 408/429/5xx responses and
/// response body read failures. Once tokens have been forwarded to a caller
/// (streaming), the request is never retried so partial output cannot be
/// duplicated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
}

impl RetryPolicy {
    pub const fn disabled() -> Self {
        Self {
            max_retries: 0,
            initial_backoff: Duration::ZERO,
            max_backoff: Duration::ZERO,
        }
    }

    pub const fn new(max_retries: u32, initial_backoff: Duration, max_backoff: Duration) -> Self {
        Self {
            max_retries,
            initial_backoff,
            max_backoff,
        }
    }

    /// Exponential backoff for the n-th retry (1-based), capped at `max_backoff`.
    pub fn backoff_for_retry(&self, retry_number: u32) -> Duration {
        let exponent = retry_number.saturating_sub(1).min(16);
        let factor = 1u32 << exponent;
        self.initial_backoff
            .saturating_mul(factor)
            .min(self.max_backoff)
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 2,
            initial_backoff: Duration::from_millis(500),
            max_backoff: Duration::from_secs(8),
        }
    }
}

#[derive(Clone)]
pub struct LLMClient {
    client: Client,
    api_base: String,
    authorization: reqwest::header::HeaderValue,
    chat_completions_url: String,
    retry: RetryPolicy,
    timeout: Duration,
}

impl LLMClient {
    pub fn new(api_base: String, api_key: String, timeout_secs: u64) -> Result<Self> {
        Self::new_with_retry(api_base, api_key, timeout_secs, RetryPolicy::disabled())
    }

    pub fn new_with_retry(
        api_base: String,
        api_key: String,
        timeout_secs: u64,
        retry: RetryPolicy,
    ) -> Result<Self> {
        let client = Client::builder()
            .build()
            .with_context(|| format!("Failed to build HTTP client for {}", api_base))?;

        let chat_completions_url =
            format!("{}/chat/completions", api_base.trim_end_matches('/'));

        let authorization = reqwest::header::HeaderValue::from_str(&format!("Bearer {}", api_key))
            .with_context(|| format!("Invalid API key for {}", api_base))?;

        Ok(Self {
            client,
            api_base,
            authorization,
            chat_completions_url,
            retry,
            timeout: Duration::from_secs(timeout_secs),
        })
    }

    #[allow(dead_code)]
    pub async fn chat_completion(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        temperature: Option<f32>,
    ) -> Result<String> {
        let result = self
            .chat_completion_with_stream(model, messages, temperature, None)
            .await?;
        Ok(result.content)
    }

    /// Sends a single-user-message chat completion.
    #[allow(dead_code)]
    pub async fn chat_completion_prompt(
        &self,
        model: &str,
        prompt: &str,
        temperature: Option<f32>,
    ) -> Result<String> {
        let result = self
            .chat_completion_prompt_with_stream(model, prompt, temperature, None)
            .await?;
        Ok(result.content)
    }

    #[allow(dead_code)]
    pub async fn chat_completion_prompt_with_stream(
        &self,
        model: &str,
        prompt: &str,
        temperature: Option<f32>,
        stream: Option<UnboundedSender<String>>,
    ) -> Result<CompletionResult> {
        let timeout = self.timeout;
        self.chat_completion_prompt_with_timeout_and_stream_tracked(
            model, prompt, temperature, timeout, stream,
        )
        .await
        .map_err(|failure| failure.error)
    }

    /// Like [`LLMClient::chat_completion_prompt`], but overrides the request
    /// timeout.
    pub(crate) async fn chat_completion_prompt_with_timeout(
        &self,
        model: &str,
        prompt: &str,
        temperature: Option<f32>,
        timeout: Duration,
    ) -> Result<String> {
        let result = self
            .chat_completion_prompt_with_timeout_and_stream_tracked(
                model, prompt, temperature, timeout, None,
            )
            .await
            .map_err(|failure| failure.error)?;
        Ok(result.content)
    }

    /// Reports whether any content had already been forwarded when the call
    /// failed.
    pub(crate) async fn chat_completion_prompt_with_timeout_and_stream_tracked(
        &self,
        model: &str,
        prompt: &str,
        temperature: Option<f32>,
        timeout: Duration,
        stream: Option<UnboundedSender<String>>,
    ) -> std::result::Result<CompletionResult, StreamFailure> {
        let messages = [BorrowedChatMessage {
            content: prompt,
            role: "user",
        }];
        let request = BorrowedChatCompletionRequest {
            messages: &messages,
            model,
            stream: stream.is_some(),
            temperature: temperature.map(f64::from),
        };
        self.send_chat_request(&request, model, timeout, stream).await
    }

    #[allow(dead_code)]
    pub async fn chat_completion_with_stream(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        temperature: Option<f32>,
        stream: Option<UnboundedSender<String>>,
    ) -> Result<CompletionResult> {
        let request = ChatCompletionRequest {
            messages: &messages,
            model,
            stream: stream.is_some(),
            temperature: temperature.map(f64::from),
        };
        self.send_chat_request(&request, model, self.timeout, stream)
            .await
            .map_err(|failure| failure.error)
    }

    async fn send_chat_request<R: Serialize>(
        &self,
        request: &R,
        model: &str,
        timeout: Duration,
        stream: Option<UnboundedSender<String>>,
    ) -> std::result::Result<CompletionResult, StreamFailure> {
        let stream_was_requested = stream.is_some();

        tracing::debug!(
            "Calling LLM API: {} with model: {}",
            self.chat_completions_url,
            model
        );
        if tracing::enabled!(tracing::Level::DEBUG) {
            tracing::debug!("Request body: {}", serde_json::to_string_pretty(request)?);
        }

        let body = Bytes::from(serde_json::to_vec(request)?);
        let mut attempt: u32 = 0;

        loop {
            attempt += 1;

            let response = match self
                .client
                .post(&self.chat_completions_url)
                .header(reqwest::header::AUTHORIZATION, self.authorization.clone())
                .header(
                    reqwest::header::CONTENT_TYPE,
                    reqwest::header::HeaderValue::from_static("application/json"),
                )
                .timeout(timeout)
                .body(body.clone())
                .send()
                .await
            {
                Ok(response) => response,
                Err(err) => {
                    if self.can_retry(attempt) && is_retryable_reqwest_error(&err) {
                        self.wait_before_retry(attempt, None, model, &err.to_string())
                            .await;
                        continue;
                    }
                    return Err(err.into());
                }
            };

            let status = response.status();
            if !status.is_success() {
                let retry_after = parse_retry_after(response.headers());
                let error_text = response.text().await;

                if self.can_retry(attempt) && is_retryable_status(status) {
                    let error_text = error_text.unwrap_or_default();
                    self.wait_before_retry(
                        attempt,
                        retry_after,
                        model,
                        &format!("HTTP {}: {}", status, error_text),
                    )
                    .await;
                    continue;
                }

                let error_text = error_text?;
                return Err(anyhow!(
                    "LLM API request failed with status {}: {}",
                    status,
                    error_text
                )
                .into());
            }

            if stream_was_requested && response_is_event_stream(&response) {
                match self.consume_event_stream(response, stream.clone()).await {
                    Ok(result) => return Ok(result),
                    Err(failure) => {
                        if !failure.streamed && self.can_retry(attempt) && failure.is_retryable() {
                            self.wait_before_retry(
                                attempt,
                                None,
                                model,
                                &failure.error.to_string(),
                            )
                            .await;
                            continue;
                        }
                        return Err(failure);
                    }
                }
            }

            let body = match response.bytes().await {
                Ok(bytes) => bytes,
                Err(err) => {
                    if self.can_retry(attempt) && is_retryable_reqwest_error(&err) {
                        self.wait_before_retry(attempt, None, model, &err.to_string())
                            .await;
                        continue;
                    }
                    return Err(err.into());
                }
            };

            // Fast path: the common OpenAI-compatible shape is parsed with a
            // borrowed view so no `Value` tree is built. Any other shape is
            // deferred to the tolerant `Value`-based path below.
            if let Ok(text) = std::str::from_utf8(&body) {
                if let Some(content) = fast_completion_text(text) {
                    if let Some(sender) = stream.as_ref() {
                        let _ = sender.send(content.clone());
                    }
                    return Ok(CompletionResult {
                        content,
                        streamed: stream_was_requested,
                    });
                }
            }

            // Be tolerant to different provider response shapes
            let v: serde_json::Value = match serde_json::from_slice(&body) {
                Ok(value) => value,
                Err(err) => {
                    // Mirrors the message reqwest's `Response::json` produced
                    // before the body was read through `bytes()`.
                    return Err(anyhow!("error decoding response body: {}", err).into());
                }
            };

            if let Some(content) = extract_completion_text(&v) {
                if let Some(sender) = stream.as_ref() {
                    let _ = sender.send(content.clone());
                }
                return Ok(CompletionResult {
                    content,
                    streamed: stream_was_requested,
                });
            }

            if let Some(err_msg) = detect_provider_error(&v) {
                return Err(anyhow!(
                    "LLM provider {} (model {}) returned error: {}",
                    self.api_base,
                    model,
                    err_msg
                )
                .into());
            }

            return Err(anyhow!("LLM response missing content field: {}", v).into());
        }
    }

    fn can_retry(&self, attempt: u32) -> bool {
        attempt <= self.retry.max_retries
    }

    async fn wait_before_retry(
        &self,
        attempt: u32,
        retry_after: Option<Duration>,
        model: &str,
        reason: &str,
    ) {
        let delay = retry_delay(&self.retry, attempt, retry_after);
        tracing::warn!(
            url = %self.chat_completions_url,
            model,
            attempt,
            max_retries = self.retry.max_retries,
            delay_ms = delay.as_millis() as u64,
            reason,
            "Transient LLM API failure, retrying"
        );
        tokio::time::sleep(delay).await;
    }

    async fn consume_event_stream(
        &self,
        response: reqwest::Response,
        stream: Option<UnboundedSender<String>>,
    ) -> std::result::Result<CompletionResult, StreamFailure> {
        let mut final_text = String::new();
        let mut streamed = false;
        let mut buffer = String::new();
        let mut pending = Vec::new();
        let mut cursor = 0usize;
        let mut data_payload = String::new();
        let mut byte_stream = response.bytes_stream();

        while let Some(item) = byte_stream.next().await {
            let chunk = match item {
                Ok(chunk) => chunk,
                Err(error) => {
                    return Err(StreamFailure {
                        error: error.into(),
                        streamed,
                    })
                }
            };
            match std::str::from_utf8(&chunk) {
                Ok(text) if pending.is_empty() => buffer.push_str(text),
                _ => {
                    pending.extend_from_slice(&chunk);
                    append_valid_utf8(&mut pending, &mut buffer);
                }
            }

            loop {
                let Some((idx, sep_len)) = find_event_separator(&buffer[cursor..]) else {
                    break;
                };

                let event = &buffer[cursor..cursor + idx];
                cursor += idx + sep_len;

                if event.trim().is_empty() {
                    continue;
                }

                let event = event.strip_suffix('\r').unwrap_or(event);

                let trimmed = match single_data_payload(event) {
                    Some(payload) => {
                        let payload = payload.trim();
                        if payload.is_empty() {
                            continue;
                        }
                        payload
                    }
                    None => {
                        data_payload.clear();
                        for line in event.lines() {
                            if let Some(rest) = line.strip_prefix("data:") {
                                if !data_payload.is_empty() {
                                    data_payload.push('\n');
                                }
                                data_payload.push_str(rest.trim_start());
                            }
                        }

                        if data_payload.is_empty() {
                            continue;
                        }

                        let payload = data_payload.trim();
                        if payload.is_empty() {
                            continue;
                        }
                        payload
                    }
                };

                if trimmed == "[DONE]" {
                    return Ok(CompletionResult {
                        content: final_text,
                        streamed,
                    });
                }

                if let FastStreamEvent::Text { text, finish } = fast_stream_event(trimmed) {
                    final_text.push_str(text);
                    streamed = true;
                    if let Some(sender) = stream.as_ref() {
                        let _ = sender.send(text.to_string());
                    }
                    if finish.is_some() {
                        return Ok(CompletionResult {
                            content: final_text,
                            streamed,
                        });
                    }
                    continue;
                }

                match serde_json::from_str::<serde_json::Value>(trimmed) {
                    Ok(value) => {
                        if let Some(text) = extract_stream_content(&value)
                            .or_else(|| extract_completion_text(&value))
                        {
                            if !text.is_empty() {
                                final_text.push_str(&text);
                                streamed = true;
                            }
                            if let Some(sender) = stream.as_ref() {
                                let _ = sender.send(text);
                            }
                        }

                        if let Some(reason) = value
                            .get("choices")
                            .and_then(|c| c.get(0))
                            .and_then(|c0| c0.get("finish_reason"))
                            .and_then(|r| r.as_str())
                        {
                            if !reason.is_empty() && reason != "null" {
                                return Ok(CompletionResult {
                                    content: final_text,
                                    streamed,
                                });
                            }
                        }
                    }
                    Err(_) => {
                        if let Some(sender) = stream.as_ref() {
                            let _ = sender.send(trimmed.to_string());
                        }
                        if !trimmed.is_empty() {
                            final_text.push_str(trimmed);
                            streamed = true;
                        }
                    }
                }
            }

            if cursor > 0 {
                buffer.drain(..cursor);
                cursor = 0;
            }
        }

        Ok(CompletionResult {
            content: final_text,
            streamed,
        })
    }
}

/// Common OpenAI-compatible streaming delta shape.
///
/// Anything that does not fit this shape (non-string content, tool calls,
/// message-shaped payloads, ...) is reported as `Unsupported` and handled by
/// the generic `Value` path so behavior stays identical for all providers.
#[derive(Deserialize)]
struct FastStreamChunk<'a> {
    #[serde(default, borrow)]
    choices: Option<FirstChoice<FastStreamChoice<'a>>>,
}

/// Deserializes only the first element of a choices sequence, skipping the
/// remaining ones without allocating.
struct FirstChoice<T>(T);

impl<'de, T: Deserialize<'de>> Deserialize<'de> for FirstChoice<T> {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct FirstChoiceVisitor<T>(std::marker::PhantomData<T>);

        impl<'de, T: Deserialize<'de>> serde::de::Visitor<'de> for FirstChoiceVisitor<T> {
            type Value = FirstChoice<T>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a sequence of choices")
            }

            fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let first = seq.next_element::<T>()?;
                while seq.next_element::<serde::de::IgnoredAny>()?.is_some() {}
                first
                    .map(FirstChoice)
                    .ok_or_else(|| serde::de::Error::custom("empty choices sequence"))
            }
        }

        deserializer.deserialize_seq(FirstChoiceVisitor(std::marker::PhantomData))
    }
}

#[derive(Deserialize)]
struct FastStreamChoice<'a> {
    #[serde(default)]
    delta: Option<FastStreamDelta<'a>>,
    #[serde(default)]
    text: Option<&'a str>,
    #[serde(default)]
    finish_reason: Option<&'a str>,
}

#[derive(Deserialize)]
struct FastStreamDelta<'a> {
    #[serde(default)]
    content: Option<&'a str>,
    #[serde(default)]
    reasoning_content: Option<&'a str>,
    #[serde(default)]
    analysis: Option<&'a str>,
    #[serde(default)]
    reasoning: Option<&'a str>,
}

enum FastStreamEvent<'a> {
    Text {
        text: &'a str,
        finish: Option<&'a str>,
    },
    Unsupported,
}

fn non_empty_str(text: Option<&str>) -> Option<&str> {
    text.filter(|value| !value.is_empty())
}

fn fast_stream_event(payload: &str) -> FastStreamEvent<'_> {
    let parsed: FastStreamChunk<'_> = match serde_json::from_str(payload) {
        Ok(parsed) => parsed,
        Err(_) => return FastStreamEvent::Unsupported,
    };

    let Some(choice) = parsed.choices.as_ref().map(|choices| &choices.0) else {
        return FastStreamEvent::Unsupported;
    };

    let delta_text = choice.delta.as_ref().and_then(|delta| {
        non_empty_str(delta.content)
            .or_else(|| non_empty_str(delta.reasoning_content))
            .or_else(|| non_empty_str(delta.analysis))
            .or_else(|| non_empty_str(delta.reasoning))
    });

    let text = match delta_text {
        Some(text) => text,
        None => match choice.text.map(str::trim).filter(|text| !text.is_empty()) {
            Some(text) => text,
            None => return FastStreamEvent::Unsupported,
        },
    };

    let finish = choice
        .finish_reason
        .filter(|reason| !reason.is_empty() && *reason != "null");

    FastStreamEvent::Text { text, finish }
}

/// Common non-streaming OpenAI-compatible completion shape.
///
/// Only used to skip building a `serde_json::Value` tree for the common case;
/// every shape that is not recognized falls back to the tolerant
/// `extract_completion_text` path, so this must only ever return a value that
/// the generic path would also return.
#[derive(Deserialize)]
struct FastCompletionBody<'a> {
    #[serde(default, borrow)]
    choices: Option<FirstChoice<FastCompletionChoice<'a>>>,
    #[serde(default, borrow)]
    output_text: Option<&'a str>,
}

#[derive(Deserialize)]
struct FastCompletionChoice<'a> {
    #[serde(default)]
    message: Option<FastCompletionMessage<'a>>,
    #[serde(default)]
    text: Option<&'a str>,
}

#[derive(Deserialize)]
struct FastCompletionMessage<'a> {
    #[serde(default)]
    content: Option<&'a str>,
    #[serde(default)]
    reasoning_content: Option<&'a str>,
    #[serde(default)]
    reasoning: Option<&'a str>,
}

fn fast_completion_text(payload: &str) -> Option<String> {
    let parsed: FastCompletionBody<'_> = serde_json::from_str(payload).ok()?;

    if let Some(choice) = parsed.choices.as_ref().map(|choices| &choices.0) {
        if let Some(message) = choice.message.as_ref() {
            if let Some(content) = non_empty_str(message.content)
                .or_else(|| non_empty_str(message.reasoning_content))
                .or_else(|| non_empty_str(message.reasoning))
            {
                return Some(content.to_string());
            }
        }

        if let Some(text) = choice.text.map(str::trim).filter(|text| !text.is_empty()) {
            return Some(text.to_string());
        }
    }

    if let Some(text) = non_empty_str(parsed.output_text) {
        return Some(text.to_string());
    }

    None
}

fn response_is_event_stream(response: &reqwest::Response) -> bool {
    response
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .map(|content_type| content_type.contains("text/event-stream"))
        .unwrap_or(false)
}

fn retry_delay(policy: &RetryPolicy, attempt: u32, retry_after: Option<Duration>) -> Duration {
    let delay = retry_after.unwrap_or_else(|| policy.backoff_for_retry(attempt));
    delay.min(policy.max_backoff)
}

fn is_retryable_status(status: reqwest::StatusCode) -> bool {
    status == reqwest::StatusCode::REQUEST_TIMEOUT
        || status == reqwest::StatusCode::TOO_MANY_REQUESTS
        || status.is_server_error()
}

fn is_retryable_reqwest_error(err: &reqwest::Error) -> bool {
    if err.is_builder() || err.is_redirect() || err.is_decode() {
        return false;
    }
    err.is_timeout() || err.is_connect() || err.is_request() || err.is_body()
}

/// Reads the numeric form of a `Retry-After` header (seconds). HTTP-date
/// values are ignored so a misconfigured provider cannot block a request
/// indefinitely.
fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    let value = headers
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim();
    value.parse::<u64>().ok().map(Duration::from_secs)
}

fn append_valid_utf8(pending: &mut Vec<u8>, out: &mut String) {
    loop {
        let (valid_up_to, error_len) = match std::str::from_utf8(pending) {
            Ok(text) => {
                out.push_str(text);
                pending.clear();
                return;
            }
            Err(err) => (err.valid_up_to(), err.error_len()),
        };

        if valid_up_to > 0 {
            if let Ok(text) = std::str::from_utf8(&pending[..valid_up_to]) {
                out.push_str(text);
            }
            pending.drain(..valid_up_to);
        }

        match error_len {
            Some(invalid_len) => {
                out.push('\u{FFFD}');
                pending.drain(..invalid_len);
            }
            None => return,
        }
    }
}

/// Returns the payload of `event` when it contains exactly one `data:` line,
/// `None` when it contains several (which need to be joined) or none at all.
fn single_data_payload(event: &str) -> Option<&str> {
    let mut found: Option<&str> = None;
    for line in event.lines() {
        if let Some(rest) = line.strip_prefix("data:") {
            if found.is_some() {
                return None;
            }
            found = Some(rest);
        }
    }
    found
}

fn find_event_separator(buffer: &str) -> Option<(usize, usize)> {
    let bytes = buffer.as_bytes();

    for index in 0..bytes.len() {
        match bytes[index] {
            b'\r' if index + 4 <= bytes.len() && &bytes[index + 1..index + 4] == b"\n\r\n" => {
                return Some((index, 4));
            }
            b'\n' if index + 1 < bytes.len() && bytes[index + 1] == b'\n' => {
                return Some((index, 2));
            }
            _ => {}
        }
    }

    None
}

fn extract_completion_text(value: &serde_json::Value) -> Option<String> {
    let choice = value.get("choices").and_then(|c| c.get(0));

    if let Some(choice) = choice {
        if let Some(message) = choice.get("message") {
            if let Some(content) = message.get("content") {
                if let Some(text) = normalize_content_value(content) {
                    if !text.is_empty() {
                        return Some(text);
                    }
                }
            }
            if let Some(reasoning) = message.get("reasoning_content") {
                if let Some(text) = normalize_content_value(reasoning) {
                    if !text.is_empty() {
                        return Some(text);
                    }
                }
            }
            if let Some(reasoning) = message.get("reasoning") {
                if let Some(text) = normalize_content_value(reasoning) {
                    if !text.is_empty() {
                        return Some(text);
                    }
                }
            }
        }

        if let Some(text) = choice.get("text").and_then(|t| t.as_str()) {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }

    if let Some(text) = value
        .get("output_text")
        .and_then(|t| normalize_content_value(t))
    {
        if !text.is_empty() {
            return Some(text);
        }
    }

    if let Some(obj) = value
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c0| c0.get("message"))
    {
        return Some(obj.to_string());
    }

    None
}

fn extract_stream_content(value: &serde_json::Value) -> Option<String> {
    let choice = value.get("choices")?.get(0)?;

    if let Some(delta) = choice.get("delta") {
        if let Some(content) = delta.get("content") {
            if let Some(text) = normalize_content_value(content) {
                if !text.is_empty() {
                    return Some(text);
                }
            }
        }
        if let Some(reasoning) = delta.get("reasoning_content") {
            if let Some(text) = normalize_content_value(reasoning) {
                if !text.is_empty() {
                    return Some(text);
                }
            }
        }
        if let Some(analysis) = delta.get("analysis") {
            if let Some(text) = normalize_content_value(analysis) {
                if !text.is_empty() {
                    return Some(text);
                }
            }
        }
        if let Some(reasoning) = delta.get("reasoning") {
            if let Some(text) = normalize_content_value(reasoning) {
                if !text.is_empty() {
                    return Some(text);
                }
            }
        }
    }

    if let Some(text) = choice.get("text").and_then(|t| t.as_str()) {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    } else {
        None
    }
}

fn append_content_value(value: &serde_json::Value, out: &mut String) {
    match value {
        serde_json::Value::Null => {}
        serde_json::Value::String(s) => out.push_str(s),
        serde_json::Value::Number(n) => {
            let _ = write!(out, "{}", n);
        }
        serde_json::Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        serde_json::Value::Array(items) => {
            for item in items {
                append_content_value(item, out);
            }
        }
        serde_json::Value::Object(map) => {
            for key in ["text", "content", "value", "message"] {
                if let Some(inner) = map.get(key) {
                    let start = out.len();
                    append_content_value(inner, out);
                    if out.len() > start {
                        return;
                    }
                }
            }

            for key in ["parts", "messages"] {
                if let Some(inner) = map.get(key) {
                    let start = out.len();
                    append_content_value(inner, out);
                    if out.len() > start {
                        return;
                    }
                }
            }
        }
    }
}

fn normalize_content_value(value: &serde_json::Value) -> Option<String> {
    let mut out = String::new();
    append_content_value(value, &mut out);
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

fn detect_provider_error(value: &serde_json::Value) -> Option<String> {
    if let Some(error_val) = value.get("error") {
        if let Some(obj) = error_val.as_object() {
            let message = ["message", "msg", "error_message", "error_msg", "detail"]
                .iter()
                .filter_map(|key| obj.get(*key))
                .filter_map(json_value_to_string)
                .map(|s| s.trim().to_string())
                .find(|s| !s.is_empty());
            let code = ["code", "status", "type"]
                .iter()
                .filter_map(|key| obj.get(*key))
                .filter_map(json_value_to_string)
                .map(|s| s.trim().to_string())
                .find(|s| !s.is_empty());
            if let Some(msg) = message {
                if let Some(code_str) = code {
                    return Some(format!("{}: {}", code_str, msg));
                } else {
                    return Some(msg);
                }
            } else {
                return Some(error_val.to_string());
            }
        } else if let Some(text) = json_value_to_string(error_val) {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }

    if let Some(status_val) = value.get("status") {
        if let Some(status_str) = interpret_status_like_error(status_val) {
            let message = extract_message_fields(
                value,
                &[
                    "msg",
                    "message",
                    "error_message",
                    "error_msg",
                    "cause",
                    "detail",
                ],
            );
            return Some(match message {
                Some(msg) => format!("status {}: {}", status_str, msg),
                None => format!("status {}", status_str),
            });
        }
    }

    if let Some(code_val) = value.get("code") {
        if let Some(code_str) = interpret_status_like_error(code_val) {
            let message = extract_message_fields(
                value,
                &[
                    "message",
                    "msg",
                    "error_message",
                    "error_msg",
                    "cause",
                    "detail",
                ],
            );
            return Some(match message {
                Some(msg) => format!("code {}: {}", code_str, msg),
                None => format!("code {}", code_str),
            });
        }
    }

    if let Some(success) = value.get("success").and_then(|v| v.as_bool()) {
        if !success {
            if let Some(msg) = extract_message_fields(
                value,
                &["message", "msg", "error_message", "error_msg", "error"],
            ) {
                return Some(msg);
            }
            return Some("success flag was false".to_string());
        }
    }

    if let Some(msg) = extract_message_fields(value, &["message", "msg"]) {
        let lowered = msg.to_ascii_lowercase();
        if lowered.contains("error")
            || lowered.contains("invalid")
            || lowered.contains("fail")
            || lowered.contains("denied")
            || lowered.contains("unauthorized")
        {
            return Some(msg);
        }
    }

    None
}

fn interpret_status_like_error(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::Number(n) => {
            if let Some(int) = n.as_i64() {
                if int != 0 && int != 200 {
                    return Some(int.to_string());
                }
            } else if let Some(f) = n.as_f64() {
                if f != 0.0 && f != 200.0 {
                    return Some(f.to_string());
                }
            }
            None
        }
        serde_json::Value::String(s) => {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                return None;
            }
            if let Ok(int) = trimmed.parse::<i64>() {
                if int != 0 && int != 200 {
                    return Some(int.to_string());
                }
                return None;
            }
            let lowered = trimmed.to_ascii_lowercase();
            if lowered == "ok"
                || lowered == "success"
                || lowered == "succeeded"
                || lowered == "true"
                || lowered == "0"
                || lowered == "200"
            {
                None
            } else if lowered.contains("error")
                || lowered.contains("fail")
                || lowered.contains("invalid")
                || lowered.contains("denied")
                || lowered.contains("unauthorized")
            {
                Some(trimmed.to_string())
            } else {
                None
            }
        }
        serde_json::Value::Bool(b) => {
            if *b {
                None
            } else {
                Some("false".to_string())
            }
        }
        _ => None,
    }
}

fn extract_message_fields(value: &serde_json::Value, fields: &[&str]) -> Option<String> {
    for key in fields {
        if let Some(inner) = value.get(*key) {
            if let Some(text) = json_value_to_string(inner) {
                let trimmed = text.trim();
                if !trimmed.is_empty() {
                    return Some(trimmed.to_string());
                }
            }
        }
    }
    None
}

fn json_value_to_string(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::Null => None,
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        serde_json::Value::Array(items) => {
            let mut joined: Option<String> = None;
            for item in items {
                if let Some(text) = json_value_to_string(item) {
                    let trimmed = text.trim();
                    if !trimmed.is_empty() {
                        match joined.as_mut() {
                            Some(existing) => {
                                existing.push(' ');
                                existing.push_str(trimmed);
                            }
                            None => joined = Some(trimmed.to_string()),
                        }
                    }
                }
            }
            joined
        }
        serde_json::Value::Object(map) => {
            for key in &[
                "message",
                "msg",
                "error_message",
                "error_msg",
                "detail",
                "description",
            ] {
                if let Some(inner) = map.get(*key) {
                    if let Some(text) = json_value_to_string(inner) {
                        let trimmed = text.trim();
                        if !trimmed.is_empty() {
                            return Some(trimmed.to_string());
                        }
                    }
                }
            }
            Some(value.to_string())
        }
    }
}

pub fn parse_temperature_from_response(response: &str) -> f32 {
    if let Some(value) = parse_temperature_from_json(response) {
        return clamp_temperature(value);
    }

    for line in response.lines() {
        if contains_ascii_case_insensitive(line, "temperature") {
            if let Some(value_part) = line.split_once(':').map(|x| x.1) {
                if let Some(value) = parse_numeric_fragment(value_part) {
                    return clamp_temperature(value);
                }
            }

            if let Some(value) = parse_numeric_fragment(line) {
                return clamp_temperature(value);
            }
        }
    }

    1.4
}

fn contains_ascii_case_insensitive(haystack: &str, needle: &str) -> bool {
    let needle = needle.as_bytes();
    if needle.is_empty() {
        return true;
    }
    let haystack = haystack.as_bytes();
    if needle.len() > haystack.len() {
        return false;
    }
    haystack
        .windows(needle.len())
        .any(|window| window.eq_ignore_ascii_case(needle))
}

fn parse_temperature_from_json(response: &str) -> Option<f32> {
    let json_value = serde_json::from_str::<serde_json::Value>(response).ok()?;
    extract_temperature_from_json(&json_value)
}

fn extract_temperature_from_json(value: &serde_json::Value) -> Option<f32> {
    match value {
        serde_json::Value::Object(map) => {
            if let Some(temp_value) = map.get("temperature") {
                return match temp_value {
                    serde_json::Value::Number(n) => n.as_f64().map(|v| v as f32),
                    serde_json::Value::String(s) => parse_numeric_fragment(s),
                    _ => None,
                };
            }

            map.values().find_map(extract_temperature_from_json)
        }
        serde_json::Value::Array(items) => items.iter().find_map(extract_temperature_from_json),
        _ => None,
    }
}

fn parse_numeric_fragment(input: &str) -> Option<f32> {
    let trimmed = input.trim().trim_matches(|c| c == '"' || c == '\'');
    if let Ok(value) = trimmed.parse::<f32>() {
        return Some(value);
    }

    trimmed
        .split(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'))
        .find(|segment| !segment.is_empty())
        .and_then(|segment| segment.parse::<f32>().ok())
}

fn clamp_temperature(value: f32) -> f32 {
    value.clamp(0.0, 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_temperature_from_json_string_value() {
        let response = r#"{"temperature":"0.65","reasoning":"ok"}"#;
        let value = parse_temperature_from_response(response);
        assert!((value - 0.65).abs() < 1e-6);
    }

    #[test]
    fn parses_temperature_from_text_fragment() {
        let response = "Temperature: \"0.42\", reasoning: details";
        let value = parse_temperature_from_response(response);
        assert!((value - 0.42).abs() < 1e-6);
    }

    #[test]
    fn clamps_out_of_range_values() {
        let response = r#"{"temperature": 3.5}"#;
        assert_eq!(parse_temperature_from_response(response), 2.0);
    }

    #[test]
    fn detects_status_based_error_message() {
        let value = serde_json::json!({
            "status": "434",
            "msg": "Invalid apiKey"
        });
        let err = detect_provider_error(&value).expect("expected error");
        assert!(err.contains("434"));
        assert!(err.to_ascii_lowercase().contains("invalid"));
    }

    #[test]
    fn detects_error_object_code() {
        let value = serde_json::json!({
            "error": {
                "code": "invalid_api_key",
                "message": "No API key provided"
            }
        });
        let err = detect_provider_error(&value).expect("expected error");
        assert!(err.to_ascii_lowercase().contains("invalid_api_key"));
        assert!(err.contains("No API key provided"));
    }

    #[test]
    fn detects_success_flag_false_message() {
        let value = serde_json::json!({
            "success": false,
            "message": "Request failed"
        });
        let err = detect_provider_error(&value).expect("expected error");
        assert!(err.contains("Request failed"));
    }

    #[test]
    fn ignores_successful_status_values() {
        let value = serde_json::json!({
            "status": 0,
            "msg": "ok"
        });
        assert!(detect_provider_error(&value).is_none());
    }

    #[test]
    fn request_body_matches_legacy_value_serialization() {
        let messages = vec![ChatMessage {
            content: "hello".to_string(),
            role: "user".to_string(),
        }];
        let temperature = Some(0.3f32);

        let request = ChatCompletionRequest {
            messages: &messages,
            model: "test-model",
            stream: true,
            temperature: temperature.map(f64::from),
        };

        let legacy = serde_json::json!({
            "model": "test-model",
            "messages": messages,
            "temperature": temperature,
            "stream": true,
        });

        assert_eq!(
            serde_json::to_string(&request).unwrap(),
            legacy.to_string()
        );
    }

    #[test]
    fn borrowed_prompt_request_serializes_like_owned_messages() {
        let prompt = "hello \"quoted\" 中文\nline";
        let temperature = Some(0.3f32);

        let borrowed_messages = [BorrowedChatMessage {
            content: prompt,
            role: "user",
        }];
        let borrowed = BorrowedChatCompletionRequest {
            messages: &borrowed_messages,
            model: "test-model",
            stream: true,
            temperature: temperature.map(f64::from),
        };

        let owned_messages = vec![ChatMessage {
            content: prompt.to_string(),
            role: "user".to_string(),
        }];
        let owned = ChatCompletionRequest {
            messages: &owned_messages,
            model: "test-model",
            stream: true,
            temperature: temperature.map(f64::from),
        };

        assert_eq!(
            serde_json::to_vec(&borrowed).unwrap(),
            serde_json::to_vec(&owned).unwrap()
        );
    }

    #[test]
    fn append_valid_utf8_reassembles_split_multibyte_characters() {
        let full = "中文".as_bytes();
        let mut pending = Vec::new();
        let mut out = String::new();

        pending.extend_from_slice(&full[..2]);
        append_valid_utf8(&mut pending, &mut out);
        assert!(out.is_empty());

        pending.extend_from_slice(&full[2..4]);
        append_valid_utf8(&mut pending, &mut out);
        assert_eq!(out, "中");

        pending.extend_from_slice(&full[4..]);
        append_valid_utf8(&mut pending, &mut out);
        assert_eq!(out, "中文");
        assert!(pending.is_empty());
    }

    #[test]
    fn append_valid_utf8_replaces_invalid_bytes_and_makes_progress() {
        let mut pending = vec![0xFF, b'a'];
        let mut out = String::new();
        append_valid_utf8(&mut pending, &mut out);
        assert_eq!(out, "\u{FFFD}a");
        assert!(pending.is_empty());
    }

    #[test]
    fn fast_stream_event_matches_generic_value_path() {
        let cases: [(&str, bool); 12] = [
            (
                r#"{"choices":[{"index":0,"delta":{"content":"hello"}}]}"#,
                true,
            ),
            (
                r#"{"choices":[{"index":0,"delta":{"content":""},"finish_reason":null}]}"#,
                false,
            ),
            (
                r#"{"choices":[{"index":0,"delta":{"reasoning_content":"think"},"finish_reason":null}]}"#,
                true,
            ),
            (
                r#"{"choices":[{"index":0,"delta":{"content":"","reasoning":"why"}}]}"#,
                true,
            ),
            (r#"{"choices":[{"index":0,"delta":{"analysis":"step"}}]}"#, true),
            (r#"{"choices":[{"text":" plain "}]}"#, true),
            (
                r#"{"choices":[{"delta":{"content":"x"},"finish_reason":"stop"}]}"#,
                true,
            ),
            (r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#, false),
            (
                r#"{"choices":[{"delta":{"content":"x"},"finish_reason":null}]}"#,
                true,
            ),
            (
                r#"{"choices":[{"delta":{"content":"x"},"finish_reason":"null"}]}"#,
                true,
            ),
            (r#"{"choices":[]}"#, false),
            (r#"{"error":{"message":"bad"}}"#, false),
        ];

        for (case, expect_fast) in cases {
            let value: serde_json::Value = serde_json::from_str(case).unwrap();
            let expected_text =
                extract_stream_content(&value).or_else(|| extract_completion_text(&value));
            let expected_finish = value
                .get("choices")
                .and_then(|c| c.get(0))
                .and_then(|c0| c0.get("finish_reason"))
                .and_then(|r| r.as_str())
                .filter(|reason| !reason.is_empty() && *reason != "null");

            match fast_stream_event(case) {
                FastStreamEvent::Text { text, finish } => {
                    assert!(expect_fast, "fast path unexpectedly handled {}", case);
                    assert_eq!(
                        Some(text.to_string()),
                        expected_text,
                        "fast path text diverged for {}",
                        case
                    );
                    assert_eq!(finish, expected_finish, "fast path finish diverged for {}", case);
                }
                FastStreamEvent::Unsupported => {
                    assert!(!expect_fast, "fast path unexpectedly deferred {}", case);
                }
            }
        }
    }

    #[test]
    fn fast_completion_text_never_diverges_from_value_path() {
        let cases = [
            r#"{"choices":[{"index":0,"message":{"role":"assistant","content":"hello"}}]}"#,
            r#"{"choices":[{"index":0,"message":{"role":"assistant","content":""},"text":" tail "}]}"#,
            r#"{"choices":[{"index":0,"message":{"content":"","reasoning_content":"think"}}]}"#,
            r#"{"choices":[{"index":0,"message":{"reasoning":"why"}}]}"#,
            r#"{"choices":[{"index":0,"text":" plain "}]}"#,
            r#"{"output_text":"out"}"#,
            r#"{"output_text":""}"#,
            r#"{"choices":[]}"#,
            r#"{"choices":[{"message":{"content":123}}]}"#,
            r#"{"choices":[{"message":{"content":[{"text":"a"}]}}]}"#,
            r#"{"choices":[{"message":{"content":""}}]}"#,
            r#"{"choices":"nope"}"#,
            r#"{"choices":[{"message":{"content":"x"}},{"message":{"content":"y"}}]}"#,
            "not json at all",
        ];

        for case in cases {
            let expected = serde_json::from_str::<serde_json::Value>(case)
                .ok()
                .and_then(|value| extract_completion_text(&value));
            if let Some(fast) = fast_completion_text(case) {
                assert_eq!(
                    Some(fast),
                    expected,
                    "fast non-stream path diverged for {}",
                    case
                );
            }
        }

        assert_eq!(
            fast_completion_text(
                r#"{"choices":[{"index":0,"message":{"role":"assistant","content":"hello"}}]}"#
            )
            .as_deref(),
            Some("hello"),
            "the common OpenAI shape must be served by the fast path"
        );
    }

    #[test]
    fn fast_stream_event_defers_shapes_it_does_not_cover() {
        let cases = [
            r#"{"choices":[{"message":{"content":"msg"}}]}"#,
            r#"{"output_text":"out"}"#,
            r#"{"choices":[{"delta":{"content":[{"text":"a"}]}}]}"#,
            r#"{"choices":[{"delta":{"content":{"text":"a"}}}]}"#,
            r#"{"choices":[{"delta":{"content":123}}]}"#,
            r#"{"choices":"nope"}"#,
            "not json at all",
        ];

        for case in cases {
            assert!(
                matches!(fast_stream_event(case), FastStreamEvent::Unsupported),
                "fast path must defer unsupported shape: {}",
                case
            );
        }
    }

    #[test]
    fn event_separator_uses_the_earliest_terminator() {
        assert_eq!(
            find_event_separator("data: a\n\ndata: b\r\n\r\n"),
            Some((7, 2))
        );
        assert_eq!(find_event_separator("data: a\r\n\r\n"), Some((7, 4)));
        assert_eq!(find_event_separator("data: a\n\n"), Some((7, 2)));
        assert_eq!(find_event_separator("data: a"), None);
    }

    #[tokio::test]
    async fn consumes_event_stream_with_split_utf8_across_chunks() {
        use axum::body::Body;
        use axum::http::header;
        use axum::response::Response;
        use axum::routing::post;
        use axum::Router;
        use bytes::Bytes;
        use std::convert::Infallible;
        use tokio::sync::mpsc;

        async fn sse_handler() -> Response {
            let mut payload = Vec::new();
            for text in ["中文测试", "完成"] {
                payload.extend_from_slice(
                    format!(
                        "data: {}\n\n",
                        serde_json::json!({"choices":[{"delta":{"content":text}}]})
                    )
                    .as_bytes(),
                );
            }
            payload.extend_from_slice(b"data: [DONE]\n\n");

            let chunks: Vec<Bytes> = payload.chunks(5).map(Bytes::copy_from_slice).collect();
            let body_stream =
                futures::stream::iter(chunks.into_iter().map(Ok::<Bytes, Infallible>));
            Response::builder()
                .header(header::CONTENT_TYPE, "text/event-stream")
                .body(Body::from_stream(body_stream))
                .unwrap()
        }

        let app = Router::new().route("/v1/chat/completions", post(sse_handler));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let client = LLMClient::new(
            format!("http://127.0.0.1:{}/v1", port),
            "sk-test".to_string(),
            30,
        )
        .unwrap();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let messages = vec![ChatMessage {
            content: "hi".to_string(),
            role: "user".to_string(),
        }];

        let result = client
            .chat_completion_with_stream("m", messages, Some(0.0), Some(tx))
            .await
            .unwrap();

        assert!(result.streamed);
        assert_eq!(result.content, "中文测试完成");

        let mut received = String::new();
        while let Ok(piece) = rx.try_recv() {
            received.push_str(&piece);
        }
        assert_eq!(received, "中文测试完成");
    }

    #[tokio::test]
    async fn non_sse_response_with_stream_is_delivered_once() {
        use axum::routing::post;
        use axum::{Json, Router};
        use tokio::sync::mpsc;

        async fn json_handler() -> Json<serde_json::Value> {
            Json(serde_json::json!({
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop"
                }]
            }))
        }

        let app = Router::new().route("/v1/chat/completions", post(json_handler));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let client = LLMClient::new(
            format!("http://127.0.0.1:{}/v1", port),
            "sk-test".to_string(),
            30,
        )
        .unwrap();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let messages = vec![ChatMessage {
            content: "hi".to_string(),
            role: "user".to_string(),
        }];

        let result = client
            .chat_completion_with_stream("m", messages, Some(0.0), Some(tx))
            .await
            .unwrap();

        assert_eq!(result.content, "hello");
        assert!(
            result.streamed,
            "content forwarded through the stream callback must not be sent again"
        );
        assert_eq!(rx.recv().await.as_deref(), Some("hello"));
        assert!(
            rx.try_recv().is_err(),
            "content must be delivered exactly once"
        );
    }

    #[test]
    fn retry_policy_backoff_is_exponential_and_capped() {
        let policy = RetryPolicy::new(3, Duration::from_millis(100), Duration::from_millis(250));
        assert_eq!(policy.backoff_for_retry(1), Duration::from_millis(100));
        assert_eq!(policy.backoff_for_retry(2), Duration::from_millis(200));
        assert_eq!(policy.backoff_for_retry(3), Duration::from_millis(250));
        assert_eq!(policy.backoff_for_retry(10), Duration::from_millis(250));
    }

    #[test]
    fn retry_delay_prefers_retry_after_and_is_capped() {
        let policy = RetryPolicy::new(3, Duration::from_millis(100), Duration::from_millis(500));
        assert_eq!(
            retry_delay(&policy, 1, Some(Duration::from_secs(1))),
            Duration::from_millis(500)
        );
        assert_eq!(
            retry_delay(&policy, 1, Some(Duration::from_millis(20))),
            Duration::from_millis(20)
        );
        assert_eq!(retry_delay(&policy, 2, None), Duration::from_millis(200));
    }

    #[test]
    fn parses_numeric_retry_after_header() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(reqwest::header::RETRY_AFTER, "3".parse().unwrap());
        assert_eq!(parse_retry_after(&headers), Some(Duration::from_secs(3)));

        headers.insert(
            reqwest::header::RETRY_AFTER,
            "Wed, 21 Oct 2015 07:28:00 GMT".parse().unwrap(),
        );
        assert_eq!(parse_retry_after(&headers), None);

        assert_eq!(parse_retry_after(&reqwest::header::HeaderMap::new()), None);
    }

    struct RetryProbe {
        calls: std::sync::atomic::AtomicUsize,
        failures: usize,
        status: axum::http::StatusCode,
    }

    async fn retry_probe_handler(
        axum::extract::State(probe): axum::extract::State<std::sync::Arc<RetryProbe>>,
    ) -> axum::response::Response {
        use axum::response::IntoResponse;
        let call = probe
            .calls
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if call < probe.failures {
            (probe.status, "upstream unavailable").into_response()
        } else {
            axum::Json(serde_json::json!({
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "recovered"},
                    "finish_reason": "stop"
                }]
            }))
            .into_response()
        }
    }

    async fn spawn_retry_probe(probe: std::sync::Arc<RetryProbe>) -> String {
        let app = axum::Router::new()
            .route(
                "/v1/chat/completions",
                axum::routing::post(retry_probe_handler),
            )
            .with_state(probe);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        format!("http://127.0.0.1:{}/v1", port)
    }

    fn test_messages() -> Vec<ChatMessage> {
        vec![ChatMessage {
            content: "hi".to_string(),
            role: "user".to_string(),
        }]
    }

    #[tokio::test]
    async fn retries_retryable_status_then_succeeds() {
        use std::sync::atomic::Ordering;
        use std::sync::Arc;

        let probe = Arc::new(RetryProbe {
            calls: std::sync::atomic::AtomicUsize::new(0),
            failures: 2,
            status: axum::http::StatusCode::SERVICE_UNAVAILABLE,
        });
        let api_base = spawn_retry_probe(Arc::clone(&probe)).await;

        let client = LLMClient::new_with_retry(
            api_base,
            "sk-test".to_string(),
            30,
            RetryPolicy::new(3, Duration::from_millis(1), Duration::from_millis(5)),
        )
        .unwrap();

        let result = client
            .chat_completion("m", test_messages(), Some(0.0))
            .await
            .unwrap();

        assert_eq!(result, "recovered");
        assert_eq!(
            probe.calls.load(Ordering::SeqCst),
            3,
            "two failures plus one successful attempt expected"
        );
    }

    #[tokio::test]
    async fn stops_retrying_after_max_retries() {
        use std::sync::atomic::Ordering;
        use std::sync::Arc;

        let probe = Arc::new(RetryProbe {
            calls: std::sync::atomic::AtomicUsize::new(0),
            failures: usize::MAX,
            status: axum::http::StatusCode::SERVICE_UNAVAILABLE,
        });
        let api_base = spawn_retry_probe(Arc::clone(&probe)).await;

        let client = LLMClient::new_with_retry(
            api_base,
            "sk-test".to_string(),
            30,
            RetryPolicy::new(2, Duration::from_millis(1), Duration::from_millis(5)),
        )
        .unwrap();

        let err = client
            .chat_completion("m", test_messages(), Some(0.0))
            .await
            .expect_err("persistent 503 must eventually fail");

        assert!(
            err.to_string().contains("503"),
            "error should include the upstream status: {}",
            err
        );
        assert_eq!(
            probe.calls.load(Ordering::SeqCst),
            3,
            "max_retries=2 means one initial attempt plus two retries"
        );
    }

    #[tokio::test]
    async fn does_not_retry_client_errors() {
        use std::sync::atomic::Ordering;
        use std::sync::Arc;

        let probe = Arc::new(RetryProbe {
            calls: std::sync::atomic::AtomicUsize::new(0),
            failures: usize::MAX,
            status: axum::http::StatusCode::BAD_REQUEST,
        });
        let api_base = spawn_retry_probe(Arc::clone(&probe)).await;

        let client = LLMClient::new_with_retry(
            api_base,
            "sk-test".to_string(),
            30,
            RetryPolicy::new(3, Duration::from_millis(1), Duration::from_millis(5)),
        )
        .unwrap();

        let err = client
            .chat_completion("m", test_messages(), Some(0.0))
            .await
            .expect_err("400 responses are not retryable");

        assert!(err.to_string().contains("400"), "unexpected error: {}", err);
        assert_eq!(
            probe.calls.load(Ordering::SeqCst),
            1,
            "client errors must not be retried"
        );
    }

    #[tokio::test]
    async fn retry_policy_disabled_keeps_single_attempt() {
        use std::sync::atomic::Ordering;
        use std::sync::Arc;

        let probe = Arc::new(RetryProbe {
            calls: std::sync::atomic::AtomicUsize::new(0),
            failures: usize::MAX,
            status: axum::http::StatusCode::BAD_GATEWAY,
        });
        let api_base = spawn_retry_probe(Arc::clone(&probe)).await;

        let client = LLMClient::new(api_base, "sk-test".to_string(), 30).unwrap();

        let _ = client
            .chat_completion("m", test_messages(), Some(0.0))
            .await
            .expect_err("502 must fail when retries are disabled");

        assert_eq!(
            probe.calls.load(Ordering::SeqCst),
            1,
            "disabled retry policy must make exactly one attempt"
        );
    }

    #[tokio::test]
    async fn retries_connection_failures() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&calls);
        tokio::spawn(async move {
            while let Ok((socket, _)) = listener.accept().await {
                counter.fetch_add(1, Ordering::SeqCst);
                drop(socket);
            }
        });

        let client = LLMClient::new_with_retry(
            format!("http://127.0.0.1:{}/v1", port),
            "sk-test".to_string(),
            5,
            RetryPolicy::new(2, Duration::from_millis(1), Duration::from_millis(5)),
        )
        .unwrap();

        let _ = client
            .chat_completion("m", test_messages(), Some(0.0))
            .await
            .expect_err("dropped connections must eventually surface an error");

        assert_eq!(
            calls.load(Ordering::SeqCst),
            3,
            "expected one initial attempt plus two retries for connection failures"
        );
    }

    #[tokio::test]
    async fn retries_streaming_body_failure_before_first_token() {
        use axum::body::Body;
        use axum::http::header;
        use axum::response::Response;
        use axum::routing::post;
        use axum::Router;
        use bytes::Bytes;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        use tokio::sync::mpsc;

        let calls = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&calls);

        let app = Router::new().route(
            "/v1/chat/completions",
            post(move || {
                let counter = Arc::clone(&counter);
                async move {
                    let call = counter.fetch_add(1, Ordering::SeqCst);
                    if call == 0 {
                        let keepalive = futures::stream::once(async {
                            Ok::<Bytes, std::io::Error>(Bytes::from_static(b": keepalive\n\n"))
                        });
                        let abort = futures::stream::once(async {
                            tokio::time::sleep(Duration::from_millis(30)).await;
                            Err::<Bytes, std::io::Error>(std::io::Error::new(
                                std::io::ErrorKind::Other,
                                "upstream aborted",
                            ))
                        });
                        Response::builder()
                            .header(header::CONTENT_TYPE, "text/event-stream")
                            .body(Body::from_stream(
                                futures::StreamExt::chain(keepalive, abort),
                            ))
                            .unwrap()
                    } else {
                        Response::builder()
                            .header(header::CONTENT_TYPE, "text/event-stream")
                            .body(Body::from(
                                "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n",
                            ))
                            .unwrap()
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let client = LLMClient::new_with_retry(
            format!("http://127.0.0.1:{}/v1", port),
            "sk-test".to_string(),
            30,
            RetryPolicy::new(2, Duration::from_millis(1), Duration::from_millis(5)),
        )
        .unwrap();

        let (tx, mut rx) = mpsc::unbounded_channel();
        let result = client
            .chat_completion_prompt_with_stream("m", "hi", Some(0.0), Some(tx))
            .await
            .expect("a stream failing before any content must be retried");

        assert_eq!(result.content, "hello");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "aborted stream should be retried exactly once"
        );
        let mut streamed = Vec::new();
        while let Ok(piece) = rx.try_recv() {
            streamed.push(piece);
        }
        assert_eq!(streamed, vec!["hello".to_string()]);
    }

    #[tokio::test]
    async fn does_not_retry_streaming_failure_after_content() {
        use axum::body::Body;
        use axum::http::header;
        use axum::response::Response;
        use axum::routing::post;
        use axum::Router;
        use bytes::Bytes;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        use tokio::sync::mpsc;

        let calls = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&calls);

        let app = Router::new().route(
            "/v1/chat/completions",
            post(move || {
                let counter = Arc::clone(&counter);
                async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                    let first = futures::stream::once(async {
                        Ok::<Bytes, std::io::Error>(Bytes::from_static(
                            b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\n",
                        ))
                    });
                    let abort = futures::stream::once(async {
                        tokio::time::sleep(Duration::from_millis(30)).await;
                        Err::<Bytes, std::io::Error>(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            "upstream aborted",
                        ))
                    });
                    let body_stream = futures::StreamExt::chain(first, abort);
                    Response::builder()
                        .header(header::CONTENT_TYPE, "text/event-stream")
                        .body(Body::from_stream(body_stream))
                        .unwrap()
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let client = LLMClient::new_with_retry(
            format!("http://127.0.0.1:{}/v1", port),
            "sk-test".to_string(),
            30,
            RetryPolicy::new(2, Duration::from_millis(1), Duration::from_millis(5)),
        )
        .unwrap();

        let (tx, mut rx) = mpsc::unbounded_channel();
        let _ = client
            .chat_completion_prompt_with_stream("m", "hi", Some(0.0), Some(tx))
            .await
            .expect_err("a stream failing after content must not be retried");

        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "partial streamed output must never be retried"
        );
        let mut streamed = Vec::new();
        while let Ok(piece) = rx.try_recv() {
            streamed.push(piece);
        }
        assert_eq!(streamed, vec!["partial".to_string()]);
    }
}
