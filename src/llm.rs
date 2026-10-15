use anyhow::{anyhow, Context, Result};
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

#[derive(Serialize)]
struct ChatCompletionRequest<'a> {
    messages: &'a [ChatMessage],
    model: &'a str,
    stream: bool,
    temperature: Option<f64>,
}

#[derive(Clone)]
pub struct LLMClient {
    client: Client,
    api_base: String,
    authorization: String,
    chat_completions_url: String,
}

impl LLMClient {
    pub fn new(api_base: String, api_key: String, timeout_secs: u64) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()
            .with_context(|| format!("Failed to build HTTP client for {}", api_base))?;

        let chat_completions_url =
            format!("{}/chat/completions", api_base.trim_end_matches('/'));

        Ok(Self {
            client,
            api_base,
            authorization: format!("Bearer {}", api_key),
            chat_completions_url,
        })
    }

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

    pub async fn chat_completion_with_stream(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        temperature: Option<f32>,
        stream: Option<UnboundedSender<String>>,
    ) -> Result<CompletionResult> {
        let stream_was_requested = stream.is_some();
        let request = ChatCompletionRequest {
            messages: &messages,
            model,
            stream: stream_was_requested,
            temperature: temperature.map(f64::from),
        };

        tracing::debug!(
            "Calling LLM API: {} with model: {}",
            self.chat_completions_url,
            model
        );
        if tracing::enabled!(tracing::Level::DEBUG) {
            tracing::debug!("Request body: {}", serde_json::to_string_pretty(&request)?);
        }

        let response = self
            .client
            .post(&self.chat_completions_url)
            .header("Authorization", &self.authorization)
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(anyhow!(
                "LLM API request failed with status {}: {}",
                status,
                error_text
            ));
        }

        if stream_was_requested && response_is_event_stream(&response) {
            return self.consume_event_stream(response, stream).await;
        }

        // Be tolerant to different provider response shapes
        let v: serde_json::Value = response.json().await?;

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
            ));
        }

        Err(anyhow!("LLM response missing content field: {}", v))
    }

    async fn consume_event_stream(
        &self,
        response: reqwest::Response,
        stream: Option<UnboundedSender<String>>,
    ) -> Result<CompletionResult> {
        let mut final_text = String::new();
        let mut streamed = false;
        let mut buffer = String::new();
        let mut pending = Vec::new();
        let mut cursor = 0usize;
        let mut byte_stream = response.bytes_stream();

        while let Some(item) = byte_stream.next().await {
            let chunk = item?;
            pending.extend_from_slice(&chunk);
            append_valid_utf8(&mut pending, &mut buffer);

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

                let mut data_payload = String::new();
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

                let trimmed = data_payload.trim();
                if trimmed.is_empty() {
                    continue;
                }

                if trimmed == "[DONE]" {
                    return Ok(CompletionResult {
                        content: final_text,
                        streamed,
                    });
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

fn response_is_event_stream(response: &reqwest::Response) -> bool {
    response
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .map(|content_type| content_type.contains("text/event-stream"))
        .unwrap_or(false)
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

fn find_event_separator(buffer: &str) -> Option<(usize, usize)> {
    let lf = buffer.find("\n\n").map(|idx| (idx, 2));
    let crlf = buffer.find("\r\n\r\n").map(|idx| (idx, 4));

    match (lf, crlf) {
        (Some(lf), Some(crlf)) => Some(if lf.0 <= crlf.0 { lf } else { crlf }),
        (Some(lf), None) => Some(lf),
        (None, Some(crlf)) => Some(crlf),
        (None, None) => None,
    }
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
            let mut parts = Vec::new();
            for item in items {
                if let Some(text) = json_value_to_string(item) {
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
}
