mod config;
mod llm;
mod server;
mod workflow;

#[cfg(test)]
mod config_tests;

use anyhow::Result;
use std::sync::Arc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<()> {
    // 初始化日志
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "chorus=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // 自动加载配置（env > ~/.config/chorus/config.toml）
    let config = config::Config::load_auto()?;

    tracing::info!(
        "Starting Chorus server on {}:{}",
        config.server.host,
        config.server.port
    );
    tracing::info!(
        "Analyzer model: {}",
        config.workflow_integration.analyzer.model
    );
    if tracing::enabled!(tracing::Level::INFO) {
        tracing::info!("Worker nodes: {:?}", config.workflow_integration.worker_labels());
    }
    if let Some(synth) = &config.workflow_integration.synthesizer {
        tracing::info!("Synthesizer model: {}", synth.model);
    } else if let Some(selector) = &config.workflow_integration.selector {
        tracing::info!(
            "No synthesizer configured; using selector: {}",
            selector.model
        );
    } else {
        tracing::warn!("No synthesizer or selector configured for the workflow");
    }

    // 启动服务器
    server::start_server(Arc::new(config)).await?;

    Ok(())
}
