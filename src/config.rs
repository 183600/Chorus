use anyhow::{anyhow, Context, Result};
use serde::de::Error as DeError;
use serde::{de::Deserializer, Deserialize, Serialize};
use serde_json::{Map as JsonMap, Number as JsonNumber, Value as JsonValue};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use toml::Value;

const DEFAULT_CONFIG: &str = r#"# Chorus 默认配置
[server]
host = "127.0.0.1"
port = 11435

[[model]]
api_base = "https://apis.iflow.cn/v1"
api_key = "your-api-key-here"
name = "qwen3-max"

[[model]]
api_base = "https://apis.iflow.cn/v1"
api_key = "your-api-key-here"
name = "qwen3-vl-plus"

[[model]]
api_base = "https://apis.iflow.cn/v1"
api_key = "your-api-key-here"
name = "kimi-k2-0905"

[[model]]
api_base = "https://apis.iflow.cn/v1"
api_key = "your-api-key-here"
name = "glm-4.6"

[[model]]
api_base = "https://apis.iflow.cn/v1"
api_key = "your-api-key-here"
name = "deepseek-v3.2"

[[model]]
api_base = "https://apis.iflow.cn/v1"
api_key = "your-api-key-here"
name = "deepseek-v3.1"

[[model]]
api_base = "https://apis.iflow.cn/v1"
api_key = "your-api-key-here"
name = "deepseek-r1"

[[model]]
api_base = "https://apis.iflow.cn/v1"
api_key = "your-api-key-here"
name = "qwen3-coder"

[[model]]
api_base = "https://api.tbox.cn/api/llm/v1"
api_key = "your-api-key-here"
name = "ring-1t"

[workflow-integration]
nested_worker_depth = 1

json = """{
  "analyzer": {
    "ref": "glm-4.6",
    "auto_temperature": true
  },
  "workers": [
    {
      "name": "deepseek-v3.2",
      "temperature": 1
    },
    {
      "analyzer": {
        "ref": "glm-4.6",
        "auto_temperature": true
      },
      "workers": [
        {
          "name": "kimi-k2-0905",
          "temperature": 1
        },
        {
          "name": "deepseek-v3.2",
          "temperature": 1
        },
        {
          "name": "glm-4.6",
          "temperature": 1
        },
        {
          "analyzer": {
            "ref": "glm-4.6",
            "auto_temperature": true
          },
          "workers": [
            {
              "name": "qwen3-coder",
              "temperature": 1
            },
            {
              "name": "deepseek-v3.1",
              "temperature": 1
            },
            {
              "name": "qwen3-max",
              "temperature": 1
            }
          ],
          "synthesizer": {
            "ref": "qwen3-max"
          }
        }
      ],
      "synthesizer": {
        "ref": "qwen3-max"
      }
    }
  ],
  "synthesizer": {
    "ref": "qwen3-max"
  },
  "selector": {
    "ref": "qwen3-max"
  }
}"""

[workflow.timeouts]
# 所有超时时间单位均为秒
analyzer_timeout_secs = 30
worker_timeout_secs = 60
synthesizer_timeout_secs = 60

[workflow.domains]

[workflow.domains."api.example.com"]
analyzer_timeout_secs = 40
worker_timeout_secs = 80

[workflow.domains."app.example.com"]
analyzer_timeout_secs = 20
synthesizer_timeout_secs = 30
"#;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    #[serde(rename = "model", deserialize_with = "deserialize_models")]
    pub models: Vec<ModelConfig>,
    #[serde(
        rename = "workflow-integration",
        deserialize_with = "deserialize_workflow_plan"
    )]
    pub workflow_integration: WorkflowPlan,
    pub workflow: WorkflowConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub api_base: String,
    pub api_key: String,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub auto_temperature: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowPlan {
    pub analyzer: WorkflowModelTarget,
    #[serde(default)]
    pub workers: Vec<WorkflowWorker>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub synthesizer: Option<WorkflowModelTarget>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selector: Option<WorkflowModelTarget>,
    #[serde(default)]
    pub nested_worker_depth: Option<u32>,
}

impl WorkflowPlan {
    pub fn label(&self) -> String {
        if let Some(synthesizer) = &self.synthesizer {
            format!("workflow:{}", synthesizer.model)
        } else if let Some(selector) = &self.selector {
            format!("workflow:selector:{}", selector.model)
        } else {
            format!("workflow:{}", self.analyzer.model)
        }
    }

    pub fn worker_labels(&self) -> Vec<String> {
        self.workers.iter().map(WorkflowWorker::label).collect()
    }

    pub fn to_json_string(&self) -> Result<String> {
        let value = self.to_json_value()?;
        serde_json::to_string_pretty(&value)
            .with_context(|| "Failed to serialize workflow integration to JSON")
    }

    pub fn from_json_str(json: &str) -> Result<Self> {
        let mut value: JsonValue = serde_json::from_str(json)
            .map_err(|err| anyhow!("Failed to parse workflow integration JSON: {}", err))?;

        Self::ensure_workflow_targets(&mut value)
            .map_err(|err| anyhow!("Failed to parse workflow integration JSON: {}", err))?;

        let mut plan: Self = serde_json::from_value(value)
            .map_err(|err| anyhow!("Failed to parse workflow integration JSON: {}", err))?;

        plan.validate_structure()
            .map_err(|err| anyhow!("Failed to parse workflow integration JSON: {}", err))?;
        plan.inherit_missing_synthesizers();

        Ok(plan)
    }

    fn ensure_workflow_targets(value: &mut JsonValue) -> Result<()> {
        if !value.is_object() {
            return Err(anyhow!(
                "Workflow integration JSON must start with an object at the root"
            ));
        }

        Self::ensure_workflow_targets_recursive(value, None, "workflow")?;
        Ok(())
    }

    fn ensure_workflow_targets_recursive(
        node: &mut JsonValue,
        inherited_synthesizer: Option<JsonValue>,
        path: &str,
    ) -> Result<Option<JsonValue>> {
        let map = node
            .as_object_mut()
            .ok_or_else(|| anyhow!("Workflow node at {} must be a JSON object", path))?;

        let mut synthesizer_value = match map.get("synthesizer") {
            Some(existing) => {
                if !existing.is_object() {
                    return Err(anyhow!(
                        "Workflow node at {} has an invalid `synthesizer` field; expected an object with `ref` or `name`.",
                        path
                    ));
                }
                Some(existing.clone())
            }
            None => None,
        };

        if let Some(selector_value) = map.get("selector") {
            if !selector_value.is_object() {
                return Err(anyhow!(
                    "Workflow node at {} has an invalid `selector` field; expected an object with `ref` or `name`.",
                    path
                ));
            }
        }

        let has_selector = map.contains_key("selector");

        if synthesizer_value.is_none() && !has_selector {
            if let Some(inherited) = &inherited_synthesizer {
                map.insert("synthesizer".to_string(), inherited.clone());
                synthesizer_value = Some(inherited.clone());
            }
        }

        if synthesizer_value.is_none() && !has_selector {
            return Err(anyhow!(
                "Workflow node at {} must define at least one of `synthesizer` or `selector`",
                path
            ));
        }

        let inherited_for_children = if let Some(synth) = &synthesizer_value {
            Some(synth.clone())
        } else {
            inherited_synthesizer.clone()
        };

        if let Some(workers) = map
            .get_mut("workers")
            .and_then(|workers| workers.as_array_mut())
        {
            for (index, worker) in workers.iter_mut().enumerate() {
                if Self::is_nested_workflow(worker) {
                    let nested_path = format!("{} -> workers[{}]", path, index);
                    Self::ensure_workflow_targets_recursive(
                        worker,
                        inherited_for_children.clone(),
                        &nested_path,
                    )?;
                }
            }
        }

        Ok(synthesizer_value.or(inherited_synthesizer))
    }

    pub fn inherit_missing_synthesizers(&mut self) {
        self.apply_synthesizer_inheritance(None);
    }

    pub fn apply_worker_replication(&mut self) {
        if let Some(depth) = self.nested_worker_depth {
            if depth > 1 {
                let analyzer = self.analyzer.clone();
                let synthesizer = self.synthesizer.clone();
                let selector = self.selector.clone();

                let mut new_workers = Vec::new();
                for worker in self.workers.drain(..) {
                    let replicated = Self::replicate_worker_static(
                        worker,
                        depth,
                        &analyzer,
                        &synthesizer,
                        &selector,
                    );
                    new_workers.push(replicated);
                }
                self.workers = new_workers;
            }
        }
    }

    fn replicate_worker_static(
        worker: WorkflowWorker,
        depth: u32,
        analyzer: &WorkflowModelTarget,
        synthesizer: &Option<WorkflowModelTarget>,
        selector: &Option<WorkflowModelTarget>,
    ) -> WorkflowWorker {
        match worker {
            WorkflowWorker::Model(model_target) => {
                if depth == 1 {
                    WorkflowWorker::Model(model_target)
                } else {
                    let mut nested_plan = WorkflowPlan {
                        analyzer: analyzer.clone(),
                        workers: vec![
                            WorkflowWorker::Model(model_target.clone()),
                            WorkflowWorker::Model(model_target),
                        ],
                        synthesizer: synthesizer.clone(),
                        selector: selector.clone(),
                        nested_worker_depth: Some(depth - 1),
                    };
                    nested_plan.apply_worker_replication();
                    WorkflowWorker::Workflow(Box::new(nested_plan))
                }
            }
            WorkflowWorker::Workflow(_) => worker,
        }
    }

    fn apply_synthesizer_inheritance(
        &mut self,
        inherited: Option<&WorkflowModelTarget>,
    ) -> Option<WorkflowModelTarget> {
        let has_selector = self.selector.is_some();

        let current = if let Some(existing) = &self.synthesizer {
            Some(existing.clone())
        } else if has_selector {
            inherited.cloned()
        } else if let Some(parent) = inherited {
            self.synthesizer = Some(parent.clone());
            self.synthesizer.clone()
        } else {
            None
        };

        for worker in self.workers.iter_mut() {
            if let WorkflowWorker::Workflow(plan) = worker {
                plan.apply_synthesizer_inheritance(current.as_ref());
            }
        }

        current
    }

    fn is_nested_workflow(value: &JsonValue) -> bool {
        value.as_object().map_or(false, |map| {
            map.contains_key("analyzer") && map.contains_key("workers")
        })
    }

    fn to_json_value(&self) -> Result<JsonValue> {
        let mut map = JsonMap::new();
        map.insert(
            "analyzer".to_string(),
            JsonValue::Object(Self::target_to_json_map(&self.analyzer, "ref")),
        );

        let mut workers = Vec::with_capacity(self.workers.len());
        for worker in &self.workers {
            workers.push(Self::worker_to_json(worker)?);
        }
        map.insert("workers".to_string(), JsonValue::Array(workers));
        if let Some(synthesizer) = &self.synthesizer {
            map.insert(
                "synthesizer".to_string(),
                JsonValue::Object(Self::target_to_json_map(synthesizer, "ref")),
            );
        }
        if let Some(selector) = &self.selector {
            map.insert(
                "selector".to_string(),
                JsonValue::Object(Self::target_to_json_map(selector, "ref")),
            );
        }

        Ok(JsonValue::Object(map))
    }

    pub fn validate_structure(&self) -> Result<()> {
        self.validate_with_context(None, "workflow")
    }

    fn validate_with_context(
        &self,
        inherited_synthesizer: Option<&WorkflowModelTarget>,
        path: &str,
    ) -> Result<()> {
        let synthesizer = self.synthesizer.as_ref().or(inherited_synthesizer);
        let has_selector = self.selector.is_some();

        if synthesizer.is_none() && !has_selector {
            return Err(anyhow!(
                "Workflow node at {} must define at least one of `synthesizer` or `selector`",
                path
            ));
        }

        for (index, worker) in self.workers.iter().enumerate() {
            if let WorkflowWorker::Workflow(plan) = worker {
                let nested_path = format!("{} -> workers[{}]", path, index);
                plan.validate_with_context(synthesizer, &nested_path)?;
            }
        }

        Ok(())
    }

    fn worker_to_json(worker: &WorkflowWorker) -> Result<JsonValue> {
        match worker {
            WorkflowWorker::Model(target) => {
                Ok(JsonValue::Object(Self::target_to_json_map(target, "name")))
            }
            WorkflowWorker::Workflow(plan) => plan.to_json_value(),
        }
    }

    fn target_to_json_map(target: &WorkflowModelTarget, key: &str) -> JsonMap<String, JsonValue> {
        let mut map = JsonMap::new();
        map.insert(key.to_string(), JsonValue::String(target.model.clone()));
        if let Some(temp) = target.temperature {
            if let Some(number) = JsonNumber::from_f64(temp as f64) {
                map.insert("temperature".to_string(), JsonValue::Number(number));
            }
        }
        if let Some(auto) = target.auto_temperature {
            map.insert("auto_temperature".to_string(), JsonValue::Bool(auto));
        }
        map
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowModelTarget {
    #[serde(rename = "ref", alias = "name")]
    pub model: String,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub auto_temperature: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub enum WorkflowWorker {
    Model(WorkflowModelTarget),
    Workflow(Box<WorkflowPlan>),
}

impl WorkflowWorker {
    pub fn label(&self) -> String {
        match self {
            WorkflowWorker::Model(target) => target.model.clone(),
            WorkflowWorker::Workflow(plan) => plan.label(),
        }
    }
}

impl<'de> Deserialize<'de> for WorkflowWorker {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = JsonValue::deserialize(deserializer)?;
        match value {
            JsonValue::Object(map) => {
                let has_name = map.contains_key("name") || map.contains_key("ref");
                let has_analyzer = map.contains_key("analyzer");
                let has_workers = map.contains_key("workers");
                let has_synthesizer = map.contains_key("synthesizer");
                let has_selector = map.contains_key("selector");

                let value = JsonValue::Object(map);

                if has_analyzer || has_workers || has_synthesizer || has_selector {
                    if !has_analyzer {
                        return Err(D::Error::custom(
                            "Nested workflow worker is missing required `analyzer` field. Each nested workflow must include an `analyzer` configuration.",
                        ));
                    }
                    if !has_workers {
                        return Err(D::Error::custom(
                            "Nested workflow worker is missing required `workers` field. Provide at least one worker entry inside the nested workflow.",
                        ));
                    }

                    let plan: WorkflowPlan = serde_json::from_value(value).map_err(|err| {
                        D::Error::custom(format!("Failed to parse nested workflow worker: {}", err))
                    })?;
                    return Ok(WorkflowWorker::Workflow(Box::new(plan)));
                }

                if has_name {
                    let target: WorkflowModelTarget =
                        serde_json::from_value(value).map_err(|err| {
                            D::Error::custom(format!(
                                "Failed to parse workflow worker model: {}",
                                err
                            ))
                        })?;
                    return Ok(WorkflowWorker::Model(target));
                }

                Err(D::Error::custom(
                    "Workflow worker entries must either specify a `name`/`ref` model or a nested workflow with `analyzer`, `workers`, and `synthesizer` fields",
                ))
            }
            JsonValue::String(name) => Ok(WorkflowWorker::Model(WorkflowModelTarget {
                model: name,
                temperature: None,
                auto_temperature: None,
            })),
            other => Err(D::Error::custom(format!(
                "Workflow worker entries must be JSON objects or string model references, got {}",
                other
            ))),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowConfig {
    pub timeouts: TimeoutConfig,
    #[serde(default)]
    pub domains: HashMap<String, DomainTimeoutOverride>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker_concurrency: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutConfig {
    pub analyzer_timeout_secs: u64,
    pub worker_timeout_secs: u64,
    pub synthesizer_timeout_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DomainTimeoutOverride {
    pub analyzer_timeout_secs: Option<u64>,
    pub worker_timeout_secs: Option<u64>,
    pub synthesizer_timeout_secs: Option<u64>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ModelOneOrMany {
    One(ModelConfig),
    Many(Vec<ModelConfig>),
}

fn deserialize_models<'de, D>(deserializer: D) -> std::result::Result<Vec<ModelConfig>, D::Error>
where
    D: Deserializer<'de>,
{
    let v = ModelOneOrMany::deserialize(deserializer)?;
    Ok(match v {
        ModelOneOrMany::One(m) => vec![m],
        ModelOneOrMany::Many(vs) => vs,
    })
}

fn deserialize_workflow_plan<'de, D>(deserializer: D) -> std::result::Result<WorkflowPlan, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    struct JsonWrapper {
        json: String,
        #[serde(default)]
        nested_worker_depth: Option<u32>,
    }

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum PlanInput {
        Json(JsonWrapper),
        PlainString(String),
        Plan(WorkflowPlan),
    }

    match PlanInput::deserialize(deserializer)? {
        PlanInput::Json(wrapper) => {
            let mut plan = WorkflowPlan::from_json_str(&wrapper.json).map_err(|err| {
                DeError::custom(format!("Failed to parse workflow json: {}", err))
            })?;
            plan.nested_worker_depth = wrapper.nested_worker_depth;
            plan.apply_worker_replication();
            Ok(plan)
        }
        PlanInput::PlainString(json) => WorkflowPlan::from_json_str(&json)
            .map_err(|err| DeError::custom(format!("Failed to parse workflow json: {}", err))),
        PlanInput::Plan(mut plan) => {
            plan.validate_structure().map_err(|err| {
                DeError::custom(format!("Failed to parse workflow json: {}", err))
            })?;
            plan.inherit_missing_synthesizers();
            plan.apply_worker_replication();
            Ok(plan)
        }
    }
}

impl Config {
    pub fn load_auto() -> Result<Self> {
        if let Ok(path) = env::var("CHORUS_CONFIG") {
            let path = PathBuf::from(path);
            if path.exists() {
                return Self::load(&path.to_string_lossy());
            } else {
                tracing::warn!(
                    "CHORUS_CONFIG points to non-existent file: {}",
                    path.display()
                );
            }
        }

        Self::load_from_user_config()
    }

    pub fn load(path: &str) -> Result<Self> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path))?;
        let cfg: Config = toml::from_str(&content)
            .with_context(|| format!("Failed to parse TOML from {}", path))?;
        Ok(cfg)
    }

    fn user_config_path() -> Result<PathBuf> {
        let home = env::var("HOME").context("HOME env var not set")?;
        Ok(Path::new(&home)
            .join(".config")
            .join("chorus")
            .join("config.toml"))
    }

    fn ensure_user_config_exists() -> Result<PathBuf> {
        let path = Self::user_config_path()?;
        if let Some(dir) = path.parent() {
            if !dir.exists() {
                fs::create_dir_all(dir)
                    .with_context(|| format!("Failed to create config dir: {}", dir.display()))?;
            }
        }
        if !path.exists() {
            fs::write(&path, DEFAULT_CONFIG)
                .with_context(|| format!("Failed to write default config to {}", path.display()))?;
        } else {
            Self::migrate_config_if_needed(&path)?;
        }
        Ok(path)
    }

    pub(crate) fn migrate_config_if_needed(config_path: &Path) -> Result<()> {
        let content = fs::read_to_string(config_path)
            .with_context(|| format!("Failed to read config file: {}", config_path.display()))?;

        let value: Value = match toml::from_str(&content) {
            Ok(value) => value,
            Err(_) => return Ok(()),
        };

        let workflow_table = value.get("workflow-integration").and_then(Value::as_table);

        let has_json = workflow_table
            .and_then(|table| table.get("json"))
            .and_then(Value::as_str)
            .is_some();

        let legacy_fields_present = workflow_table
            .map(|table| {
                table.contains_key("analyzer_model")
                    || table.contains_key("worker_models")
                    || table.contains_key("synthesizer_model")
            })
            .unwrap_or(false);

        let nested_worker_depth = workflow_table
            .and_then(|table| table.get("nested_worker_depth"))
            .and_then(Value::as_integer)
            .and_then(|value| u32::try_from(value).ok());

        if has_json && !legacy_fields_present {
            return Ok(());
        }

        let mut migrations: Vec<&str> = Vec::new();

        let config = if legacy_fields_present {
            tracing::info!(
                "Detected legacy workflow-integration format, migrating to workflow plan JSON"
            );

            #[derive(Deserialize)]
            struct LegacyWorkflowIntegration {
                analyzer_model: String,
                worker_models: Vec<String>,
                synthesizer_model: String,
            }

            #[derive(Deserialize)]
            struct LegacyConfig {
                server: ServerConfig,
                #[serde(rename = "model", deserialize_with = "deserialize_models")]
                models: Vec<ModelConfig>,
                #[serde(rename = "workflow-integration")]
                workflow_integration: LegacyWorkflowIntegration,
                workflow: WorkflowConfig,
            }

            migrations.push("workflow 节点结构");

            match toml::from_str::<LegacyConfig>(&content) {
                Ok(legacy) => {
                    let mut config = Config {
                        server: legacy.server,
                        models: legacy.models,
                        workflow_integration: WorkflowPlan {
                            analyzer: WorkflowModelTarget {
                                model: legacy.workflow_integration.analyzer_model,
                                temperature: None,
                                auto_temperature: None,
                            },
                            workers: legacy
                                .workflow_integration
                                .worker_models
                                .into_iter()
                                .map(|model| {
                                    WorkflowWorker::Model(WorkflowModelTarget {
                                        model,
                                        temperature: None,
                                        auto_temperature: None,
                                    })
                                })
                                .collect(),
                            synthesizer: Some(WorkflowModelTarget {
                                model: legacy.workflow_integration.synthesizer_model,
                                temperature: None,
                                auto_temperature: None,
                            }),
                            selector: None,
                            nested_worker_depth,
                        },
                        workflow: legacy.workflow,
                    };
                    config.workflow_integration.apply_worker_replication();
                    config
                }
                Err(err) => {
                    tracing::warn!(
                        "Detected legacy workflow-integration fields but failed to parse legacy format: {}. Falling back to workflow plan JSON parser.",
                        err
                    );
                    toml::from_str::<Config>(&content).with_context(|| {
                        "Failed to parse config after falling back to workflow plan JSON parser"
                    })?
                }
            }
        } else {
            toml::from_str::<Config>(&content)
                .with_context(|| "Failed to parse config for migration")?
        };

        if !has_json {
            migrations.push("workflow json 格式");
        }

        if migrations.is_empty() {
            return Ok(());
        }

        let backup_path = Self::backup_config_file(config_path)?;
        tracing::info!("Old config backed up to: {}", backup_path.display());

        let mut new_content = String::new();
        new_content.push_str(&format!(
            "# Chorus 配置文件（已自动迁移：{}）\n",
            migrations.join("，")
        ));
        new_content.push_str(&format!("# 旧配置已备份到: {}\n\n", backup_path.display()));

        let new_value = Self::config_to_toml_value(&config)?;
        new_content.push_str(
            &toml::to_string_pretty(&new_value)
                .with_context(|| "Failed to serialize migrated config")?,
        );

        fs::write(config_path, new_content).with_context(|| {
            format!(
                "Failed to write migrated config to {}",
                config_path.display()
            )
        })?;

        tracing::info!(
            "Config migration completed successfully: {}",
            migrations.join("，")
        );
        tracing::info!("New config written to: {}", config_path.display());

        Ok(())
    }

    fn config_to_toml_value(config: &Config) -> Result<Value> {
        let mut root = toml::map::Map::new();

        let server_value =
            Value::try_from(&config.server).with_context(|| "Failed to serialize server config")?;
        root.insert("server".to_string(), server_value);

        let models_value =
            Value::try_from(&config.models).with_context(|| "Failed to serialize model configs")?;
        root.insert("model".to_string(), models_value);

        let workflow_json = config
            .workflow_integration
            .to_json_string()
            .with_context(|| "Failed to serialize workflow integration to JSON string")?;
        let mut workflow_integration = toml::map::Map::new();
        workflow_integration.insert("json".to_string(), Value::String(workflow_json));
        if let Some(depth) = config.workflow_integration.nested_worker_depth {
            workflow_integration.insert(
                "nested_worker_depth".to_string(),
                Value::Integer(i64::from(depth)),
            );
        }
        root.insert(
            "workflow-integration".to_string(),
            Value::Table(workflow_integration),
        );

        let workflow_value = Value::try_from(&config.workflow)
            .with_context(|| "Failed to serialize workflow configuration")?;
        root.insert("workflow".to_string(), workflow_value);

        Ok(Value::Table(root))
    }

    fn backup_config_file(config_path: &Path) -> Result<PathBuf> {
        let mut backup_path = config_path.with_extension("toml.bak");
        if backup_path.exists() {
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            backup_path = config_path.with_extension(format!("toml.bak.{}", timestamp));
        }

        fs::copy(config_path, &backup_path)
            .with_context(|| format!("Failed to backup config to {}", backup_path.display()))?;

        Ok(backup_path)
    }

    pub fn load_from_user_config() -> Result<Self> {
        let path = Self::ensure_user_config_exists()?;
        Self::load(&path.to_string_lossy())
    }

    pub fn effective_timeouts_for_domain(&self, domain: Option<&str>) -> TimeoutConfig {
        if let Some(d) = domain {
            if let Some(ovr) = self.workflow.domains.get(d) {
                return TimeoutConfig {
                    analyzer_timeout_secs: ovr
                        .analyzer_timeout_secs
                        .unwrap_or(self.workflow.timeouts.analyzer_timeout_secs),
                    worker_timeout_secs: ovr
                        .worker_timeout_secs
                        .unwrap_or(self.workflow.timeouts.worker_timeout_secs),
                    synthesizer_timeout_secs: ovr
                        .synthesizer_timeout_secs
                        .unwrap_or(self.workflow.timeouts.synthesizer_timeout_secs),
                };
            }
        }
        self.workflow.timeouts.clone()
    }
}
