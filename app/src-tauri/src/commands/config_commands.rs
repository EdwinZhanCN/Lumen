use crate::config::{self, ConfigMetadata};
use tauri::command;

#[command]
pub async fn save_config_file(
    filename: String,
    content: String,
    cache_dir: Option<String>,
) -> Result<String, String> {
    let cache_dir_ref = cache_dir.as_deref();
    config::save_config(&filename, &content, cache_dir_ref)
}

#[command]
pub async fn load_config_files(cache_dir: Option<String>) -> Result<Vec<ConfigMetadata>, String> {
    let cache_dir_ref = cache_dir.as_deref();
    config::load_configs(cache_dir_ref)
}

#[command]
pub async fn read_config_file(filepath: String) -> Result<String, String> {
    config::read_config(&filepath)
}

#[command]
pub async fn delete_config_file(filepath: String) -> Result<(), String> {
    config::delete_config(&filepath)
}

#[command]
pub async fn get_default_config_directory() -> Result<String, String> {
    config::get_default_config_dir().map(|p| p.to_string_lossy().to_string())
}

#[command]
pub async fn expand_path(path: String) -> Result<String, String> {
    config::expand_tilde(&path).map(|p| p.to_string_lossy().to_string())
}
