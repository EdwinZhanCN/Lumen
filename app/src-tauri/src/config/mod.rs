use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ConfigMetadata {
    pub filename: String,
    pub path: String,
    pub service: String,
    pub modified: String,
}

/// Expands ~ to the user's home directory
pub fn expand_tilde(path: &str) -> Result<PathBuf, String> {
    if path.starts_with("~/") {
        let home =
            dirs::home_dir().ok_or_else(|| "Could not determine home directory".to_string())?;
        Ok(home.join(&path[2..]))
    } else if path == "~" {
        dirs::home_dir().ok_or_else(|| "Could not determine home directory".to_string())
    } else {
        Ok(PathBuf::from(path))
    }
}

/// Get the default Lumen config directory (~/.lumen)
pub fn get_default_config_dir() -> Result<PathBuf, String> {
    let home = dirs::home_dir().ok_or_else(|| "Could not determine home directory".to_string())?;
    Ok(home.join(".lumen"))
}

/// Ensure a directory exists, creating it if necessary
pub fn ensure_dir_exists(path: &Path) -> Result<(), String> {
    if !path.exists() {
        fs::create_dir_all(path)
            .map_err(|e| format!("Failed to create directory {}: {}", path.display(), e))?;
    }
    Ok(())
}

/// Save configuration content to a file
pub fn save_config(
    filename: &str,
    content: &str,
    cache_dir: Option<&str>,
) -> Result<String, String> {
    // Determine the target directory
    let config_dir = if let Some(dir) = cache_dir {
        expand_tilde(dir)?
    } else {
        get_default_config_dir()?
    };

    // Ensure directory exists
    ensure_dir_exists(&config_dir)?;

    // Construct full path
    let full_path = config_dir.join(filename);

    // Write the file
    fs::write(&full_path, content).map_err(|e| format!("Failed to write config file: {}", e))?;

    Ok(full_path.to_string_lossy().to_string())
}

/// Load all Lumen config files from the config directory
pub fn load_configs(cache_dir: Option<&str>) -> Result<Vec<ConfigMetadata>, String> {
    let config_dir = if let Some(dir) = cache_dir {
        expand_tilde(dir)?
    } else {
        get_default_config_dir()?
    };

    // If directory doesn't exist, return empty list
    if !config_dir.exists() {
        return Ok(vec![]);
    }

    let mut configs = Vec::new();

    // Read directory entries
    let entries =
        fs::read_dir(&config_dir).map_err(|e| format!("Failed to read config directory: {}", e))?;

    for entry in entries {
        let entry = entry.map_err(|e| format!("Failed to read directory entry: {}", e))?;
        let path = entry.path();

        // Check if it's a file matching lumen-*.yaml pattern
        if path.is_file() {
            if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                if filename.starts_with("lumen-") && filename.ends_with(".yaml") {
                    // Extract service name (e.g., "face" from "lumen-face.yaml")
                    let service = filename
                        .strip_prefix("lumen-")
                        .and_then(|s| s.strip_suffix(".yaml"))
                        .unwrap_or("unknown")
                        .to_string();

                    // Get modified time
                    let modified = entry
                        .metadata()
                        .ok()
                        .and_then(|m| m.modified().ok())
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_secs().to_string())
                        .unwrap_or_else(|| "unknown".to_string());

                    configs.push(ConfigMetadata {
                        filename: filename.to_string(),
                        path: path.to_string_lossy().to_string(),
                        service,
                        modified,
                    });
                }
            }
        }
    }

    // Sort by filename
    configs.sort_by(|a, b| a.filename.cmp(&b.filename));

    Ok(configs)
}

/// Read the content of a config file
pub fn read_config(filepath: &str) -> Result<String, String> {
    let path = expand_tilde(filepath)?;
    fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read config file {}: {}", path.display(), e))
}

/// Delete a config file
pub fn delete_config(filepath: &str) -> Result<(), String> {
    let path = expand_tilde(filepath)?;
    fs::remove_file(&path)
        .map_err(|e| format!("Failed to delete config file {}: {}", path.display(), e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_tilde() {
        let result = expand_tilde("~/test/path");
        assert!(result.is_ok());
        assert!(!result.unwrap().to_string_lossy().contains('~'));
    }

    #[test]
    fn test_expand_tilde_without_tilde() {
        let result = expand_tilde("/absolute/path");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), PathBuf::from("/absolute/path"));
    }
}
