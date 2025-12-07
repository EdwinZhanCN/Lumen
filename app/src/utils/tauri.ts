import { invoke } from "@tauri-apps/api/core";

export interface ConfigMetadata {
  filename: string;
  path: string;
  service: string;
  modified: string;
}

/**
 * Save a configuration file to disk
 */
export async function saveConfigFile(
  filename: string,
  content: string,
  cacheDir?: string,
): Promise<string> {
  return invoke<string>("save_config_file", {
    filename,
    content,
    cacheDir: cacheDir || null,
  });
}

/**
 * Load all configuration files from the config directory
 */
export async function loadConfigFiles(
  cacheDir?: string,
): Promise<ConfigMetadata[]> {
  return invoke<ConfigMetadata[]>("load_config_files", {
    cacheDir: cacheDir || null,
  });
}

/**
 * Read the content of a specific config file
 */
export async function readConfigFile(filepath: string): Promise<string> {
  return invoke<string>("read_config_file", { filepath });
}

/**
 * Delete a configuration file
 */
export async function deleteConfigFile(filepath: string): Promise<void> {
  return invoke<void>("delete_config_file", { filepath });
}

/**
 * Get the default config directory path (~/.lumen)
 */
export async function getDefaultConfigDirectory(): Promise<string> {
  return invoke<string>("get_default_config_directory");
}

/**
 * Expand a path containing ~ to absolute path
 */
export async function expandPath(path: string): Promise<string> {
  return invoke<string>("expand_path", { path });
}
