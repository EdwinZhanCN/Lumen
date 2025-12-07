mod commands;
mod config;

use commands::*;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            save_config_file,
            load_config_files,
            read_config_file,
            delete_config_file,
            get_default_config_directory,
            expand_path,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
