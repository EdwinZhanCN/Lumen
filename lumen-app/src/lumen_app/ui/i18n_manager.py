import i18n
import os

class I18nManager:
    def __init__(self, locales_dir, initial_locale='en'):
        i18n.load_path.append(locales_dir)
        i18n.set('file_format', 'yml')
        i18n.set('filename_format', '{locale}.{format}')
        self.locales_dir = locales_dir
        self.available_locales = [f.split('.')[0] for f in os.listdir(locales_dir) if f.endswith('.yml')]
        self.set_locale(initial_locale)

    def set_locale(self, locale):
        if locale in self.available_locales:
            i18n.set('locale', locale)
            self.current_locale = locale
        else:
            raise ValueError(f"Locale '{locale}' not found. Available locales: {self.available_locales}")

    def t(self, key):
        return i18n.t(key)

# Get the absolute path to the locales directory
locales_path = os.path.join(os.path.dirname(__file__), 'locales')
i18n_manager = I18nManager(locales_path)

def get_i18n_manager():
    return i18n_manager

def t(key):
    return i18n_manager.t(key)
