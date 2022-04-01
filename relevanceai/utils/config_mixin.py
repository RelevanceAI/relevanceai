import tempfile
import shutil
import os
import shutil
from relevanceai.constants import CONFIG_PATH, Config

class ConfigMixin:
    def _create_temp_config(self, config_path):
        tmp = tempfile.NamedTemporaryFile(delete=True)
        shutil.copy2(config_path, tmp.name)
        return tmp.name
    
    @property
    def config_path(self):
        # Create a temporary config path
        if not hasattr(self, "_config_path"):
            # generate
            self._config_path = self._create_temp_config(CONFIG_PATH)
        return self._config_path

    @property
    def config(self):
        if not hasattr(self, "_config"):
            self._config = Config(CONFIG_PATH)
        return self._config

    @config.setter
    def config(self, config: Config):
        self._config = config
