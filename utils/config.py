import os
import json

class Config:
    def __init__(self):
        self.config_dir = os.path.expanduser("~/.imagebotx")
        self.config_file = os.path.join(self.config_dir, "config.json")
        self.data = {}
        self.load()
    
    def load(self):
        """Load configuration from file"""
        # Create config directory if it doesn't exist
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
        
        # Try to load existing config
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}
    
    def save(self):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(self.data, f, indent=4)
    
    def get(self, key, default=None):
        """Get a config value"""
        return self.data.get(key, default)
    
    def set(self, key, value):
        """Set a config value"""
        self.data[key] = value
        self.save()

# Singleton instance
_config = None

def get_config():
    """Get the config singleton"""
    global _config
    if _config is None:
        _config = Config()
    return _config
