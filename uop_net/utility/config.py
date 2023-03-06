import os
import json


class config():
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = {}
    
    def get_base_config(self):
        base_config = self.load_json(os.path.join(self.config_path, 'base.json'))
        self.config['base'] = base_config

    def load_json(self, json_path):
        with open(json_path, 'r') as f:
            out = json.load(f)
        return out
    
    def export_config(self, save_path):
        with open(save_path, 'w') as f:
            json.dump(self.config, f)
