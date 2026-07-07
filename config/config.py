import json
def load_primary_model():
    """Load the recommended model from config"""
    try:
        with open('config/model_config.json', 'r') as f:
            config = json.load(f)
            return config['primary_model']
    except:
        return 'gpt-4o-mini'