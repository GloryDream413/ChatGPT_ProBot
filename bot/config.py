import yaml
import dotenv
from pathlib import Path

config_dir = Path(__file__).parent.parent.resolve() / "config"

# load yaml config
with open(config_dir / "config.yml", 'r') as f:
    config_yaml = yaml.safe_load(f)

# load .env config
config_env = dotenv.dotenv_values(config_dir / "config.env")

# config parameters
telegram_token = config_yaml["telegram_token"]
openai_api_key = config_yaml["openai_api_key"]
use_chatgpt_api = config_yaml.get("use_chatgpt_api", True)
allowed_telegram_usernames = config_yaml["allowed_telegram_usernames"]
new_dialog_timeout = config_yaml["new_dialog_timeout"]
enable_message_streaming = config_yaml.get("enable_message_streaming", True)
initial_token_balance = config_yaml["initial_token_balance"]
mongodb_uri = f"mongodb://mongo:{config_env['MONGODB_PORT']}"

# admin
admin_chat_id = config_yaml["admin_chat_id"]
admin_usernames = config_yaml["admin_usernames"]

# chat_modes
with open(config_dir / "chat_modes.yml", 'r') as f:
    chat_modes = yaml.safe_load(f)

# api prices
chatgpt_price_per_1000_tokens = config_yaml.get("chatgpt_price_per_1000_tokens", 0.002)
gpt_price_per_1000_tokens = config_yaml.get("gpt_price_per_1000_tokens", 0.02)
whisper_price_per_1_min = config_yaml.get("whisper_price_per_1_min", 0.006)

# payments
payment_methods = config_yaml["payment_methods"]
products = config_yaml["products"]
