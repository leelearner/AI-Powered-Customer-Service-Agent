import yaml, os, re
from utils.path_tool import get_abs_path
from dotenv import load_dotenv

load_dotenv()


def resolve_env_vars(config):
    if isinstance(config, str):
        return re.sub(
            r"\$\{(\w+)\}",
            lambda m: os.environ.get(m.group(1), m.group(0)),
            config,
        )
    elif isinstance(config, dict):
        return {k: resolve_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [resolve_env_vars(item) for item in config]
    return config


def load_rag_config(
    config_path: str = get_abs_path("config/rag.yml"), encoding: str = "utf-8"
) -> dict:
    with open(config_path, "r", encoding=encoding) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        return resolve_env_vars(config)


def load_chroma_config(
    config_path: str = get_abs_path("config/chroma.yml"), encoding: str = "utf-8"
) -> dict:
    with open(config_path, "r", encoding=encoding) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        return resolve_env_vars(config)


def load_prompts_config(
    config_path: str = get_abs_path("config/prompts.yml"), encoding: str = "utf-8"
) -> dict:
    with open(config_path, "r", encoding=encoding) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        return resolve_env_vars(config)


def load_agent_config(
    config_path: str = get_abs_path("config/agent.yml"), encoding: str = "utf-8"
) -> dict:
    with open(config_path, "r", encoding=encoding) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        return resolve_env_vars(config)


rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()

if __name__ == "__main__":
    print(rag_conf["chat_model_name"])
