from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


def load_system_prompt() -> str:
    try:
        system_prompt_path = get_abs_path(prompts_conf["main_prompt_path"])
    except KeyError as e:
        logger.error(
            f"[load_system_prompt]Key {str(e)} not found in prompts configuration."
        )
        raise e

    try:
        with open(system_prompt_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError as e:
        logger.error(f"[load_system_prompt]Error when parsing system prompt, {str(e)}")
        raise e


def load_rag_prompt() -> str:
    try:
        rag_prompt_path = get_abs_path(prompts_conf["rag_summarize_path"])
    except KeyError as e:
        logger.error(
            f"[load_rag_prompt]Key {str(e)} not found in prompts configuration."
        )
        raise e

    try:
        with open(rag_prompt_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError as e:
        logger.error(
            f"[load_rag_prompt]Error when parsing RAG summarize prompt, {str(e)}"
        )
        raise e


def load_report_prompt() -> str:
    try:
        report_prompt_path = get_abs_path(prompts_conf["report_prompt_path"])
    except KeyError as e:
        logger.error(
            f"[load_report_prompt]Key {str(e)} not found in prompts configuration."
        )
        raise e

    try:
        with open(report_prompt_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError as e:
        logger.error(f"[load_report_prompt]Error when parsing report prompt, {str(e)}")
        raise e


if __name__ == "__main__":
    print(load_system_prompt())
    print(load_rag_prompt())
    print(load_report_prompt())
