import hashlib
import os
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def get_file_md5_hex(
    filepath: str,
):  # This function is used to get the md5 hex of a file
    if not os.path.exists(filepath):
        logger.error(f"[md5_cal]File {filepath} does not exist.")
        return None

    if not os.path.isfile(filepath):
        logger.error(f"[md5_cal]Path {filepath} is not a file.")
        return None

    md5_obj = hashlib.md5()

    chunk_size = 4096
    try:
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                md5_obj.update(chunk)
    except Exception as e:
        logger.error(f"[md5_cal]Error reading file {filepath}: {str(e)}")
        return None

    return md5_obj.hexdigest()


def listdir_with_allowed_type(
    path: str, allowed_types: tuple[str]
):  # This function is used to list the files in a directory with allowed types
    files = []

    if not os.path.isdir(path):
        logger.error(f"[listdir]Path {path} is not a directory.")
        return files

    for file in os.listdir(path):
        if file.endswith(allowed_types):
            files.append(os.path.join(path, file))

    return tuple(files)


def pdf_loader(
    filepath: str, password: str = None
):  # This function is used to load a pdf file and return the text content
    return PyPDFLoader(filepath, password=password).load()


def txt_loader(
    filepath: str,
):  # This function is used to load a txt file and return the text content
    return TextLoader(filepath, encoding="utf-8").load()
