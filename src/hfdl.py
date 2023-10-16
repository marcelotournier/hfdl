"""
Command line tool to download files from huggingface.
"""
import os
import logging
import argparse
from typing import Union, NoneType
from huggingface_hub import hf_hub_download


# Logging config
logging.basicConfig(format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO,
                    )
logger = logging.getLogger(__name__)


def download_model(hf_repo: str,
                   model_filename: str,
                   destination_folder: str="models",
                   token: Union[str, bool, NoneType] = None,
                   ):
    """
    Downloads a model file resource from huggingface
    Params:
    
    hf_repo: The repository ID (e.g., 'TheBloke/mpt-30B-chat-GGML').
    model_filename: The model filename to download (e.g., 'mpt-30b-chat.ggmlv0.q4_1.bin').
    destination_folder (optional): Path to destination of files
    token (optional): Huggingface token, in case of restricted/private models
    """
    local_path = os.path.abspath(destination_folder)
    
    download = hf_hub_download(
        repo_id=hf_repo,
        filename=model_filename,
        local_dir=local_path,
        local_dir_use_symlinks=False,
        cache_dir=None,
        token=token
    )

    logger.info(f"Model '{model_filename}' downloaded and saved to '{download}'")


def hfdl():
    """
    Command line interface for downloading models
    """
    parser = argparse.ArgumentParser(description="Download a model from the Hugging Face Model Hub.")

    parser.add_argument("--hf-repo",
                        required=True,
                        type=str,
                        help="The repository ID (e.g., 'TheBloke/mpt-30B-chat-GGML').",
                        )
    
    parser.add_argument("--filename",
                        required=True,
                        type=str,
                        help="The model filename to download (e.g., 'mpt-30b-chat.ggmlv0.q4_1.bin').",
                        )
    
    parser.add_argument("--destination-dir",
                        type=str,
                        default="./models",
                        help="The folder where the model will be saved (default: './models').",
                        )

    args = parser.parse_args()
    hf_repo = args.hf_repo
    model_filename = args.filename
    destination_folder = args.destination_dir

    download_model(hf_repo, model_filename, destination_folder)


if __name__ == "__main__":
    hfdl()
