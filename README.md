# HFDL - Yet Another Huggingface downloader tool

A python module and command line tool do directly download hugginface model.

# Usage:

As a script:
```python
from hfdl import download_model

download_model(hf_repo="TheBloke/Llama-2-13B-chat-GGML",
               model_filename="llama-2-13b-chat.ggmlv3.q2_K.bin",
               destination_folder="models")
```

As a command line tool:
```sh
hfdl \
    --hf-repo="TheBloke/Llama-2-13B-chat-GGML" \
    --filename="llama-2-13b-chat.ggmlv3.q2_K.bin"
```
