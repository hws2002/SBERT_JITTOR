"""
Download Hugging Face tokenizer files for offline use.
"""

from huggingface_hub import snapshot_download


def main():
    snapshot_download(
        repo_id="bert-large-uncased",
        local_dir="./hf/bert-large-uncased",
        local_dir_use_symlinks=False,
        allow_patterns=[
            "tokenizer.json",
            "vocab.txt",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "config.json",
        ],
    )


if __name__ == "__main__":
    main()
