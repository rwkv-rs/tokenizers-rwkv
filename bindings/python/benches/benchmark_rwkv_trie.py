"""Small manual benchmark for the Rust RWKV trie against a pure-Python trie.

Run after building the local extension with `maturin develop --release`:

    python benches/benchmark_rwkv_trie.py

The model is read the way a `tokenizer.json` is read now: the bench writes a canonical `2.0`
config next to itself, and `Tokenizer.from_file` builds the pipeline from it.
"""

import argparse
import json
import tempfile
import time
from pathlib import Path

from tokenizers import Tokenizer

EOD_TOKEN = b"<|endoftext|>"


def bytes_to_unicode() -> dict[int, str]:
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(0xA1, 0xAD))
    byte_values += list(range(0xAE, 0x100))
    codepoints = list(byte_values)
    extra = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    return dict(zip(byte_values, map(chr, codepoints)))


class PythonTrie:
    def __init__(self, vocab: dict[bytes, int]):
        self.root: dict[int | None, object] = {}
        for pattern, token_id in vocab.items():
            if token_id == 0:
                continue
            node = self.root
            for byte in pattern:
                node = node.setdefault(byte, {})
            node[None] = token_id

    def encode(self, text: str) -> list[int]:
        source = text.encode()
        result = []
        cursor = 0
        while cursor < len(source):
            node = self.root
            end = cursor
            token_id = None
            probe = cursor
            while probe < len(source) and source[probe] in node:
                node = node[source[probe]]
                probe += 1
                if None in node:
                    token_id = node[None]
                    end = probe
            if token_id is None:
                raise ValueError(f"missing byte at offset {cursor}")
            result.append(token_id)
            cursor = end
        return result


def build_vocab() -> tuple[dict[bytes, int], dict[str, int]]:
    raw_vocab = {bytes([byte]): byte + 1 for byte in range(256)}
    raw_vocab[EOD_TOKEN] = 0
    raw_vocab[b"RWKV "] = 300
    raw_vocab["你".encode()] = 301
    raw_vocab["好".encode()] = 302
    encoder = bytes_to_unicode()
    bytelevel_vocab = {"".join(encoder[byte] for byte in pattern): token_id for pattern, token_id in raw_vocab.items()}
    return raw_vocab, bytelevel_vocab


def build_tokenizer(bytelevel_vocab: dict[str, int], directory: Path) -> Tokenizer:
    config = {
        "version": "2.0",
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": None,
        # ByteLevel is the only component RWKV World needs; it inverts the byte map on decode.
        "decoder": {"type": "ByteLevel"},
        "model": {"type": "RwkvTrie", "vocab": bytelevel_vocab},
        "padding": None,
    }
    path = directory / "tokenizer.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return Tokenizer.from_file(str(path))


def best_time(callback, repeats: int = 3) -> float:
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        callback()
        timings.append(time.perf_counter() - start)
    return min(timings)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--megabytes", type=float, default=4.0)
    parser.add_argument("--min-speedup", type=float, default=2.0)
    args = parser.parse_args()

    raw_vocab, bytelevel_vocab = build_vocab()
    python_trie = PythonTrie(raw_vocab)
    with tempfile.TemporaryDirectory() as directory:
        tokenizer = build_tokenizer(bytelevel_vocab, Path(directory))

        line = "RWKV 你好 — fast UTF-8 bytes.\n"
        target_bytes = max(1, int(args.megabytes * 1024 * 1024))
        lines_per_document = 256
        document = line * lines_per_document
        documents = [document] * max(1, target_bytes // len(document.encode()))
        total_bytes = sum(len(text.encode()) for text in documents)

        expected = python_trie.encode(document)
        encoding = tokenizer.encode(document, add_special_tokens=False)
        assert encoding.ids == expected
        assert tokenizer.decode(encoding.ids, skip_special_tokens=False) == document
        tokenizer.encode_batch(documents[:2], add_special_tokens=False)

        python_seconds = best_time(lambda: [python_trie.encode(text) for text in documents])
        rust_seconds = best_time(lambda: tokenizer.encode_batch(documents, add_special_tokens=False))
    speedup = python_seconds / rust_seconds
    mib = total_bytes / (1024 * 1024)
    print(f"data: {mib:.2f} MiB across {len(documents)} documents")
    print(f"python trie: {mib / python_seconds:.2f} MiB/s")
    print(f"rust RwkvTrie: {mib / rust_seconds:.2f} MiB/s")
    print(f"speedup: {speedup:.2f}x")
    if speedup < args.min_speedup:
        raise SystemExit(f"speedup {speedup:.2f}x is below required {args.min_speedup:.2f}x")


if __name__ == "__main__":
    main()
