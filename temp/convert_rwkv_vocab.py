"""Convert BlinkDL's RWKV World vocabulary text file to ByteLevel JSON.

Example:

    python temp/convert_rwkv_vocab.py \
        rwkv_vocab_v20230424.txt \
        rwkv-vocab.json

The output is accepted by ``tokenizers.models.RwkvTrie``. Token ID 0 is
reserved for ``<|endoftext|>`` and is added when the source file does not
define it.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

EOD_TOKEN = b"<|endoftext|>"
MAX_TOKEN_ID = 2**32 - 1


def bytes_to_unicode() -> dict[int, str]:
    """Return the reversible byte alphabet used by Hugging Face ByteLevel."""
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


def parse_vocab_line(line: str, line_number: int) -> tuple[int, bytes]:
    first_space = line.find(" ")
    last_space = line.rfind(" ")
    if first_space <= 0 or last_space <= first_space:
        raise ValueError(f"line {line_number}: expected '<id> <literal> <length>'")

    try:
        token_id = int(line[:first_space])
        expected_length = int(line[last_space + 1 :])
    except ValueError as error:
        raise ValueError(
            f"line {line_number}: invalid token ID or byte length"
        ) from error

    if not 0 <= token_id <= MAX_TOKEN_ID:
        raise ValueError(
            f"line {line_number}: token ID {token_id} is outside the u32 range"
        )
    if expected_length < 0:
        raise ValueError(f"line {line_number}: byte length cannot be negative")

    literal = line[first_space + 1 : last_space]
    try:
        value = ast.literal_eval(literal)
    except (SyntaxError, ValueError) as error:
        raise ValueError(
            f"line {line_number}: invalid Python string or bytes literal"
        ) from error

    if isinstance(value, str):
        token = value.encode("utf-8")
    elif isinstance(value, bytes):
        token = value
    else:
        raise TypeError(
            f"line {line_number}: token literal must evaluate to str or bytes, "
            f"got {type(value).__name__}"
        )

    if not token:
        raise ValueError(f"line {line_number}: empty tokens are not supported")
    if len(token) != expected_length:
        raise ValueError(
            f"line {line_number}: declared length {expected_length} does not match "
            f"the token's {len(token)} UTF-8 bytes"
        )
    return token_id, token


def read_rwkv_vocab(path: Path) -> dict[int, bytes]:
    tokens: dict[int, bytes] = {}
    token_ids_by_bytes: dict[bytes, int] = {}

    with path.open(encoding="utf-8") as vocab_file:
        for line_number, raw_line in enumerate(vocab_file, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue

            token_id, token = parse_vocab_line(line, line_number)
            if token_id in tokens:
                raise ValueError(f"line {line_number}: duplicate token ID {token_id}")
            if token in token_ids_by_bytes:
                previous_id = token_ids_by_bytes[token]
                raise ValueError(
                    f"line {line_number}: token bytes duplicate token ID {previous_id}"
                )

            tokens[token_id] = token
            token_ids_by_bytes[token] = token_id

    if not tokens:
        raise ValueError("the source vocabulary is empty")

    if 0 not in tokens:
        if EOD_TOKEN in token_ids_by_bytes:
            existing_id = token_ids_by_bytes[EOD_TOKEN]
            raise ValueError(
                f"cannot reserve ID 0 for <|endoftext|>: the token already uses ID {existing_id}"
            )
        tokens[0] = EOD_TOKEN

    return tokens


def to_bytelevel_vocab(tokens: dict[int, bytes]) -> dict[str, int]:
    encoder = bytes_to_unicode()
    vocab: dict[str, int] = {}

    for token_id, token in sorted(tokens.items()):
        encoded = "".join(encoder[byte] for byte in token)
        if encoded in vocab:
            raise ValueError(
                f"ByteLevel token for ID {token_id} duplicates token ID {vocab[encoded]}"
            )
        vocab[encoded] = token_id

    return vocab


def write_vocab(vocab: dict[str, int], path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(
            f"output already exists: {path} (pass --force to replace it)"
        )
    if not path.parent.exists():
        raise FileNotFoundError(f"output directory does not exist: {path.parent}")

    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
            json.dump(vocab, output_file, ensure_ascii=False, separators=(",", ":"))
            output_file.write("\n")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert BlinkDL rwkv_vocab_v20230424.txt into the Hugging Face "
            "ByteLevel vocab JSON accepted by RwkvTrie."
        )
    )
    parser.add_argument("input", type=Path, help="BlinkDL RWKV vocabulary text file")
    parser.add_argument("output", type=Path, help="destination ByteLevel vocab JSON")
    parser.add_argument(
        "--force", action="store_true", help="replace an existing output file"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokens = read_rwkv_vocab(args.input)
    vocab = to_bytelevel_vocab(tokens)
    write_vocab(vocab, args.output, args.force)

    print(
        f"converted {len(vocab):,} tokens from {args.input} to {args.output} "
        f"(IDs {min(vocab.values())}..{max(vocab.values())})"
    )


if __name__ == "__main__":
    main()
