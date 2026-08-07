//! The `RwkvTrie` model: RWKV World's greedy longest match, run over UTF-8 bytes.
//!
//! The vocabulary keys are ByteLevel-encoded byte strings, the alphabet in
//! [`crate::utils::byte_level`], so the model matches raw bytes rather than words. That makes it a
//! whole-sequence model: it is meant to run with no normalizer and no pre-tokenizer, where the
//! single span the pipeline hands over *is* the input.
//!
//! Token id 0 is the end-of-document token. It stays in the vocabulary, so decoding it works, but
//! it is never produced by matching.

use crate::pipeline::{self, ModelScratch, PipelineToken};
use crate::tokenizer::Result;
use crate::utils::byte_level::char_to_byte;
use ahash::AHashMap;
use daachorse::{DoubleArrayAhoCorasick, DoubleArrayAhoCorasickBuilder, MatchKind};
use std::collections::HashMap;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("RwkvTrie vocabulary cannot be empty")]
    EmptyVocabulary,
    #[error("RwkvTrie vocabulary contains an empty token")]
    EmptyToken,
    #[error("RwkvTrie vocabulary contains duplicate token ID {0}")]
    DuplicateTokenId(u32),
    #[error("RwkvTrie token {token:?} contains non-ByteLevel character {character:?}")]
    InvalidByteLevelCharacter { token: String, character: char },
    #[error("RwkvTrie automaton could not be built: {0}")]
    AutomatonBuild(String),
    #[error("RwkvTrie cannot match input at byte offset {0}")]
    MissingByte(usize),
}

#[derive(Clone, PartialEq, Eq)]
pub struct RwkvTrie {
    vocab: AHashMap<String, u32>,
    vocab_r: AHashMap<u32, String>,
    automaton: DoubleArrayAhoCorasick<u32>,
}

impl std::fmt::Debug for RwkvTrie {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        fmt.debug_struct("RwkvTrie")
            .field("vocab", &self.vocab.len())
            .finish()
    }
}

impl RwkvTrie {
    pub fn new(vocab: AHashMap<String, u32>) -> Result<Self> {
        if vocab.is_empty() {
            return Err(Box::new(Error::EmptyVocabulary));
        }

        let mut vocab_r = AHashMap::with_capacity(vocab.len());
        // Id 0 is the document boundary, so it is not a match pattern.
        let mut patterns = Vec::with_capacity(vocab.len().saturating_sub(1));

        for (token, token_id) in &vocab {
            if token.is_empty() {
                return Err(Box::new(Error::EmptyToken));
            }
            if vocab_r.insert(*token_id, token.clone()).is_some() {
                return Err(Box::new(Error::DuplicateTokenId(*token_id)));
            }
            if *token_id == 0 {
                continue;
            }

            let pattern = token
                .chars()
                .map(|character| {
                    char_to_byte(character).ok_or_else(|| {
                        Box::new(Error::InvalidByteLevelCharacter {
                            token: token.clone(),
                            character,
                        }) as crate::tokenizer::Error
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            patterns.push((pattern, *token_id));
        }

        let automaton = DoubleArrayAhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .build_with_values(patterns)
            .map_err(|error| Box::new(Error::AutomatonBuild(error.to_string())))?;
        Ok(Self {
            vocab,
            vocab_r,
            automaton,
        })
    }

    pub fn vocab(&self) -> Vec<(String, u32)> {
        self.vocab_r
            .iter()
            .map(|(id, token)| (token.clone(), *id))
            .collect()
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab_r.get(&id).cloned()
    }

    pub fn get_vocab(&self) -> HashMap<String, u32> {
        self.vocab.clone().into_iter().collect()
    }

    pub fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn for_each_match<F>(&self, sequence: &str, mut callback: F) -> Result<()>
    where
        F: FnMut(u32),
    {
        let mut cursor = 0;
        for matched in self.automaton.leftmost_find_iter(sequence.as_bytes()) {
            if matched.start() != cursor {
                return Err(Box::new(Error::MissingByte(cursor)));
            }
            callback(matched.value());
            cursor = matched.end();
        }
        if cursor != sequence.len() {
            return Err(Box::new(Error::MissingByte(cursor)));
        }
        Ok(())
    }
}

/// The automaton is the model itself, so a call carries no state between spans.
#[derive(Default)]
pub struct RwkvTrieScratch;

impl ModelScratch for RwkvTrieScratch {}

impl pipeline::Model for RwkvTrie {
    type Scratch = RwkvTrieScratch;

    fn init_scratch(&self) -> Self::Scratch {
        RwkvTrieScratch
    }

    fn tokenize_pipeline(
        &self,
        sequence: &str,
        _scratch: &mut Self::Scratch,
        output: &mut Vec<PipelineToken>,
    ) -> Result<()> {
        self.for_each_match(sequence, |token_id| {
            output.push(PipelineToken::from(token_id))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::Model as _;
    use crate::utils::byte_level::BYTES_CHAR_LOOKUP;

    fn token(bytes: &[u8]) -> String {
        bytes
            .iter()
            .map(|byte| BYTES_CHAR_LOOKUP[*byte as usize])
            .collect()
    }

    fn vocab(entries: &[(&[u8], u32)]) -> AHashMap<String, u32> {
        entries
            .iter()
            .map(|(bytes, id)| (token(bytes), *id))
            .collect()
    }

    fn encode(model: &RwkvTrie, text: &str) -> Result<Vec<u32>> {
        let mut output = Vec::new();
        model.tokenize_pipeline(text, &mut RwkvTrieScratch, &mut output)?;
        Ok(output.into_iter().map(u32::from).collect())
    }

    #[test]
    fn greedy_longest_match() {
        let model = RwkvTrie::new(vocab(&[
            (b"<|endoftext|>", 0),
            (b"a", 1),
            (b"b", 2),
            (b"<", 5),
            (b">", 14),
            (b"|", 6),
            (b"e", 7),
            (b"n", 8),
            (b"d", 9),
            (b"o", 10),
            (b"f", 11),
            (b"t", 12),
            (b"x", 13),
            (b"ab", 3),
            (b"aba", 4),
        ]))
        .unwrap();
        // Leftmost-longest: `aba` wins over `ab`, and the trailing `b` still matches alone.
        assert_eq!(encode(&model, "abab").unwrap(), vec![4, 2]);
        // Id 0 is a boundary, never a match, even where the vocabulary spells it out.
        assert!(!encode(&model, "<|endoftext|>").unwrap().contains(&0));
        assert_eq!(encode(&model, "abba").unwrap(), vec![3, 2, 1]);
        assert_eq!(encode(&model, "").unwrap(), Vec::<u32>::new());
    }

    #[test]
    fn multi_byte_tokens_match_through_utf8() {
        let text = "Hi 你好";
        let model = RwkvTrie::new(vocab(&[
            (b"<|endoftext|>", 0),
            (b"Hi ", 1),
            ("你".as_bytes(), 2),
            ("好".as_bytes(), 3),
        ]))
        .unwrap();
        assert_eq!(encode(&model, text).unwrap(), vec![1, 2, 3]);
        assert_eq!(model.id_to_token(2).unwrap(), token("你".as_bytes()));
        assert_eq!(model.token_to_id(&token("好".as_bytes())), Some(3));
        assert_eq!(model.get_vocab_size(), 4);
    }

    #[test]
    fn reports_invalid_vocab_and_missing_bytes() {
        assert!(matches!(
            *RwkvTrie::new(AHashMap::new())
                .unwrap_err()
                .downcast::<Error>()
                .unwrap(),
            Error::EmptyVocabulary
        ));
        assert!(matches!(
            *RwkvTrie::new(AHashMap::from([("🙂".to_string(), 1)]))
                .unwrap_err()
                .downcast::<Error>()
                .unwrap(),
            Error::InvalidByteLevelCharacter { .. }
        ));
        assert!(matches!(
            *RwkvTrie::new(AHashMap::from([(token(b"a"), 1), (token(b"b"), 1),]))
                .unwrap_err()
                .downcast::<Error>()
                .unwrap(),
            Error::DuplicateTokenId(1)
        ));

        let model = RwkvTrie::new(vocab(&[(b"a", 1)])).unwrap();
        assert_eq!(
            encode(&model, "ab").unwrap_err().to_string(),
            "RwkvTrie cannot match input at byte offset 1"
        );
    }
}
