use std::{error::Error, fs::File, io::Read};

use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{self, BertModel};
use tokenizers::{PaddingParams, Tokenizer};

// This code is pretty heavily based on the candle example:
// <https://github.com/huggingface/candle/blob/530ab96036604b125276433b67ebb840e841aede/candle-examples/examples/bert/main.rs#L146C9-L205C10>

/// Sentence embeddings
pub struct SentEmbed {
    tokenizer: Tokenizer,
    bert: BertModel,
}
impl SentEmbed {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let mut buf_model = Vec::new();
        let mut buf_config = Vec::new();
        let mut buf_tokenizer = Vec::new();

        // Load all the necessary files into memory
        File::open("model.safetensors")?.read_to_end(&mut buf_model)?;
        File::open("config.json")?.read_to_end(&mut buf_config)?;
        File::open("tokenizer.json")?.read_to_end(&mut buf_tokenizer)?;

        let mut tokenizer = Tokenizer::from_bytes(buf_tokenizer).unwrap();

        // Deserialize transformers config
        let config: bert::Config = serde_json::from_slice(&buf_config)?;

        // Initialize transformers model
        let vb = VarBuilder::from_buffered_safetensors(buf_model, DType::F32, &Device::Cpu)?;
        let bert = BertModel::load(vb, &config)?;

        if let Some(pp) = tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest
        } else {
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            tokenizer.with_padding(Some(pp));
        }

        Ok(Self { tokenizer, bert })
    }

    /// Generate an embedding for the given sentence
    ///
    /// This returns a raw tensor without any conversions.
    fn gen_embedding(&mut self, sentence: String) -> Result<Tensor, Box<dyn Error>> {
        let tokens = self
            .tokenizer
            .encode(vec![sentence], true)
            .expect("aaaaaaaaaaa");

        let embeddings = self.run_inference(&[tokens])?;

        Ok(embeddings.get(0)?)
    }

    /// Generate an embedding for the given sentence
    ///
    /// This converts the embedding to a [Vec] internally.
    pub fn generate_embedding(&mut self, sentence: String) -> Result<Vec<f32>, Box<dyn Error>> {
        Ok(self.gen_embedding(sentence)?.to_vec1()?)
    }

    /// Run inference on some tokens
    fn run_inference(&self, tokens: &[tokenizers::Encoding]) -> Result<Tensor, Box<dyn Error>> {
        let token_ids = tokens
            .iter()
            .map(|tokens| {
                Tensor::new(tokens.get_ids(), &self.bert.device).expect("something's pretty wrong")
            })
            .collect::<Vec<_>>();

        let attention_mask = tokens
            .iter()
            .map(|tokens| {
                Tensor::new(tokens.get_attention_mask(), &self.bert.device)
                    .expect("something's pretty wrong")
            })
            .collect::<Vec<_>>();

        let token_ids = Tensor::stack(&token_ids, 0)?;
        let attention_mask = Tensor::stack(&attention_mask, 0)?;
        let token_type_ids = token_ids.zeros_like()?;

        let embeddings = self
            .bert
            .forward(&token_ids, &token_type_ids, Some(&attention_mask))?;

        // Apply some avg-pooling by taking the mean embedding value for all tokens (including padding)
        let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
        let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
        let embeddings = normalize_l2(&embeddings)?;

        Ok(embeddings)
    }

    /// Sort a set of candidates by their similarity to the given query
    pub fn sort_by_similarity(
        &mut self,
        query: Vec<f32>,
        candidates: impl Iterator<Item = Vec<f32>>,
    ) -> Result<Vec<(usize, f32)>, Box<dyn Error>> {
        let qe = Tensor::from_vec(query, Shape::from_dims(&[384]), &self.bert.device)?;

        // Calculate cosine similarities for each candidate
        let mut similarities = candidates
            .enumerate()
            .map(|(i, candidate)| {
                // Rebuild a tensor for candidate embedding
                let ce =
                    Tensor::from_slice(&candidate, Shape::from_dims(&[384]), &self.bert.device)?;

                // Calculate sum of elements for q*c, q^2, and c^2
                let sum_qc = (&qe * &ce)?.sum_all()?.to_scalar::<f32>()?;
                let sum_qq = (&qe * &qe)?.sum_all()?.to_scalar::<f32>()?;
                let sum_cc = (&ce * &ce)?.sum_all()?.to_scalar::<f32>()?;

                let cos_similarity = sum_qc / (sum_qq * sum_cc).sqrt();

                Ok((i, cos_similarity))
            })
            .collect::<Result<Vec<(usize, f32)>, Box<dyn Error>>>()?;

        // Sort each candidate by cosine similarity
        similarities.sort_by(|a, b| b.1.total_cmp(&a.1));

        Ok(similarities)
    }
}

// <https://github.com/huggingface/candle/blob/530ab96036604b125276433b67ebb840e841aede/candle-examples/examples/bert/main.rs#L210C1-L212C2>
pub fn normalize_l2(v: &Tensor) -> Result<Tensor, Box<dyn Error>> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}
