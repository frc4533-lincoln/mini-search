use std::{error::Error, ops::Deref};

use spider::{
    packages::scraper::{Html, Selector},
    website::Website,
    Client,
};
use tantivy::{Document, IndexWriter, TantivyDocument};

use crate::{index::SearchIndex, transformers::SentenceEmbeddings};

pub struct Crawler {}
impl Crawler {
    pub async fn new(
        se: &mut SentenceEmbeddings,
        index: &SearchIndex,
    ) -> Result<Self, Box<dyn Error>> {
        let mut w = Website::new("https://docs.python.org")
            .with_respect_robots_txt(true)
            .with_block_assets(true)
            .with_limit(500)
            .build()?;

        w.scrape_smart().await;

        let mut writer = index.writer()?;

        for page in w.get_pages().unwrap().iter() {
            let url = page.get_url();
            info!("scraping {url}");

            let html = Html::parse_document(&page.get_html());
            let body = html
                .select(&Selector::parse("p, h1, h2, h3, h4").unwrap())
                .map(|elem| {
                    elem
                .text()
                .collect::<Vec<_>>()
                .join(" ")
                })
                .collect::<Vec<_>>()
                .join(" ");
            let title = html
                .select(&Selector::parse("title").unwrap())
                .next()
                .map(|x| x.inner_html())
                .unwrap_or(url.to_string());
            let embedding = se.generate_embedding(title.clone())?;
            let embedding: Vec<u8> = unsafe {
                core::slice::from_raw_parts(embedding.as_ptr() as *const u8, embedding.len() * 4).to_vec()
            };

            let schema = index.schema();
            let mut doc = TantivyDocument::new();
            doc.add_text(schema.get_field("url")?, url);
            doc.add_text(schema.get_field("title")?, title);
            doc.add_text(schema.get_field("body")?, body);
            doc.add_bytes(schema.get_field("embedding")?, embedding);

            writer.add_document(doc)?;
            writer.commit()?;
        }

        Ok(Self {})
    }
}
