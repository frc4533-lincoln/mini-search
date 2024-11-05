use std::error::Error;

use spider::{
    packages::scraper::{Html, Selector},
    url::Url,
    website::Website,
};
use tantivy::TantivyDocument;

use crate::{index::SearchIndex, transformers::SentEmbed};

pub async fn crawl(
    site: &str,
    mut is_good_url: impl FnMut(Url) -> bool,
    se: &mut SentEmbed,
    index: &SearchIndex,
) -> Result<usize, Box<dyn Error>> {
    let mut w = Website::new(site);
    w.with_respect_robots_txt(true);
    w.with_block_assets(true);
    w.with_limit(10_000);
    //w.with_limit(40);

    w.scrape().await;

    let mut writer = index.writer()?;

    let mut total = 0usize;

    'index: for page in w.get_pages().unwrap().iter() {
        if total == 10_000 {
            break 'index;
        }
        if let Some(url) = page.get_url_parsed() {
            if is_good_url(url.clone()) {
                let html = Html::parse_document(&page.get_html());

                let body = html
                    .select(&Selector::parse("p, h1, h2, h3, h4").unwrap())
                    .map(|elem| elem.text().collect::<Vec<_>>().join(" "))
                    .collect::<Vec<_>>()
                    .join(" ");

                let title = html
                    .select(&Selector::parse("title").unwrap())
                    .next()
                    .map(|x| x.inner_html())
                    .unwrap_or(url.to_string());

                let embedding = se.generate_embedding(title.clone())?;
                let embedding: Vec<u8> = unsafe {
                    core::slice::from_raw_parts(
                        embedding.as_ptr() as *const u8,
                        embedding.len() * 4,
                    )
                    .to_vec()
                };

                let schema = index.schema();
                let mut doc = TantivyDocument::new();
                doc.add_text(schema.get_field("url")?, url);
                doc.add_text(schema.get_field("title")?, title);
                doc.add_text(schema.get_field("body")?, body);
                doc.add_bytes(schema.get_field("embedding")?, embedding);

                writer.add_document(doc)?;
                writer.commit()?;
                total += 1;
            }
        }
    }

    Ok(total)
}
