use std::{error::Error, fs::create_dir_all};

use tantivy::{
    query::QueryParser,
    schema::{Schema, FAST, STORED, TEXT},
    store::{Compressor, ZstdCompressor},
    Index, IndexReader, IndexSettings, IndexWriter,
};
use tokio::runtime::Handle as TokioRtHandle;

pub struct SearchIndex {
    schema: Schema,
    index: Index,
    reader: IndexReader,
    parser: QueryParser,
}
impl SearchIndex {
    /// Open the search index (or initialize it, if it doesn't already exist)
    pub async fn new() -> Result<Self, Box<dyn Error>> {
        let mut schema = Schema::builder();

        let _url = schema.add_text_field("url", TEXT | FAST | STORED);
        let title = schema.add_text_field("title", TEXT | FAST | STORED);
        let body = schema.add_text_field("body", TEXT | FAST | STORED);
        let _embedding = schema.add_bytes_field("embedding", FAST | STORED);

        let schema = schema.build();

        let mut index = match Index::open_in_dir("mini-search-index") {
            Ok(index) => index,
            Err(_) => {
                warn!("no existing index found, creating one");
                create_dir_all("mini-search-index").unwrap();
                Index::builder()
                    .schema(schema.clone())
                    .settings(IndexSettings {
                        docstore_compression: Compressor::Zstd(ZstdCompressor {
                            compression_level: Some(3),
                        }),
                        ..Default::default()
                    })
                    .create_in_dir("mini-search-index")
                    .unwrap()
            }
        };

        // Use as many threads as Tokio is using, since it gets that from num_cpu
        index.set_multithread_executor(TokioRtHandle::current().metrics().num_workers())?;

        let parser = QueryParser::for_index(&index, vec![title, body]);

        let reader = index.reader()?;

        Ok(Self {
            schema,
            index,
            parser,
            reader,
        })
    }
    pub fn schema(&self) -> Schema {
        self.schema.clone()
    }
    pub fn writer(&self) -> Result<IndexWriter, Box<dyn Error>> {
        Ok(self.index.writer(100_000_000)?)
    }
    pub fn reader(&self) -> IndexReader {
        self.reader.clone()
    }
    pub fn query_parser(&self) -> QueryParser {
        self.parser.clone()
    }
}
