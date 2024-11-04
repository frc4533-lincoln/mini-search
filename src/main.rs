use std::{error::Error, sync::Arc, time::Instant};

use axum::{extract::{Query, State}, response::{Html, IntoResponse}, routing::get, Router};
use crawler::Crawler;
use index::SearchIndex;
use tantivy::{collector::TopDocs, query::QueryParser, schema::{Schema, Value}, Document, IndexReader, Searcher, SnippetGenerator, TantivyDocument};
use tera::{Context, Tera};
use tokio::{net::TcpListener, sync::Mutex};
use transformers::SentenceEmbeddings;

#[macro_use]
extern crate log;
#[macro_use]
extern crate tokio;
extern crate axum;
#[macro_use]
extern crate serde;
extern crate candle_core;
extern crate candle_nn;
extern crate candle_transformers;
extern crate env_logger;
extern crate spider;
extern crate tantivy;
extern crate tokenizers;
extern crate tera;

mod crawler;
mod index;
mod transformers;

#[derive(Deserialize)]
struct SearchParams {
    #[serde(rename(deserialize = "q"))]
    query: Option<String>,
}

#[derive(Serialize, Clone)]
struct Res {
    url: String,
    title: String,
    snippet: String,
}
#[derive(Serialize)]
struct SearchRes {
    query: String,
    results: Vec<Res>,
    time: String,
}

async fn search(State(st): State<AppState>, Query(params): Query<SearchParams>) -> impl IntoResponse {
    if let Some(q) = params.query {
        let AppState { reader, parser, schema, templates, se } = st;

        let total_st = Instant::now();

        let jh = {
            let se = se.clone();
            let query = q.clone();
            tokio::spawn(async move {
                let st = Instant::now();
                (se.lock().await.generate_embedding(query).ok(), st.elapsed())
            })
        };

        let searcher = reader.searcher();

        let parse_st = Instant::now();
        let query = parser.parse_query(&q).expect("failed to parse query");
        let parse_tm = parse_st.elapsed();

        let search_st = Instant::now();
        let results_raw = searcher.search(&query, &TopDocs::with_limit(20)).expect("search failed");
        let search_tm = search_st.elapsed();

        let mut results = Vec::new();

        for (_, doc_addr) in results_raw {
            let doc = searcher.doc::<TantivyDocument>(doc_addr).expect("couldn't get doc");
            let embedding = doc.get_first(schema.get_field("embedding").unwrap()).unwrap().as_bytes().unwrap();
            let embedding = unsafe {
                std::slice::from_raw_parts(embedding.as_ptr() as *const f32, embedding.len() / 4).to_vec()
            };
            
            let url = doc.get_first(schema.get_field("url").unwrap()).unwrap().as_str().unwrap().to_string();
            let title = doc.get_first(schema.get_field("title").unwrap()).unwrap().as_str().unwrap().to_string();
            let snippet = SnippetGenerator::create(&searcher, &query, schema.get_field("body").unwrap()).unwrap().snippet_from_doc(&doc).to_html();

            results.push((embedding.clone(), Res {
                url,
                title,
                snippet,
            }));
        }

        let (embedding, embedding_gen_tm) = jh.await.expect("something broke");

        let sort_st = Instant::now();
        se.lock().await.sort_by_similarity(embedding.unwrap(), results.iter().map(|x| x.0.clone())).unwrap();
        let sort_tm = sort_st.elapsed();

        let total_tm = total_st.elapsed();

        Html(templates.render("index.html", &Context::from_serialize(SearchRes {
            query: q,
            results: Vec::from_iter(results.iter().map(|x| x.1.clone())),
            time: format!(
                "{total_tm:?} = parse({parse_tm:?}) + search({search_tm:?}) + embedding({embedding_gen_tm:?}) + sort({sort_tm:?})",
                //(parse_tm + search_tm + embedding_gen_tm + sort_tm)
            ),
        }).unwrap()).unwrap()).into_response()
    } else {
        Html(st.templates.render("index.html", &Context::default()).unwrap()).into_response()
    }
}

#[derive(Clone)]
struct AppState {
    reader: IndexReader,
    parser: QueryParser,
    schema: Schema,
    se: Arc<Mutex<SentenceEmbeddings>>,
    templates: Tera,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();

    let tera = Tera::new("views/*.html").unwrap();

    let mut se = SentenceEmbeddings::new()?;
    for _ in 0..70 {
        let st = Instant::now();
        se.generate_embedding("Hello, world!".to_string())?;
        println!("{:?}", st.elapsed());
    }

    let index = SearchIndex::new().await.unwrap();

    //Crawler::new(&mut se, &index).await.unwrap();

    let r = Router::new().route("/", get(search)).with_state(AppState {
        reader: index.reader(),
        parser: index.query_parser(),
        schema: index.schema(),
        se: Arc::new(Mutex::new(se)),
        templates: tera,
    });

    let srv = axum::serve(
        TcpListener::bind("0.0.0.0:8080").await?,
        r.into_make_service(),
    );

    // Run the web server until a fatal error is encountered
    // or ctrl+c is pressed
    tokio::select! {
        _ = srv => {}
        _ = tokio::signal::ctrl_c() => {}
    }

    Ok(())
}
