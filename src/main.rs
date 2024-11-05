use std::{
    error::Error,
    sync::Arc,
    time::{Duration, Instant},
};

use axum::{
    extract::{Query, State},
    response::{Html, IntoResponse},
    routing::get,
    Router,
};
use crawler::Crawler;
use index::SearchIndex;
use spider::hashbrown::HashMap;
use tantivy::{
    collector::TopDocs,
    query::QueryParser,
    schema::{Schema, Value},
    IndexReader, SnippetGenerator, TantivyDocument,
};
use tera::{Context, Tera};
use tokio::{net::TcpListener, sync::Mutex};
use transformers::SentEmbed;

#[macro_use]
extern crate log;
extern crate axum;
extern crate tokio;
#[macro_use]
extern crate serde;
extern crate candle_core;
extern crate candle_nn;
extern crate candle_transformers;
extern crate env_logger;
extern crate spider;
extern crate tantivy;
extern crate tera;
extern crate tokenizers;

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

#[derive(Deserialize, Serialize)]
struct MiniDoc {
    url: String,
    title: String,
    body: String,
    embedding: Vec<u8>,
}

async fn search(
    State(st): State<AppState>,
    Query(params): Query<SearchParams>,
) -> impl IntoResponse {
    if let Some(q) = params.query {
        let AppState {
            reader,
            parser,
            schema,
            templates,
            se,
        } = st;
        let mut templates = templates.clone();
        templates.full_reload().unwrap();

        let mut snippet_gen_tm = Duration::default();

        let total_st = Instant::now();

        // Spawn a future to generate an embedding for the search query
        // and keep the join handle for later
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
        let results_raw = searcher
            .search(&query, &TopDocs::with_limit(20))
            .expect("search failed");
        let search_tm = search_st.elapsed();

        let mut results = Vec::new();

        // Fetch documents from the search index and extract their embeddings
        let fetch_st = Instant::now();
        let docs_with_embeddings: Vec<(Vec<f32>, TantivyDocument)> = results_raw
            .iter()
            .map(|&(_, doc_addr)| {
                let doc = searcher
                    .doc::<TantivyDocument>(doc_addr)
                    .expect("couldn't get doc");
                let embedding = doc
                    .get_first(schema.get_field("embedding").unwrap())
                    .unwrap()
                    .as_bytes()
                    .unwrap();
                // Convert the Vec<u8> storage back to Vec<f32>
                // This is safe, as long as the input size is a multiple of 4 bytes
                let embedding = unsafe {
                    std::slice::from_raw_parts(
                        embedding.as_ptr() as *const f32,
                        embedding.len() / 4,
                    )
                    .to_vec()
                };

                (embedding.clone(), doc)
            })
            .collect();
        let fetch_tm = fetch_st.elapsed();

        // Wait for the future to generate an embedding
        let (embedding, embedding_gen_tm) = jh.await.expect("something broke");

        // Sort by cosine similarity
        let sort_st = Instant::now();
        let scores = se
            .lock()
            .await
            .sort_by_similarity(
                embedding.unwrap(),
                docs_with_embeddings.iter().map(|x| x.0.clone()),
            )
            .unwrap();
        let sort_tm = sort_st.elapsed();

        // Create a snippet generator
        let mut snippet_gen_st = Instant::now();
        let snippet_gen =
            SnippetGenerator::create(&searcher, &query, schema.get_field("body").unwrap()).unwrap();
        snippet_gen_tm += snippet_gen_st.elapsed();

        // Get fields we need for the top 10 results and generate a snippet relevant to the search
        // query for each
        for &(i, _score) in scores.iter().take(10) {
            let doc = docs_with_embeddings.get(i).unwrap().1.clone();

            let url = doc
                .get_first(schema.get_field("url").unwrap())
                .unwrap()
                .as_str()
                .unwrap()
                .to_string();
            let title = doc
                .get_first(schema.get_field("title").unwrap())
                .unwrap()
                .as_str()
                .unwrap()
                .to_string();

            // Generate snippet for the document
            snippet_gen_st = Instant::now();
            let snippet = snippet_gen.snippet_from_doc(&doc).to_html();
            snippet_gen_tm += snippet_gen_st.elapsed();

            results.push(Res {
                url,
                title,
                snippet,
            });
        }

        let total_tm = total_st.elapsed();

        Html(templates.render("index.html", &Context::from_serialize(SearchRes {
            query: q,
            results,
            time: format!(
                "{total_tm:?} = parse({parse_tm:?}) + search({search_tm:?}) + fetch({fetch_tm:?}) + embedding({embedding_gen_tm:?}) + sort({sort_tm:?})",
            ),
        }).unwrap()).unwrap()).into_response()
    } else {
        Html(
            st.templates
                .render("index.html", &Context::default())
                .unwrap(),
        )
        .into_response()
    }
}

#[derive(Clone)]
struct AppState {
    reader: IndexReader,
    parser: QueryParser,
    schema: Schema,
    se: Arc<Mutex<SentEmbed>>,
    templates: Tera,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();

    let tera = Tera::new("views/*.html").unwrap();

    let mut se = SentEmbed::new()?;

    let index = SearchIndex::new().await.unwrap();

    Crawler::new(
        "https://docs.python.org/3.13/",
        |url| {
            let path = url.path();
            path.starts_with("/3.13") ||
                path.starts_with("/3.12") ||
                path.starts_with("/3.8") ||
                path.starts_with("/2.7")
        },
        &mut se,
        &index,
    )
    .await
    .unwrap();

    Crawler::new("https://docs.ruby-lang.org/", |url| {
        let path = url.path();
        (path.starts_with("/en/3.3") ||
        path.starts_with("/en/3.4") ||
        path.starts_with("/en/master")) && !(path.ends_with("/index.html") || path.ends_with("/"))
    }, &mut se, &index).await.unwrap();

    Crawler::new("https://doc.rust-lang.org/stable/std/index.html", |url| {
        let path = url.path();
        path.starts_with("/stable") && !path.ends_with("/index.html") && !path.ends_with("/all.html")
    }, &mut se, &index).await.unwrap();

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
