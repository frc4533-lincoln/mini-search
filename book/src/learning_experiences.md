
# Sentence embedding

I haven't really used any of the Rust ML libraries before, so I had a lot of fun with this part.
I had some trouble getting started, because there wasn't really any documentation on how to do sentence embedding in Rust.
I was able to piece together some of the Rust equivalents of the Python code to get `tokenizers` working.
For the `transformers` part, I started with `candle`'s BERT example, then did a little precision guesswork.

I was able to optimize their code down to be pretty efficient for my use case.
With more time, it could probably be improved, but I don't think it'd be a big enough difference to matter.

# Indexing

I've used `tantivy` on many projects in the past, so I mostly just had to re-learn a little.
I found that snippet generation is the biggest time sink, but I don't think that's `tantivy`'s fault.

I had been shoving HTML tags into the body field.
After implementing some logic to collect only actual text, performance improved greatly for most of the domains.
`docs.python.org` still slows it down quite a bit, because they have very large documents.

# Crawling

I had a lot of trouble with `spider`.
I wanted to crawl only pages under a specific base path, but eventually came to the realization that they simply don't have support for this.

I tried a few different things:
 - Just adding the base path to the URL:
   - Preferred the base path(?), but happily did everything else too.
 - `with_budget`:
   - The documentation says to enable the `budget` feature, which doesn't exist.
   - I tried using it without the `budget` feature. It did something, but I couldn't get it to work properly.
 - `with_blacklist_url`:
   - It doesn't accept wildcards and ignores subpaths.

In the end, I decided to just use `with_limit` to limit the total crawl to 10,000 pages and gave up on keeping it from crawling pages I don't want.
On my previous project Searched, I wrote a custom crawler from scratch before it morphed into a meta-search engine.
I had tried to use `spider`, but it frustrated me back then too. :')

