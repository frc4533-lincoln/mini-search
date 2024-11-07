
# From source (Linux)

 0. First, [install Rust](https://www.rust-lang.org/tools/install) if you haven't already.
 1. Clone the Git repo (and `cd` into it)
    ```shell
    git clone https://github.com/frc4533-lincoln/mini-search && cd mini-search
    ```
 2. Download the required resources for sentence embedding
    ```shell
    chmod +x ./scripts/get_model.sh && ./scripts/get_model.sh
    ```
 3. Build the program (may take a while)
    ```shell
    cargo b -r
    ```
 4. Run it (it will take a while to index)
    ```shell
    RUST_LOG=info cargo r -r
    ```

