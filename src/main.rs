use std::env;
use std::path::PathBuf;
use std::sync::Arc;

use tanpopo::index::IndexManager;
use tanpopo::server::build_router;
use tanpopo::tokenizer::load_dictionary_from_dir;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mcp_stdio = env::args().any(|a| a == "--mcp") || env::var("TANPOPO_MCP_STDIO").is_ok_and(|v| v == "1");
    // stdout is reserved for JSON-RPC in stdio mode
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let bind_addr = env::var("TANPOPO_BIND").unwrap_or_else(|_| "127.0.0.1:9200".to_string());
    let data_dir = env::var("TANPOPO_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("./data"));
    let dict_dir = env::var("TANPOPO_SUDACHI_DICT_DIR").ok().map(PathBuf::from);
    let dict_path = env::var("TANPOPO_DICT_PATH").ok().map(PathBuf::from);
    let writer_heap_bytes: usize = env::var("TANPOPO_WRITER_HEAP_BYTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(15_000_000);

    let dict = load_dictionary_from_dir(dict_dir.as_deref(), dict_path.as_deref())?;
    let mgr = Arc::new(IndexManager::new(data_dir.clone(), dict, writer_heap_bytes)?);
    if mcp_stdio {
        return Ok(tokio::task::spawn_blocking(move || tanpopo::mcp::run_stdio_loop(mgr)).await??);
    }
    let app = build_router(mgr);

    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    tracing::info!(
        "Tanpopo listening on http://{} (data_dir: {}, writer_heap: {} MB)",
        bind_addr,
        data_dir.display(),
        writer_heap_bytes / 1_000_000
    );

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("Shutting down Tanpopo...");
}
