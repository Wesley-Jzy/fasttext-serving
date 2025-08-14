use clap::{Arg, ArgAction, Command};
use fasttext::FastText;
use std::env;
use std::path::Path;

#[cfg(feature = "grpc")]
mod grpc;
#[cfg(feature = "http")]
mod http;

#[cfg(all(unix, not(target_env = "musl"), not(target_arch = "aarch64")))]
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

#[cfg(windows)]
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Debug)]
pub enum PredictError {
    ModelError(String),
    InputError(String),
}

impl std::fmt::Display for PredictError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            PredictError::ModelError(msg) => write!(f, "Model error: {}", msg),
            PredictError::InputError(msg) => write!(f, "Input error: {}", msg),
        }
    }
}

impl std::error::Error for PredictError {}

#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub max_text_length: usize,
    pub default_threshold: f32,
    pub default_vector_dim: usize,
    pub max_request_size_mb: u32,
}

#[inline]
pub fn predict_one_safe(
    model: &FastText,
    text: &str,
    k: u32,
    threshold: f32,
    max_text_length: usize,
) -> Result<(Vec<String>, Vec<f32>), PredictError> {
    // Validate input
    if text.is_empty() {
        return Err(PredictError::InputError("Empty text input".to_string()));
    }
    
    if text.len() > max_text_length {
        return Err(PredictError::InputError(format!("Text too long: {} bytes", text.len())));
    }
    
    // Ensure k >= 1
    let k = if k > 0 { k } else { 1 };
    
    // NOTE: text needs to end in a newline
    // to exactly mimic the behavior of the cli
    let preds = if text.ends_with('\n') {
        model
            .predict(text, k as i32, threshold)
            .map_err(|e| PredictError::ModelError(format!("Prediction failed: {}", e)))?
    } else {
        let mut text = text.to_string();
        text.push('\n');
        model
            .predict(&text, k as i32, threshold)
            .map_err(|e| PredictError::ModelError(format!("Prediction failed: {}", e)))?
    };
    
    let mut labels = Vec::with_capacity(preds.len());
    let mut probs = Vec::with_capacity(preds.len());
    for pred in &preds {
        labels.push(pred.label.trim_start_matches("__label__").to_string());
        probs.push(pred.prob);
    }
    
    Ok((labels, probs))
}

// 保留原始的predict_one函数以保持向后兼容，但内部使用安全版本
#[inline]
pub fn predict_one(
    model: &FastText,
    text: &str,
    k: u32,
    threshold: f32,
    max_text_length: usize,
) -> (Vec<String>, Vec<f32>) {
    match predict_one_safe(model, text, k, threshold, max_text_length) {
        Ok(result) => result,
        Err(e) => {
            log::error!("Prediction failed, returning default result: {}", e);
            (vec!["error".to_string()], vec![0.0])
        }
    }
}

fn main() {
    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", "fasttext_serving=info");
    }
    pretty_env_logger::init();

    let num_threads = num_cpus::get().to_string();
    let matches = Command::new("fasttext-serving")
        .version(env!("CARGO_PKG_VERSION"))
        .about("fastText model serving service")
        .author("Messense Lv <messense@icloud.com>")
        .arg(
            Arg::new("model")
                .required(true)
                .short('m')
                .long("model")
                .value_name("model")
                .num_args(1)
                .help("Model path"),
        )
        .arg(
            Arg::new("address")
                .short('a')
                .long("address")
                .default_value("127.0.0.1")
                .num_args(1)
                .help("Listen address"),
        )
        .arg(
            Arg::new("port")
                .short('p')
                .long("port")
                .default_value("8000")
                .num_args(1)
                .help("Listen port"),
        )
        .arg(
            Arg::new("workers")
                .short('w')
                .long("workers")
                .alias("concurrency")
                .alias("threads")
                .default_value(&num_threads)
                .num_args(1)
                .help("Worker thread count, defaults to CPU count"),
        )
        .arg(
            Arg::new("grpc")
                .long("grpc")
                .action(ArgAction::SetTrue)
                .help("Serving gRPC API instead of HTTP API"),
        )
        .arg(
            Arg::new("max-request-size")
                .long("max-request-size")
                .default_value("500")
                .num_args(1)
                .help("Maximum request size in MB (default: 500MB)"),
        )
        .arg(
            Arg::new("max-text-length")
                .long("max-text-length")
                .default_value("5000000")
                .num_args(1)
                .help("Maximum text length in bytes (default: 5MB)"),
        )
        .arg(
            Arg::new("default-threshold")
                .long("default-threshold")
                .default_value("0.0")
                .num_args(1)
                .help("Default prediction threshold (default: 0.0)"),
        )
        .arg(
            Arg::new("default-vector-dim")
                .long("default-vector-dim")
                .default_value("100")
                .num_args(1)
                .help("Default sentence vector dimension for errors (default: 100)"),
        )
        .get_matches();
        
    let model_path = matches.get_one::<String>("model").unwrap();
    if !Path::new(model_path).exists() {
        log::error!("Model file does not exist: {}", model_path);
        std::process::exit(1);
    }
    
    let address = matches
        .get_one::<String>("address")
        .expect("missing address");
    let port = matches.get_one::<String>("port").expect("missing port");
    let workers = matches
        .get_one::<String>("workers")
        .expect("missing workers");
    let max_request_size = matches
        .get_one::<String>("max-request-size")
        .expect("missing max-request-size");
    let max_text_length = matches
        .get_one::<String>("max-text-length")
        .expect("missing max-text-length");
    let default_threshold = matches
        .get_one::<String>("default-threshold")
        .expect("missing default-threshold");
    let default_vector_dim = matches
        .get_one::<String>("default-vector-dim")
        .expect("missing default-vector-dim");
        
    log::info!("Loading FastText model from: {}", model_path);
    let mut model = FastText::new();
    match model.load_model(model_path) {
        Ok(_) => {
            log::info!("FastText model loaded successfully");
        }
        Err(e) => {
            log::error!("Failed to load FastText model: {}", e);
            std::process::exit(1);
        }
    }
    
    let port: u16 = port.parse().unwrap_or_else(|_| {
        log::error!("Invalid port number: {}", port);
        std::process::exit(1);
    });
    
    let workers: usize = workers.parse().unwrap_or_else(|_| {
        log::error!("Invalid worker count: {}", workers);
        std::process::exit(1);
    });
    
    let _max_request_size_mb: u32 = max_request_size.parse().unwrap_or_else(|_| {
        log::error!("Invalid max request size: {}", max_request_size);
        std::process::exit(1);
    });
    let _max_text_length_bytes: usize = max_text_length.parse().unwrap_or_else(|_| {
        log::error!("Invalid max text length: {}", max_text_length);
        std::process::exit(1);
    });
    let _default_threshold: f32 = default_threshold.parse().unwrap_or_else(|_| {
        log::error!("Invalid default threshold: {}", default_threshold);
        std::process::exit(1);
    });
    let _default_vector_dim: usize = default_vector_dim.parse().unwrap_or_else(|_| {
        log::error!("Invalid default vector dim: {}", default_vector_dim);
        std::process::exit(1);
    });
    
    log::info!("Starting server with {} workers on {}:{}", workers, address, port);
    log::info!("Maximum request size: {}MB", _max_request_size_mb);
    log::info!("Maximum text length: {} bytes", _max_text_length_bytes);
    log::info!("Default threshold: {}", _default_threshold);
    log::info!("Default vector dimension: {}", _default_vector_dim);
    
    let config = ServerConfig {
        max_text_length: _max_text_length_bytes,
        default_threshold: _default_threshold,
        default_vector_dim: _default_vector_dim,
        max_request_size_mb: _max_request_size_mb,
    };

    if matches.get_flag("grpc") {
        #[cfg(feature = "grpc")]
        crate::grpc::runserver(model, address, port, workers, config);
        #[cfg(not(feature = "grpc"))]
        {
            log::error!("gRPC support is not enabled!");
            std::process::exit(1);
        }
    } else {
        #[cfg(feature = "http")]
        crate::http::runserver(model, address, port, workers, config);
        #[cfg(not(feature = "http"))]
        {
            log::error!("HTTP support is not enabled!");
            std::process::exit(1);
        }
    }
}
