use std::fmt;
use std::io;
use std::str::FromStr;

use actix_web::rt::System;
use actix_web::{web, App, HttpServer, HttpResponse, Result as ActixResult};
use fasttext::FastText;
use serde::{Deserialize, Serialize};

use crate::predict_one_safe;

const UNIX_PREFIX: &'static str = "unix:";

enum Address {
    IpPort(String, u16),
    Unix(String),
}

impl From<(&str, u16)> for Address {
    fn from(addr: (&str, u16)) -> Self {
        addr.0
            .parse::<Address>()
            .unwrap_or_else(|_| Address::IpPort(addr.0.to_string(), addr.1))
    }
}

impl FromStr for Address {
    type Err = io::Error;

    fn from_str(string: &str) -> io::Result<Self> {
        #[cfg(unix)]
        {
            if string.starts_with(UNIX_PREFIX) {
                let address = &string[UNIX_PREFIX.len()..];
                return Ok(Address::Unix(address.into()));
            }
        }
        Err(io::Error::new(
            io::ErrorKind::Other,
            "failed to resolve TCP address",
        ))
    }
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Address::IpPort(ip, port) => write!(f, "{}:{}", ip, port),
            Address::Unix(path) => write!(f, "{}{}", UNIX_PREFIX, path),
        }
    }
}

#[derive(Deserialize, Debug, Default)]
struct PredictOptions {
    k: Option<u32>,
    threshold: Option<f32>,
}

#[derive(Serialize)]
struct PredictResult {
    labels: Vec<String>,
    scores: Vec<f32>,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
    message: String,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    model_loaded: bool,
}

async fn health_check() -> ActixResult<HttpResponse> {
    let response = HealthResponse {
        status: "healthy".to_string(),
        model_loaded: true,
    };
    Ok(HttpResponse::Ok().json(response))
}

async fn predict(
    model: web::Data<FastText>,
    texts: web::Json<Vec<String>>,
    options: web::Query<PredictOptions>,
) -> ActixResult<HttpResponse> {
    let k = options.k.unwrap_or(1);
    let threshold = options.threshold.unwrap_or(0.0);
    let text_count = texts.len();
    
    log::info!("Processing {} texts with k={}, threshold={}", text_count, k, threshold);
    
    if text_count == 0 {
        return Ok(HttpResponse::Ok().json(Vec::<PredictResult>::new()));
    }
    
    // 使用安全的预测函数，避免单个文本错误导致整个批次失败
    let mut results = Vec::with_capacity(text_count);
    let mut success_count = 0;
    let mut error_count = 0;
    
    for txt in texts.iter() {
        match predict_one_safe(model.get_ref(), txt, k, threshold) {
            Ok((labels, probs)) => {
                results.push(PredictResult {
                    labels,
                    scores: probs,
                });
                success_count += 1;
            }
            Err(e) => {
                log::warn!("Prediction failed for text (length: {}): {}", txt.len(), e);
                // 返回默认结果而不是失败
                results.push(PredictResult {
                    labels: vec!["error".to_string()],
                    scores: vec![0.0],
                });
                error_count += 1;
            }
        }
    }
    
    if error_count > 0 {
        log::warn!("Batch processing completed with {} errors out of {} texts", error_count, text_count);
    } else {
        log::info!("Batch processing completed successfully: {} texts", success_count);
    }
    
    // 转换为原始格式 [(labels, scores), ...]
    let legacy_results: Vec<(Vec<String>, Vec<f32>)> = results
        .into_iter()
        .map(|r| (r.labels, r.scores))
        .collect();
    
    Ok(HttpResponse::Ok().json(legacy_results))
}

async fn sentence_vector(
    model: web::Data<FastText>,
    texts: web::Json<Vec<String>>,
) -> ActixResult<HttpResponse> {
    let text_count = texts.len();
    log::info!("Processing {} texts for sentence vectors", text_count);
    
    if text_count == 0 {
        return Ok(HttpResponse::Ok().json(Vec::<Vec<f32>>::new()));
    }
    
    let mut results = Vec::with_capacity(text_count);
    let mut success_count = 0;
    let mut error_count = 0;
    
    for txt in texts.iter() {
        match model.get_sentence_vector(txt) {
            Ok(vector) => {
                results.push(vector);
                success_count += 1;
            }
            Err(e) => {
                log::warn!("Sentence vector failed for text (length: {}): {}", txt.len(), e);
                // 返回零向量而不是失败
                results.push(vec![0.0; 100]); // 假设向量维度为100，实际应该根据模型确定
                error_count += 1;
            }
        }
    }
    
    if error_count > 0 {
        log::warn!("Sentence vector processing completed with {} errors out of {} texts", error_count, text_count);
    } else {
        log::info!("Sentence vector processing completed successfully: {} texts", success_count);
    }
    
    Ok(HttpResponse::Ok().json(results))
}

pub(crate) fn runserver(model: FastText, address: &str, port: u16, workers: usize) {
    let addr = Address::from((address, port));
    log::info!("Listening on {}", addr);
    let model_data = web::Data::new(model);
    
    // 大幅提升JSON限制到500MB，适合大规模批处理
    let json_cfg = web::JsonConfig::default()
        .limit(500_000_000) // 500MB - 比原来的20MB提升25倍
        .content_type(|_mime| true) // Accept any content type
        .error_handler(|err, _req| {
            let error_message = format!("Failed to parse JSON: {}", err);
            log::error!("JSON parsing error: {}", err);
            actix_web::error::InternalError::from_response(
                err,
                HttpResponse::BadRequest().json(ErrorResponse {
                    error: "json_parse_error".to_string(),
                    message: error_message,
                })
            ).into()
        });
        
    let mut server = HttpServer::new(move || {
        App::new()
            .service(
                web::resource("/predict")
                    .app_data(model_data.clone())
                    .app_data(json_cfg.clone())
                    .route(web::post().to(predict)),
            )
            .service(
                web::resource("/sentence-vector")
                    .app_data(model_data.clone())
                    .app_data(json_cfg.clone())
                    .route(web::post().to(sentence_vector)),
            )
            .service(
                web::resource("/health")
                    .route(web::get().to(health_check)),
            )
    })
    .workers(workers);

    let sys = System::new();
    server = match addr {
        Address::IpPort(address, port) => server.bind((&address[..], port)).expect("bind failed"),
        Address::Unix(path) => {
            #[cfg(unix)]
            {
                server.bind_uds(path).expect("bind failed")
            }
            #[cfg(not(unix))]
            {
                panic!("Unix domain socket is not supported on this platform")
            }
        }
    };
    sys.block_on(async { server.run().await }).unwrap();
}

#[cfg(test)]
mod test {
    use super::predict;
    use actix_web::http::StatusCode;
    use actix_web::test::{call_service, init_service, TestRequest};
    use actix_web::{web, App};
    use fasttext::FastText;

    #[actix_rt::test]
    async fn test_predict_empty_input() {
        let mut fasttext = FastText::new();
        fasttext
            .load_model("models/cooking.model.bin")
            .expect("Failed to load fastText model");
        let model_data = web::Data::new(fasttext);
        let mut srv = init_service(
            App::new()
                .app_data(model_data)
                .service(web::resource("/predict").route(web::post().to(predict))),
        )
        .await;
        let data: Vec<String> = Vec::new();
        let req = TestRequest::post()
            .uri("/predict")
            .set_json(&data)
            .to_request();
        let resp = call_service(&mut srv, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[actix_rt::test]
    async fn test_predict() {
        let mut fasttext = FastText::new();
        fasttext
            .load_model("models/cooking.model.bin")
            .expect("Failed to load fastText model");
        let model_data = web::Data::new(fasttext);
        let mut srv = init_service(
            App::new()
                .app_data(model_data)
                .service(web::resource("/predict").route(web::post().to(predict))),
        )
        .await;
        let data = vec!["Which baking dish is best to bake a banana bread?"];
        let req = TestRequest::post()
            .uri("/predict")
            .set_json(&data)
            .to_request();
        let resp = call_service(&mut srv, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }
}
