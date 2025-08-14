use std::net::ToSocketAddrs;
use std::sync::Arc;

use fasttext::FastText;
use futures::StreamExt;
use tonic::transport::Server;
use tonic::{Request, Response, Status, Streaming};



#[allow(non_camel_case_types)]
mod proto {
    tonic::include_proto!("fasttext_serving");

    pub(crate) const FILE_DESCRIPTOR_SET: &'static [u8] =
        tonic::include_file_descriptor_set!("fasttext_serving_descriptor");
}

use proto::{
    fasttext_serving_server as server, PredictRequest, PredictResponse, Prediction, SentenceVector,
    SentenceVectorRequest, SentenceVectorResponse,
};

#[derive(Debug, Clone)]
struct FastTextServingService {
    model: Arc<FastText>,
    config: Arc<crate::ServerConfig>,
}

#[tonic::async_trait]
impl server::FasttextServing for FastTextServingService {
    async fn predict(
        &self,
        request: Request<Streaming<PredictRequest>>,
    ) -> Result<Response<PredictResponse>, Status> {
        let stream = request.into_inner();
        futures::pin_mut!(stream);
        let model = self.model.clone();
        let mut predictions = Vec::new();
        let mut processed_count = 0;
        let mut error_count = 0;
        
        while let Some(req) = stream.next().await {
            let req = req?;
            let text = req.text;
            let k = req.k.unwrap_or(1);
            let threshold = req.threshold.unwrap_or(self.config.default_threshold);
            
            match crate::predict_one_safe(&model, &text, k, threshold, self.config.max_text_length) {
                Ok((labels, probs)) => {
            predictions.push(Prediction { labels, probs });
                    processed_count += 1;
                }
                Err(e) => {
                    log::warn!("gRPC prediction failed for text (length: {}): {}", text.len(), e);
                    // 返回错误标记而不是失败整个请求，使用完整标签格式
                    predictions.push(Prediction { 
                        labels: vec!["__label__error".to_string()], 
                        probs: vec![0.0] 
                    });
                    error_count += 1;
                }
            }
        }
        
        if error_count > 0 {
            log::warn!("gRPC batch processing completed with {} errors out of {} texts", error_count, processed_count + error_count);
        } else {
            log::info!("gRPC batch processing completed successfully: {} texts", processed_count);
        }
        
        Ok(Response::new(PredictResponse { predictions }))
    }

    async fn sentence_vector(
        &self,
        request: Request<Streaming<SentenceVectorRequest>>,
    ) -> Result<Response<SentenceVectorResponse>, Status> {
        let stream = request.into_inner();
        futures::pin_mut!(stream);
        let mut vectors = Vec::new();
        let model = self.model.clone();
        let mut processed_count = 0;
        let mut error_count = 0;
        
        while let Some(req) = stream.next().await {
            let req = req?;
            let text = req.text;
            
            match model.get_sentence_vector(&text) {
                Ok(values) => {
            vectors.push(SentenceVector { values });
                    processed_count += 1;
                }
                Err(e) => {
                    log::warn!("gRPC sentence vector failed for text (length: {}): {}", text.len(), e);
                    // 返回零向量而不是失败
                    vectors.push(SentenceVector { values: vec![0.0; self.config.default_vector_dim] });
                    error_count += 1;
                }
            }
        }
        
        if error_count > 0 {
            log::warn!("gRPC sentence vector processing completed with {} errors out of {} texts", error_count, processed_count + error_count);
        } else {
            log::info!("gRPC sentence vector processing completed successfully: {} texts", processed_count);
        }
        
        Ok(Response::new(SentenceVectorResponse { vectors }))
    }
}

pub(crate) fn runserver(model: FastText, address: &str, port: u16, num_threads: usize, config: crate::ServerConfig) {
    let reflection_service = tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(proto::FILE_DESCRIPTOR_SET)
        .build()
        .unwrap();
    let instance = FastTextServingService {
        model: Arc::new(model),
        config: Arc::new(config),
    };
    let service = server::FasttextServingServer::new(instance);
    let addr = (address, port).to_socket_addrs().unwrap().next().unwrap();
    let server = Server::builder()
        .add_service(reflection_service)
        .add_service(service);
    log::info!("Listening on {}:{}", address, port);
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(num_threads)
        .build()
        .unwrap()
        .block_on(async {
            server.serve(addr).await.unwrap();
        });
}

// FIXME: add test case
