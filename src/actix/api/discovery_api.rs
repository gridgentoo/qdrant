use actix_web::{post, web, Responder};
use actix_web_validator::{Json, Path, Query};
use collection::operations::shard_selector_internal::ShardSelectorInternal;
use collection::operations::types::{DiscoverRequest, DiscoverRequestBatch};
use storage::content_manager::toc::TableOfContent;
use tokio::time::Instant;

use crate::actix::api::read_params::ReadParams;
use crate::actix::api::CollectionPath;
use crate::actix::helpers::process_response;
use crate::common::points::do_discover_batch_points;

#[post("/collections/{name}/points/discover")]
async fn discover_points(
    toc: web::Data<TableOfContent>,
    collection: Path<CollectionPath>,
    request: Json<DiscoverRequest>,
    params: Query<ReadParams>,
) -> impl Responder {
    let timing = Instant::now();

    let DiscoverRequest {
        discover_request,
        shard_key,
    } = request.into_inner();

    let shard_selection = match shard_key {
        None => ShardSelectorInternal::All,
        Some(shard_keys) => shard_keys.into(),
    };

    let response = toc
        .discover(
            &collection.name,
            discover_request,
            params.consistency,
            shard_selection,
            params.timeout(),
        )
        .await;

    process_response(response, timing)
}

#[post("/collections/{name}/points/discover/batch")]
async fn discover_batch_points(
    toc: web::Data<TableOfContent>,
    collection: Path<CollectionPath>,
    request: Json<DiscoverRequestBatch>,
    params: Query<ReadParams>,
) -> impl Responder {
    let timing = Instant::now();

    let response = do_discover_batch_points(
        toc.get_ref(),
        &collection.name,
        request.into_inner(),
        params.consistency,
        params.timeout(),
    )
    .await;

    process_response(response, timing)
}

pub fn config_discovery_api(cfg: &mut web::ServiceConfig) {
    cfg.service(discover_points);
    cfg.service(discover_batch_points);
}
