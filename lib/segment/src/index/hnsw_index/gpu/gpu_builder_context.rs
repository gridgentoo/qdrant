use std::sync::Arc;

use crate::types::PointOffsetType;

#[repr(C)]
struct GpuBuilderContextParamsBuffer {
    processed_requests_count: u32,
}

pub struct GpuBuilderContext {
    pub device: Arc<gpu::Device>,
    pub builder_params_buffer: Arc<gpu::Buffer>,
    pub requests_buffer: Arc<gpu::Buffer>,
    pub responses_buffer: Arc<gpu::Buffer>,
    pub process_points_buffer: Arc<gpu::Buffer>,
    pub process_points_staging_buffer: Arc<gpu::Buffer>,
    pub descriptor_set_layout: Arc<gpu::DescriptorSetLayout>,
    pub descriptor_set: Arc<gpu::DescriptorSet>,
}

impl GpuBuilderContext {
    pub fn new(
        device: Arc<gpu::Device>,
        m: usize,
        points_count: usize,
        threads_count: usize,
    ) -> Self {
        let builder_params_buffer = Arc::new(gpu::Buffer::new(
            device.clone(),
            gpu::BufferType::Uniform,
            std::mem::size_of::<GpuBuilderContextParamsBuffer>(),
        ));

        let requests_buffer = Arc::new(gpu::Buffer::new(
            device.clone(),
            gpu::BufferType::Storage,
            points_count * std::mem::size_of::<PointOffsetType>(),
        ));

        let response_size = (m + 2) * std::mem::size_of::<PointOffsetType>();
        let thread_response_size = m * response_size + std::mem::size_of::<PointOffsetType>();
        let full_responses_size = threads_count * thread_response_size;
        let responses_buffer = Arc::new(gpu::Buffer::new(
            device.clone(),
            gpu::BufferType::Storage,
            full_responses_size,
        ));

        let process_points_buffer = Arc::new(gpu::Buffer::new(
            device.clone(),
            gpu::BufferType::Storage,
            threads_count * std::mem::size_of::<PointOffsetType>(),
        ));
        let process_points_staging_buffer = Arc::new(gpu::Buffer::new(
            device.clone(),
            gpu::BufferType::CpuToGpu,
            threads_count * std::mem::size_of::<PointOffsetType>() + std::mem::size_of::<u32>(),
        ));

        let descriptor_set_layout = gpu::DescriptorSetLayout::builder()
            .add_uniform_buffer(0)
            .add_storage_buffer(1)
            .add_storage_buffer(2)
            .add_storage_buffer(3)
            .build(device.clone());

        let descriptor_set = gpu::DescriptorSet::builder(descriptor_set_layout.clone())
            .add_uniform_buffer(0, builder_params_buffer.clone())
            .add_storage_buffer(1, requests_buffer.clone())
            .add_storage_buffer(2, responses_buffer.clone())
            .add_storage_buffer(3, process_points_buffer.clone())
            .build();

        Self {
            device,
            builder_params_buffer,
            requests_buffer,
            responses_buffer,
            process_points_buffer,
            process_points_staging_buffer,
            descriptor_set_layout,
            descriptor_set,
        }
    }

    pub fn upload_entries(&self, gpu_context: &mut gpu::Context, entries: &[PointOffsetType]) {
        let staging_buffer = Arc::new(gpu::Buffer::new(
            self.device.clone(),
            gpu::BufferType::CpuToGpu,
            entries.len() * std::mem::size_of::<PointOffsetType>(),
        ));
        staging_buffer.upload_slice(entries, 0);
        gpu_context.copy_gpu_buffer(
            staging_buffer.clone(),
            self.requests_buffer.clone(),
            0,
            0,
            entries.len() * std::mem::size_of::<PointOffsetType>(),
        );
        gpu_context.run();
        gpu_context.wait_finish();
    }

    pub fn upload_process_points(
        &self,
        gpu_context: &mut gpu::Context,
        points: &[PointOffsetType],
    ) {
        let requests_count = points.len() as u32;
        self.process_points_staging_buffer
            .upload(&requests_count, 0);
        self.process_points_staging_buffer
            .upload_slice(points, std::mem::size_of::<u32>());
        gpu_context.copy_gpu_buffer(
            self.process_points_staging_buffer.clone(),
            self.process_points_buffer.clone(),
            std::mem::size_of::<u32>(),
            0,
            points.len() * std::mem::size_of::<PointOffsetType>(),
        );
        gpu_context.copy_gpu_buffer(
            self.process_points_staging_buffer.clone(),
            self.builder_params_buffer.clone(),
            0,
            0,
            std::mem::size_of::<u32>(),
        );
        gpu_context.run();
        gpu_context.wait_finish();
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use super::*;
    use crate::common::rocksdb_wrapper::{open_db, DB_VECTOR_CF};
    use crate::fixtures::index_fixtures::{FakeFilterContext, TestRawScorerProducer};
    use crate::index::hnsw_index::entry_points::EntryPoints;
    use crate::index::hnsw_index::gpu::gpu_links::GpuLinks;
    use crate::index::hnsw_index::gpu::gpu_search_context::GpuSearchContext;
    use crate::index::hnsw_index::gpu::gpu_vector_storage::GpuVectorStorage;
    use crate::index::hnsw_index::graph_layers_builder::GraphLayersBuilder;
    use crate::index::hnsw_index::graph_linear_builder::GraphLinearBuilder;
    use crate::index::hnsw_index::point_scorer::FilteredScorer;
    use crate::index::hnsw_index::search_context;
    use crate::spaces::simple::{CosineMetric, DotProductMetric};
    use crate::types::{Distance, PointOffsetType};
    use crate::vector_storage::simple_vector_storage::open_simple_vector_storage;
    use crate::vector_storage::VectorStorage;

    #[test]
    fn hnsw_build_graph_on_gpu_only() {
        let num_vectors = 10000;
        let dim = 32;
        let m = 8;
        let ef_construct = 16;
        let entry_points_num = 10;
        let candidates_capacity = num_vectors / 4;

        let mut rng = StdRng::seed_from_u64(42);
        let vector_holder =
            TestRawScorerProducer::<DotProductMetric>::new(dim, num_vectors, &mut rng);

        let mut graph_layers_builder = GraphLayersBuilder::new_with_params(
            num_vectors,
            m,
            m,
            ef_construct,
            entry_points_num,
            true,
            true,
        );

        let point_levels = (0..num_vectors)
            .map(|_| graph_layers_builder.get_random_layer(&mut rng))
            .collect::<Vec<_>>();
        let max_level = point_levels.iter().max().cloned().unwrap();

        for idx in 0..(num_vectors as PointOffsetType) {
            let fake_filter_context = FakeFilterContext {};
            let added_vector = vector_holder.vectors.get(idx).to_vec();
            let raw_scorer = vector_holder.get_raw_scorer(added_vector.clone());

            let scorer = FilteredScorer::new(raw_scorer.as_ref(), Some(&fake_filter_context));
            graph_layers_builder.set_levels(idx, point_levels[idx as usize]);
            graph_layers_builder.link_new_point(idx, scorer);
        }

        let mut entry_points = EntryPoints::new(entry_points_num);
        let entries = (0..num_vectors as PointOffsetType)
            .map(|idx| entry_points.new_point(idx, point_levels[idx as usize], |_| true))
            .collect::<Vec<_>>();

        // init vector storage
        let dir = tempfile::Builder::new().prefix("db_dir").tempdir().unwrap();
        let db = open_db(dir.path(), &[DB_VECTOR_CF]).unwrap();
        let storage = open_simple_vector_storage(db, DB_VECTOR_CF, dim, Distance::Dot).unwrap();
        {
            let mut borrowed_storage = storage.borrow_mut();
            for idx in 0..(num_vectors as PointOffsetType) {
                borrowed_storage
                    .insert_vector(idx, vector_holder.vectors.get(idx))
                    .unwrap();
            }
        }

        // init gpu resources
        let debug_messenger = gpu::PanicIfErrorMessenger {};
        let instance =
            Arc::new(gpu::Instance::new("qdrant", Some(&debug_messenger), false).unwrap());
        let device =
            Arc::new(gpu::Device::new(instance.clone(), instance.vk_physical_devices[0]).unwrap());
        let mut gpu_context = gpu::Context::new(device.clone());

        let gpu_vector_storage = GpuVectorStorage::new(device.clone(), &storage.borrow()).unwrap();
        let mut gpu_links = GpuLinks::new(device.clone(), m, ef_construct, num_vectors).unwrap();
        let gpu_search_context = GpuSearchContext::new(
            1,
            num_vectors,
            m,
            ef_construct,
            candidates_capacity,
            device.clone(),
        );
        let gpu_builder_context = GpuBuilderContext::new(device.clone(), m, num_vectors, 1);
        let gpu_entries = entries
            .iter()
            .cloned()
            .map(|entry| entry.map(|e| e.point_id).unwrap_or(PointOffsetType::MAX))
            .collect::<Vec<_>>();
        gpu_builder_context.upload_entries(&mut gpu_context, &gpu_entries);

        // init gpu pilelines
        let update_entry_shader = Arc::new(gpu::Shader::new(
            device.clone(),
            include_bytes!("./shaders/update_entries.spv"),
        ));
        let update_entry_pipeline = gpu::Pipeline::builder()
            .add_descriptor_set_layout(0, gpu_vector_storage.descriptor_set_layout.clone())
            .add_descriptor_set_layout(1, gpu_links.descriptor_set_layout.clone())
            .add_descriptor_set_layout(2, gpu_search_context.descriptor_set_layout.clone())
            .add_descriptor_set_layout(3, gpu_builder_context.descriptor_set_layout.clone())
            .add_shader(update_entry_shader.clone())
            .build(device.clone());

        let link_shader = Arc::new(gpu::Shader::new(
            device.clone(),
            include_bytes!("./shaders/run_requests.spv"),
        ));
        let link_pipeline = gpu::Pipeline::builder()
            .add_descriptor_set_layout(0, gpu_vector_storage.descriptor_set_layout.clone())
            .add_descriptor_set_layout(1, gpu_links.descriptor_set_layout.clone())
            .add_descriptor_set_layout(2, gpu_search_context.descriptor_set_layout.clone())
            .add_descriptor_set_layout(3, gpu_builder_context.descriptor_set_layout.clone())
            .add_shader(link_shader.clone())
            .build(device.clone());

        let apply_links_shader = Arc::new(gpu::Shader::new(
            device.clone(),
            include_bytes!("./shaders/apply_responses.spv"),
        ));
        let apply_links_pipeline = gpu::Pipeline::builder()
            .add_descriptor_set_layout(0, gpu_vector_storage.descriptor_set_layout.clone())
            .add_descriptor_set_layout(1, gpu_links.descriptor_set_layout.clone())
            .add_descriptor_set_layout(2, gpu_search_context.descriptor_set_layout.clone())
            .add_descriptor_set_layout(3, gpu_builder_context.descriptor_set_layout.clone())
            .add_shader(apply_links_shader.clone())
            .build(device.clone());

        // test building each level
        for level in (0..=max_level).rev() {
            gpu_links.clear(&mut gpu_context);
            for idx in 0..num_vectors {
                if let Some(Some(entry)) = entries.get(idx) {
                    let entry_level = point_levels[entry.point_id as usize];
                    let point_level = point_levels[idx];
                    let request_level = std::cmp::min(entry_level, point_level);
                    if level > request_level && entry_level >= point_level {
                        gpu_builder_context
                            .upload_process_points(&mut gpu_context, &[idx as PointOffsetType]);

                        gpu_context.bind_pipeline(
                            update_entry_pipeline.clone(),
                            &[
                                gpu_vector_storage.descriptor_set.clone(),
                                gpu_links.descriptor_set.clone(),
                                gpu_search_context.descriptor_set.clone(),
                                gpu_builder_context.descriptor_set.clone(),
                            ],
                        );
                        gpu_context.dispatch(1, 1, 1);
                        gpu_context.run();
                        gpu_context.wait_finish();
                    } else if request_level >= level {
                        println!("Link point {} with entry {}", idx, entry.point_id);
                        gpu_search_context.clear(&mut gpu_context);
                        gpu_builder_context
                            .upload_process_points(&mut gpu_context, &[idx as PointOffsetType]);

                        gpu_context.bind_pipeline(
                            link_pipeline.clone(),
                            &[
                                gpu_vector_storage.descriptor_set.clone(),
                                gpu_links.descriptor_set.clone(),
                                gpu_search_context.descriptor_set.clone(),
                                gpu_builder_context.descriptor_set.clone(),
                            ],
                        );
                        gpu_context.dispatch(1, 1, 1);
                        gpu_context.run();
                        gpu_context.wait_finish();

                        gpu_context.bind_pipeline(
                            apply_links_pipeline.clone(),
                            &[
                                gpu_vector_storage.descriptor_set.clone(),
                                gpu_links.descriptor_set.clone(),
                                gpu_search_context.descriptor_set.clone(),
                                gpu_builder_context.descriptor_set.clone(),
                            ],
                        );
                        gpu_context.dispatch(1, 1, 1);
                        gpu_context.run();
                        gpu_context.wait_finish();
                    }
                }
            }
            gpu_links.download(&mut gpu_context);

            println!(
                "Layer {} contains {} points",
                level,
                (0..num_vectors)
                    .filter(|i| point_levels[*i] >= level)
                    .count(),
            );
            for i in 0..num_vectors {
                if !gpu_links.get_links(i as PointOffsetType).is_empty() {
                    println!("{}: {:?}", i, gpu_links.get_links(i as PointOffsetType));
                }
            }

            for i in 0..num_vectors {
                if point_levels[i] >= level {
                    let links_cpu = graph_layers_builder.links_layers[i][level].read().clone();
                    let links_gpu = gpu_links.get_links(i as PointOffsetType).clone();

                    println!("{}: {:?} vs {:?}", i, links_gpu, links_cpu);
                    assert_eq!(links_cpu, links_gpu);
                } else {
                    assert!(gpu_links.get_links(i as PointOffsetType).is_empty());
                }
            }
        }
    }

    #[test]
    fn hnsw_build_graph_on_gpu() {
        let num_vectors = 10000;
        let dim = 32;
        let m = 8;
        let ef_construct = 16;
        let entry_points_num = 10;
        let candidates_capacity = num_vectors / 4;

        let mut rng = StdRng::seed_from_u64(42);
        let vector_holder = TestRawScorerProducer::<CosineMetric>::new(dim, num_vectors, &mut rng);

        let added_vector = vector_holder.vectors.get(0).to_vec();
        let raw_scorer = vector_holder.get_raw_scorer(added_vector.clone());
        let mut graph_layers_builder = GraphLinearBuilder::new(
            num_vectors,
            m,
            m,
            ef_construct,
            entry_points_num,
            raw_scorer,
            &mut rng,
        );

        let mut graph_layers_orig = GraphLayersBuilder::new_with_params(
            num_vectors,
            m,
            m,
            ef_construct,
            entry_points_num,
            true,
            true,
        );

        let point_levels = (0..num_vectors)
            .map(|i| graph_layers_builder.get_point_level(i as PointOffsetType))
            .collect::<Vec<_>>();
        let max_level = point_levels.iter().max().cloned().unwrap();

        for idx in 0..(num_vectors as PointOffsetType) {
            let fake_filter_context = FakeFilterContext {};
            let added_vector = vector_holder.vectors.get(idx).to_vec();
            let raw_scorer = vector_holder.get_raw_scorer(added_vector.clone());

            let scorer = FilteredScorer::new(raw_scorer.as_ref(), Some(&fake_filter_context));
            graph_layers_orig.set_levels(idx, point_levels[idx as usize]);
            graph_layers_orig.link_new_point(idx, scorer);
        }

        let mut entry_points = EntryPoints::new(entry_points_num);
        let entries = (0..num_vectors as PointOffsetType)
            .map(|idx| entry_points.new_point(idx, point_levels[idx as usize], |_| true))
            .collect::<Vec<_>>();

        // init vector storage
        let dir = tempfile::Builder::new().prefix("db_dir").tempdir().unwrap();
        let db = open_db(dir.path(), &[DB_VECTOR_CF]).unwrap();
        let storage = open_simple_vector_storage(db, DB_VECTOR_CF, dim, Distance::Dot).unwrap();
        {
            let mut borrowed_storage = storage.borrow_mut();
            for idx in 0..(num_vectors as PointOffsetType) {
                borrowed_storage
                    .insert_vector(idx, vector_holder.vectors.get(idx))
                    .unwrap();
            }
        }

        // init gpu resources
        let debug_messenger = gpu::PanicIfErrorMessenger {};
        let instance =
            Arc::new(gpu::Instance::new("qdrant", Some(&debug_messenger), false).unwrap());
        let device =
            Arc::new(gpu::Device::new(instance.clone(), instance.vk_physical_devices[0]).unwrap());
        let mut gpu_context = gpu::Context::new(device.clone());

        let gpu_vector_storage = GpuVectorStorage::new(device.clone(), &storage.borrow()).unwrap();
        let mut gpu_links = GpuLinks::new(device.clone(), m, ef_construct, num_vectors).unwrap();
        let gpu_search_context = GpuSearchContext::new(
            1,
            num_vectors,
            m,
            ef_construct,
            candidates_capacity,
            device.clone(),
        );
        let gpu_builder_context = GpuBuilderContext::new(device.clone(), m, num_vectors, 1);
        let gpu_entries = entries
            .iter()
            .cloned()
            .map(|entry| entry.map(|e| e.point_id).unwrap_or(PointOffsetType::MAX))
            .collect::<Vec<_>>();
        gpu_builder_context.upload_entries(&mut gpu_context, &gpu_entries);

        // init gpu pilelines
        let update_entry_shader = Arc::new(gpu::Shader::new(
            device.clone(),
            include_bytes!("./shaders/update_entries.spv"),
        ));
        let update_entry_pipeline = gpu::Pipeline::builder()
            .add_descriptor_set_layout(0, gpu_vector_storage.descriptor_set_layout.clone())
            .add_descriptor_set_layout(1, gpu_links.descriptor_set_layout.clone())
            .add_descriptor_set_layout(2, gpu_search_context.descriptor_set_layout.clone())
            .add_descriptor_set_layout(3, gpu_builder_context.descriptor_set_layout.clone())
            .add_shader(update_entry_shader.clone())
            .build(device.clone());

        let link_shader = Arc::new(gpu::Shader::new(
            device.clone(),
            include_bytes!("./shaders/run_requests.spv"),
        ));
        let link_pipeline = gpu::Pipeline::builder()
            .add_descriptor_set_layout(0, gpu_vector_storage.descriptor_set_layout.clone())
            .add_descriptor_set_layout(1, gpu_links.descriptor_set_layout.clone())
            .add_descriptor_set_layout(2, gpu_search_context.descriptor_set_layout.clone())
            .add_descriptor_set_layout(3, gpu_builder_context.descriptor_set_layout.clone())
            .add_shader(link_shader.clone())
            .build(device.clone());

        let apply_links_shader = Arc::new(gpu::Shader::new(
            device.clone(),
            include_bytes!("./shaders/apply_responses.spv"),
        ));
        let apply_links_pipeline = gpu::Pipeline::builder()
            .add_descriptor_set_layout(0, gpu_vector_storage.descriptor_set_layout.clone())
            .add_descriptor_set_layout(1, gpu_links.descriptor_set_layout.clone())
            .add_descriptor_set_layout(2, gpu_search_context.descriptor_set_layout.clone())
            .add_descriptor_set_layout(3, gpu_builder_context.descriptor_set_layout.clone())
            .add_shader(apply_links_shader.clone())
            .build(device.clone());

        // test building each level
        for level in (0..=max_level).rev() {
            gpu_links.clear(&mut gpu_context);

            check_state(
                &mut gpu_context,
                num_vectors,
                level,
                &graph_layers_builder,
                &gpu_search_context,
                &mut gpu_links,
                &gpu_builder_context,
            );

            for idx in 0..num_vectors {
                if let Some(Some(entry)) = entries.get(idx) {
                    let entry_level = point_levels[entry.point_id as usize];
                    let point_level = point_levels[idx];
                    let request_level = std::cmp::min(entry_level, point_level);
                    if level > request_level && entry_level >= point_level {
                        graph_layers_builder.update_entry(level, idx as PointOffsetType);

                        gpu_builder_context
                            .upload_process_points(&mut gpu_context, &[idx as PointOffsetType]);

                        gpu_context.bind_pipeline(
                            update_entry_pipeline.clone(),
                            &[
                                gpu_vector_storage.descriptor_set.clone(),
                                gpu_links.descriptor_set.clone(),
                                gpu_search_context.descriptor_set.clone(),
                                gpu_builder_context.descriptor_set.clone(),
                            ],
                        );
                        gpu_context.dispatch(1, 1, 1);
                        gpu_context.run();
                        gpu_context.wait_finish();

                        check_state(
                            &mut gpu_context,
                            num_vectors,
                            level,
                            &graph_layers_builder,
                            &gpu_search_context,
                            &mut gpu_links,
                            &gpu_builder_context,
                        );
                    } else if request_level >= level {
                        println!("Link point {} with entry {}", idx, entry.point_id);
                        graph_layers_builder.link_point(level, idx as PointOffsetType);

                        gpu_search_context.clear(&mut gpu_context);
                        gpu_builder_context
                            .upload_process_points(&mut gpu_context, &[idx as PointOffsetType]);

                        gpu_context.bind_pipeline(
                            link_pipeline.clone(),
                            &[
                                gpu_vector_storage.descriptor_set.clone(),
                                gpu_links.descriptor_set.clone(),
                                gpu_search_context.descriptor_set.clone(),
                                gpu_builder_context.descriptor_set.clone(),
                            ],
                        );
                        gpu_context.dispatch(1, 1, 1);
                        gpu_context.run();
                        gpu_context.wait_finish();

                        gpu_context.bind_pipeline(
                            apply_links_pipeline.clone(),
                            &[
                                gpu_vector_storage.descriptor_set.clone(),
                                gpu_links.descriptor_set.clone(),
                                gpu_search_context.descriptor_set.clone(),
                                gpu_builder_context.descriptor_set.clone(),
                            ],
                        );
                        gpu_context.dispatch(1, 1, 1);
                        gpu_context.run();
                        gpu_context.wait_finish();

                        check_state(
                            &mut gpu_context,
                            num_vectors,
                            level,
                            &graph_layers_builder,
                            &gpu_search_context,
                            &mut gpu_links,
                            &gpu_builder_context,
                        );
                    }
                }
            }
            gpu_links.download(&mut gpu_context);

            for i in 0..num_vectors {
                if point_levels[i] >= level {
                    let links_orig = graph_layers_orig.links_layers[i][level].read().clone();
                    let links_gpu = graph_layers_builder.get_links(i as PointOffsetType, level);
                    //gpu_links.get_links(i as PointOffsetType).clone();

                    println!("{}: {:?} vs {:?}", i, links_gpu, links_orig);
                    assert_eq!(links_orig, links_gpu);
                } else {
                    assert!(gpu_links.get_links(i as PointOffsetType).is_empty());
                }
            }
        }
    }

    fn check_state(
        context: &mut gpu::Context,
        points_count: usize,
        level: usize,
        builder: &GraphLinearBuilder,
        search_context: &GpuSearchContext,
        links: &mut GpuLinks,
        builder_context: &GpuBuilderContext,
    ) {
        // check requests
        let mut requests = vec![PointOffsetType::MAX; points_count];
        let staging_buffer = Arc::new(gpu::Buffer::new(
            builder_context.device.clone(),
            gpu::BufferType::GpuToCpu,
            points_count * std::mem::size_of::<PointOffsetType>(),
        ));
        context.copy_gpu_buffer(
            builder_context.requests_buffer.clone(),
            staging_buffer.clone(),
            0,
            0,
            points_count * std::mem::size_of::<PointOffsetType>(),
        );
        context.run();
        context.wait_finish();
        staging_buffer.download_slice(&mut requests, 0);

        for i in 0..points_count {
            let cpu_entry = builder.requests[i]
                .clone()
                .map(|e| e.entry.idx)
                .unwrap_or(PointOffsetType::MAX);
            let gpu_entry = requests[i];
            if gpu_entry != cpu_entry {
                println!(
                    "Request {} at level {} is wrong: gpu {} vs cpu {}",
                    i, level, gpu_entry, cpu_entry
                );
            }
            assert_eq!(cpu_entry, gpu_entry);
        }

        links.download(context);
        for i in 0..points_count {
            let cpu_links = builder.get_links(i as PointOffsetType, level);
            let gpu_links = links.get_links(i as PointOffsetType);
            if gpu_links != cpu_links {
                println!(
                    "Links {} at level {} is wrong: gpu {:?} vs cpu {:?}",
                    i, level, gpu_links, cpu_links
                );
            }
            assert_eq!(cpu_links, gpu_links);
        }
    }
}