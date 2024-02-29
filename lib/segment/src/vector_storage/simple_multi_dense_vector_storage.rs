use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use atomic_refcell::AtomicRefCell;
use bitvec::prelude::BitVec;
use common::types::PointOffsetType;
use parking_lot::RwLock;
use rocksdb::DB;

use crate::common::operation_error::{check_process_stopped, OperationError, OperationResult};
use crate::common::rocksdb_wrapper::DatabaseColumnWrapper;
use crate::data_types::vectors::MultiDenseVector;
use crate::types::Distance;
use crate::vector_storage::bitvec::bitvec_set_deleted;
use crate::vector_storage::common::StoredRecord;
use crate::vector_storage::VectorStorageEnum;

type StoredMultiDenseVector = StoredRecord<MultiDenseVector>;

/// In-memory vector storage with on-update persistence using `store`
#[allow(unused)]
pub struct SimpleMultiDenseVectorStorage {
    dim: usize,
    distance: Distance,
    /// Keep vectors in memory
    vectors: Vec<MultiDenseVector>,
    db_wrapper: DatabaseColumnWrapper,
    update_buffer: StoredMultiDenseVector,
    /// BitVec for deleted flags. Grows dynamically upto last set flag.
    deleted: BitVec,
    /// Current number of deleted vectors.
    deleted_count: usize,
}

#[allow(unused)]
pub fn open_simple_multi_dense_vector_storage(
    database: Arc<RwLock<DB>>,
    database_column_name: &str,
    dim: usize,
    distance: Distance,
    stopped: &AtomicBool,
) -> OperationResult<Arc<AtomicRefCell<VectorStorageEnum>>> {
    let mut vectors = vec![];
    let (mut deleted, mut deleted_count) = (BitVec::new(), 0);
    let db_wrapper = DatabaseColumnWrapper::new(database, database_column_name);

    let mut total_vector_count = 0;
    let mut total_sparse_size = 0;
    db_wrapper.lock_db().iter()?;
    for (key, value) in db_wrapper.lock_db().iter()? {
        let point_id: PointOffsetType = bincode::deserialize(&key)
            .map_err(|_| OperationError::service_error("cannot deserialize point id from db"))?;
        let stored_record: StoredMultiDenseVector = bincode::deserialize(&value)
            .map_err(|_| OperationError::service_error("cannot deserialize record from db"))?;

        // Propagate deleted flag
        if stored_record.deleted {
            bitvec_set_deleted(&mut deleted, point_id, true);
            deleted_count += 1;
        }
        vectors.insert(point_id as usize, stored_record.vector);

        check_process_stopped(stopped)?;
    }

    Ok(Arc::new(AtomicRefCell::new(
        VectorStorageEnum::MultiDenseSimple(SimpleMultiDenseVectorStorage {
            dim,
            distance,
            vectors,
            db_wrapper,
            update_buffer: StoredMultiDenseVector {
                deleted: false,
                vector: MultiDenseVector::default(),
            },
            deleted,
            deleted_count,
        }),
    )))
}

impl SimpleMultiDenseVectorStorage {
    /// Set deleted flag for given key. Returns previous deleted state.
    #[inline]
    #[allow(unused)]
    fn set_deleted(&mut self, key: PointOffsetType, deleted: bool) -> bool {
        if key as usize >= self.vectors.len() {
            return false;
        }
        let was_deleted = bitvec_set_deleted(&mut self.deleted, key, deleted);
        if was_deleted != deleted {
            if !was_deleted {
                self.deleted_count += 1;
            } else {
                self.deleted_count = self.deleted_count.saturating_sub(1);
            }
        }
        was_deleted
    }

    #[allow(unused)]
    fn update_stored(
        &mut self,
        key: PointOffsetType,
        deleted: bool,
        vector: Option<&MultiDenseVector>,
    ) -> OperationResult<()> {
        // Write vector state to buffer record
        let record = &mut self.update_buffer;
        record.deleted = deleted;
        if let Some(vector) = vector {
            record.vector = vector.clone();
        }

        // Store updated record
        self.db_wrapper.put(
            bincode::serialize(&key).unwrap(),
            bincode::serialize(&record).unwrap(),
        )?;

        Ok(())
    }
}

// TODO integrate MultiDenseVector to Vectors enum to enable this implementation
// impl VectorStorage for SimpleMultiDenseVectorStorage
