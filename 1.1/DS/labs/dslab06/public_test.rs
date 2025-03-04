#[cfg(test)]
mod tests {
    use crate::solution::build_stable_storage;
    use ntest::timeout;
    use tempfile::tempdir;

    #[tokio::test]
    #[timeout(500)]
    async fn storage_retrieves_inserted_key() {
        // Given:
        let root_storage_dir = tempdir().unwrap();
        let mut storage = build_stable_storage(root_storage_dir.path().to_path_buf()).await;

        // When:
        let before_insertion = storage.get("key").await;
        // Then:
        assert_eq!(before_insertion, None);

        // When:
        storage
            .put("key", vec![1_u8, 2, 3].as_slice())
            .await
            .unwrap();
        // Then:
        assert_eq!(storage.get("key").await.unwrap(), vec![1_u8, 2, 3]);
    }

    #[tokio::test]
    #[timeout(500)]
    async fn remove_removes_key() {
        // Given:
        let root_storage_dir = tempdir().unwrap();
        let mut storage = build_stable_storage(root_storage_dir.path().to_path_buf()).await;
        storage
            .put("key", vec![1_u8, 2, 3].as_slice())
            .await
            .unwrap();

        // When:
        storage.remove("key").await;

        // Then:
        assert_eq!(storage.get("key").await, None);
    }
}
