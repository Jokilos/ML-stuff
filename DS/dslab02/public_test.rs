#[cfg(test)]
mod tests {
    use crate::solution::Threadpool;
    use crossbeam_channel::unbounded;
    use ntest::timeout;
    use std::sync::Arc;

    #[test]
    #[timeout(200)]
    fn smoke_test() {
        let (tx, rx) = unbounded();
        let pool = Threadpool::new(1);

        pool.submit(Box::new(move || {
            tx.send(14).unwrap();
        }));

        assert_eq!(14, rx.recv().unwrap());
    }

    #[test]
    #[timeout(200)]
    fn threadpool_is_sync() {
        let send_only_when_threadpool_is_sync = Arc::new(Threadpool::new(1));
        let (tx, rx) = unbounded();

        let _handle = std::thread::spawn(move || {
            tx.send(send_only_when_threadpool_is_sync).unwrap();
        });

        rx.recv().unwrap();
    }

    #[test]
    #[timeout(500)]
    fn multi_thread_test() {
        let (tx, rx) = unbounded();
        let pool = Threadpool::new(4);
    
        for i in 0..10 {
            let tx = tx.clone();
            pool.submit(Box::new(move || {
                tx.send(i * 2).unwrap();
            }));
        }
    
        let mut results: Vec<i32> = (0..10).map(|_| rx.recv().unwrap()).collect();
        results.sort();
        assert_eq!(results, vec![0, 2, 4, 6, 8, 10, 12, 14, 16, 18]);
    }
    
    #[test]
    #[timeout(500)]
    fn task_queueing_test() {
        let (tx, rx) = unbounded();
        let pool = Threadpool::new(2);
    
        for i in 0..5 {
            let tx = tx.clone();
            pool.submit(Box::new(move || {
                tx.send(i).unwrap();
            }));
        }
    
        let mut results: Vec<i32> = (0..5).map(|_| rx.recv().unwrap()).collect();
        results.sort();
        assert_eq!(results, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    #[timeout(1000)]
    fn high_churn_test() {
        let (tx, rx) = unbounded();
        let pool = Threadpool::new(4);
    
        for i in 0..100 {
            let tx = tx.clone();
            pool.submit(Box::new(move || {
                tx.send(i).unwrap();
            }));
        }
    
        let mut results: Vec<i32> = (0..100).map(|_| rx.recv().unwrap()).collect();
        results.sort();
        assert_eq!(results, (0..100).collect::<Vec<_>>());
    }
    
    #[test]
    #[timeout(500)]
    fn graceful_shutdown_on_drop_test() {
        let (tx, rx) = unbounded();

        {
            // Scope to ensure `Threadpool` is dropped at the end of this block
            let pool = Threadpool::new(2);

            for i in 0..5 {
                let tx = tx.clone();
                pool.submit(Box::new(move || {
                    tx.send(i).unwrap();
                }));
            }

            // `pool` will be dropped here, which should trigger its shutdown.
        }

        // After the pool is dropped, we expect all tasks to have been completed.
        let mut results: Vec<i32> = (0..5).map(|_| rx.recv().unwrap()).collect();
        results.sort();
        assert_eq!(results, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    #[timeout(500)]
    fn drop_during_task_execution() {
        let (tx, rx) = unbounded();
    
        {
            // Scope to ensure `Threadpool` is dropped at the end of this block
            let pool = Threadpool::new(2);
    
            // Clone `tx` for each task, so they each have their own sender
            let tx_long_task = tx.clone();
            let tx_short_task = tx.clone();
    
            // Submit a long-running task
            pool.submit(Box::new(move || {
                std::thread::sleep(std::time::Duration::from_millis(100));
                tx_long_task.send(1).unwrap();
            }));
    
            // Submit a quick task that should complete before the pool is dropped
            pool.submit(Box::new(move || {
                tx_short_task.send(2).unwrap();
            }));
    
            // `pool` will be dropped here, but tasks in progress should finish
        }
    
        // Only tasks that started before drop should have completed
        let mut results: Vec<i32> = Vec::new();
        while let Ok(result) = rx.try_recv() {
            results.push(result);
        }
    
        results.sort();
        assert!(results.contains(&1));
        assert!(results.contains(&2));
        assert_eq!(results.len(), 2);
    }
    
}
