use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Condvar, Mutex};
use std::rc::Rc;
use std::thread::{spawn, JoinHandle, Thread};

type Task = Box<dyn FnOnce() + Send>;

// You can define new types (e.g., structs) if you need.
// However, they shall not be public (i.e., do not use the `pub` keyword).

/// The thread pool.
pub struct Threadpool {
    worker_count: usize,
    join_handles: Vec<JoinHandle<()>>,
    available_tasks: <Vec<Task>>,
    is_running: bool,

    // Add here any fields you need.
    // We suggest storing handles of the worker threads, submitted tasks,
    // and information whether the pool is running or is shutting down.
}

// struct Worker{
//     available_tasks: Arc<(Mutex<Vec<Task>>, Condvar)>,
//     is_running: AtomicBool,
// }
type TaskArc = Arc<(Mutex<Vec<Task>>, Condvar)>;

fn is_sync<T: Sync>() {}

impl Threadpool {
    /// Create new thread pool with `workers_count` workers.
    pub fn new(workers_count: usize) -> Self {
        let available_tasks: TaskArc =
            Arc::new((Mutex::new(Vec::new()), Condvar::new()));

        let mut tp = Threadpool{
            worker_count: workers_count,
            join_handles: Vec::new(),
            available_tasks: available_tasks,
            is_running: Arc::new(AtomicBool::new(true)),
        };

        is_sync::<Threadpool>();
        
        for _ in 0..workers_count {
            // let at_cloned = 
            //     tp.available_tasks.clone();
            
            // let ir_cloned  = tp.is_running.clone();
            // let mut &tp = tp;            

            let thread = spawn(move || {
                &tp.worker_loop();
            } );

            tp.join_handles.push(thread);

            //println!("{:?}", tp);

            unimplemented!("Create the workers.");
        }

        unimplemented!("Return the new Threadpool.");
    }

    /// Submit a new task.
    pub fn submit(&self, task: Task) {
        unimplemented!("We suggest saving the task, and notifying the worker(s)");
    }

    // We suggest extracting the implementation of the worker to an associated
    // function, like this one (however, it is not a part of the public
    // interface, so you can delete it if you implement it differently):
    fn worker_loop(&self) {
        unimplemented!("Initialize necessary variables.");

        loop {
            unimplemented!("Wait for a task and then execute it.");
            unimplemented!(
                "If there are no tasks, and the thread pool is to be shut down, break the loop."
            );
            unimplemented!("Be careful with locking! The tasks shall be executed concurrently.");
        }
    }
}

impl Drop for Threadpool {
    /// Gracefully end the thread pool.
    ///
    /// It waits until all submitted tasks are executed,
    /// and until all threads are joined.
    fn drop(&mut self) {
        unimplemented!("Notify the workers that the thread pool is to be shut down.");
        unimplemented!("Wait for all threads to be finished.");
    }
}
