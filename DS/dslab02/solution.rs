use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::thread::{JoinHandle, spawn};

type Task = Box<dyn FnOnce() + Send>;
type TaskArc = Arc<(Mutex<Vec<Task>>, Condvar)>;
// You can define new types (e.g., structs) if you need.
// However, they shall not be public (i.e., do not use the `pub` keyword).

/// The thread pool.
pub struct Threadpool {
    worker_count: usize,
    join_handles: Vec<JoinHandle<()>>,
    available_tasks: TaskArc,
    is_running: Arc<AtomicBool>,

    // Add here any fields you need.
    // We suggest storing handles of the worker threads, submitted tasks,
    // and information whether the pool is running or is shutting down.
}

struct Worker{
    available_tasks: TaskArc, 
    is_running: Arc<AtomicBool>,
}

impl Worker{
    fn worker_loop(&self) {
        let thread_id = thread::current().id();

        let wait_for_tasks= |vec: &Vec<Task>, ir: Arc<AtomicBool>| {
            let isr = ir.load(Ordering::Relaxed);
            let isemp = vec.is_empty();

            println!("isr: {}, ise: {} id: {:?}", isr, isemp, thread_id);
            return isr && isemp
        };

        loop {
            let cloned = self.available_tasks.clone();

            {
                let (lock, cond) = &*cloned;
                let mut guard = lock.lock().unwrap();
                while wait_for_tasks(&*guard, self.is_running.clone()) {
                    // If the predicate does not hold, call `wait()`. It atomically
                    // releases the mutex and waits for a notification. The while loop
                    // is required because of the possible spurious wakeups:
                    guard = cond.wait(guard).unwrap();
                }                

                // The predicate holds and the mutex is locked here.
                let my_task = guard.pop();
                match my_task{
                    Some(t) => t(),
                    None => {
                        let exit = !self.is_running.load(Ordering::Relaxed);

                        if exit {
                            println!("{:?} exiting!", thread_id);
                            return;
                        }
                        else{
                            panic!("The vector was not supposed to be empty!")
                        }
                    },
                }
            }

            //Be careful with locking! The tasks shall be executed concurrently.
        }
    }
}

fn is_sync<T: Sync>() {}
fn is_send<T: Send>() {}

impl Threadpool {
    /// Create new thread pool with `workers_count` workers.
    pub fn new(workers_count: usize) -> Self {
        let available_tasks: TaskArc =
            Arc::new((Mutex::new(Vec::new()), Condvar::new()));

        let is_running = Arc::new(AtomicBool::new(true));

        let mut join_handles: Vec<JoinHandle<()>> = Vec::new();

        is_sync::<Threadpool>();
        is_send::<Worker>();

        // Create the workers 
        for _ in 0..workers_count {
            let at_clone: TaskArc = available_tasks.clone();
            let ir_clone: Arc<AtomicBool> = is_running.clone();
            
            let worker = Worker{
                available_tasks: at_clone,
                is_running: ir_clone,
            };

            let thread = spawn(move || {
                worker.worker_loop();
            });

            join_handles.push(thread);
        }
        
        // Return the new Threadpool
        return Threadpool{
            worker_count: workers_count,
            join_handles: join_handles,
            available_tasks: available_tasks,
            is_running: is_running, 
        };
    }

    /// Submit a new task.
    pub fn submit(&self, task: Task) {
        let cloned = self.available_tasks.clone();
        {
            let (lock, cond) = &*cloned;
            let mut guard = lock.lock().unwrap();
            guard.push(task);

            // Wake up a thread waiting on the variable:
            cond.notify_one();
        }

        //We suggest saving the task, and notifying the worker(s).
    }

  }

impl Drop for Threadpool {
    /// Gracefully end the thread pool.
    ///
    /// It waits until all submitted tasks are executed,
    /// and until all threads are joined.
    
    fn drop(&mut self) {
        self.is_running.store(false, Ordering::Relaxed);
        
        let (_, cond) = &*self.available_tasks;

        cond.notify_one();

        for handle in self.join_handles.drain(..){
            handle.join().unwrap();
            cond.notify_one();
        }

        //Notify the workers that the thread pool is to be shut down.
        //Wait for all threads to be finished.
    }
}

#[cfg(test)]
mod tests {
    use crate::solution::Threadpool;
    use ntest::timeout;

    #[test]
    #[timeout(100000)]
    fn test1(){
        let tp = Threadpool::new(5);
        println!("spawn vec: {:?}", tp.join_handles);
    }
}