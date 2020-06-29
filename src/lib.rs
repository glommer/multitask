//! An executor for running async tasks.

use std::cell::Cell;
use std::future::Future;
use std::marker::PhantomData;
use std::panic::{RefUnwindSafe, UnwindSafe};
use std::pin::Pin;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::task::{Context, Poll};
use std::thread::{self, ThreadId};

use concurrent_queue::ConcurrentQueue;
use scoped_tls::scoped_thread_local;
use slab::Slab;

/// A runnable future, ready for execution.
///
/// When a future is internally spawned using `async_task::spawn()` or `async_task::spawn_local()`,
/// we get back two values:
///
/// 1. an `async_task::Task<()>`, which we refer to as a `Runnable`
/// 2. an `async_task::JoinHandle<T, ()>`, which is wrapped inside a `Task<T>`
///
/// Once a `Runnable` is run, it "vanishes" and only reappears when its future is woken. When it's
/// woken up, its schedule function is called, which means the `Runnable` gets pushed into a task
/// queue in an executor.
type Runnable = async_task::Task<()>;

/// A spawned future.
///
/// Tasks are also futures themselves and yield the output of the spawned future.
///
/// When a task is dropped, its gets canceled and won't be polled again. To cancel a task a bit
/// more gracefully and wait until it stops running, use the [`cancel()`][Task::cancel()] method.
///
/// Tasks that panic get immediately canceled. Awaiting a canceled task also causes a panic.
///
/// If a task panics, the panic will be thrown by the [`Worker::tick()`] invocation that polled it.
///
/// # Examples
///
/// ```
/// use blocking::block_on;
/// use multitask::Queue;
/// use std::thread;
///
/// let queue = Queue::new();
///
/// // Spawn a future onto the queue.
/// let task = queue.spawn(async {
///     println!("Hello from a task!");
///     1 + 2
/// });
///
/// // Run an executor thread.
/// thread::spawn(move || {
///     let (p, u) = parking::pair();
///     let worker = queue.worker(move || u.unpark());
///     loop {
///         if !worker.tick() {
///             p.park();
///         }
///     }
/// });
///
/// // Wait for the result.
/// assert_eq!(block_on(task), 3);
/// ```
#[must_use = "tasks get canceled when dropped, use `.detach()` to run them in the background"]
#[derive(Debug)]
pub struct Task<T>(Option<async_task::JoinHandle<T, ()>>);

impl<T> Task<T> {
    /// Detaches the task to let it keep running in the background.
    ///
    /// # Examples
    ///
    /// ```
    /// use async_io::Timer;
    /// use multitask::Queue;
    /// use std::time::Duration;
    ///
    /// let queue = Queue::new();
    ///
    /// // Spawn a deamon future.
    /// queue.spawn(async {
    ///     loop {
    ///         println!("I'm a daemon task looping forever.");
    ///         Timer::new(Duration::from_secs(1)).await;
    ///     }
    /// })
    /// .detach();
    /// ```
    pub fn detach(mut self) {
        self.0.take().unwrap();
    }

    /// Cancels the task and waits for it to stop running.
    ///
    /// Returns the task's output if it was completed just before it got canceled, or [`None`] if
    /// it didn't complete.
    ///
    /// While it's possible to simply drop the [`Task`] to cancel it, this is a cleaner way of
    /// canceling because it also waits for the task to stop running.
    ///
    /// # Examples
    ///
    /// ```
    /// use async_io::Timer;
    /// use blocking::block_on;
    /// use multitask::Queue;
    /// use std::thread;
    /// use std::time::Duration;
    ///
    /// let queue = Queue::new();
    ///
    /// // Spawn a deamon future.
    /// let task = queue.spawn(async {
    ///     loop {
    ///         println!("Even though I'm in an infinite loop, you can still cancel me!");
    ///         Timer::new(Duration::from_secs(1)).await;
    ///     }
    /// });
    ///
    /// // Run an executor thread.
    /// thread::spawn(move || {
    ///     let (p, u) = parking::pair();
    ///     let worker = queue.worker(move || u.unpark());
    ///     loop {
    ///         if !worker.tick() {
    ///             p.park();
    ///         }
    ///     }
    /// });
    ///
    /// block_on(async {
    ///     Timer::new(Duration::from_secs(3)).await;
    ///     task.cancel().await;
    /// });
    /// ```
    pub async fn cancel(self) -> Option<T> {
        let mut task = self;
        let handle = task.0.take().unwrap();
        handle.cancel();
        handle.await
    }
}

impl<T> Drop for Task<T> {
    fn drop(&mut self) {
        if let Some(handle) = &self.0 {
            handle.cancel();
        }
    }
}

impl<T> Future for Task<T> {
    type Output = T;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.0.as_mut().unwrap()).poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(output) => Poll::Ready(output.expect("task has failed")),
        }
    }
}

scoped_thread_local! {
    static WORKER: Worker
}

/// State shared between [`Queue`] and [`Worker`].
struct Global {
    /// The global queue.
    queue: ConcurrentQueue<Runnable>,

    /// Shards of the global queue created by workers.
    shards: RwLock<Slab<Arc<ConcurrentQueue<Runnable>>>>,

    /// Set to `true` when a sleeping worker is notified or no workers are sleeping.
    notified: AtomicBool,

    /// A list of sleeping workers.
    sleepers: Mutex<Sleepers>,
}

impl Global {
    /// Notifies a sleeping worker.
    fn notify(&self) {
        if !self
            .notified
            .compare_and_swap(false, true, Ordering::SeqCst)
        {
            let callback = self.sleepers.lock().unwrap().notify();
            if let Some(cb) = callback {
                cb.call();
            }
        }
    }
}

/// A list of sleeping workers.
struct Sleepers {
    /// Number of sleeping workers (both notified and unnotified).
    count: usize,

    /// Callbacks of sleeping unnotified workers.
    ///
    /// A sleeping worker is notified when its callback is missing from this list.
    callbacks: Vec<Callback>,
}

impl Sleepers {
    /// Inserts a new sleeping worker.
    fn insert(&mut self, callback: &Callback) {
        self.count += 1;
        self.callbacks.push(callback.clone());
    }

    /// Re-inserts a sleeping worker's callback if it was notified.
    ///
    /// Returns `true` if the worker was notified.
    fn update(&mut self, callback: &Callback) -> bool {
        if self.callbacks.iter().all(|cb| cb != callback) {
            self.callbacks.push(callback.clone());
            true
        } else {
            false
        }
    }

    /// Removes a previously inserted sleeping worker.
    fn remove(&mut self, callback: &Callback) {
        self.count -= 1;
        for i in (0..self.callbacks.len()).rev() {
            if &self.callbacks[i] == callback {
                self.callbacks.remove(i);
                return;
            }
        }
    }

    /// Returns `true` if a sleeping worker is notified or no workers are sleeping.
    fn is_notified(&self) -> bool {
        self.count == 0 || self.count > self.callbacks.len()
    }

    /// Returns notification callback for a sleeping worker.
    ///
    /// If a worker was notified already or there are no workers, `None` will be returned.
    fn notify(&mut self) -> Option<Callback> {
        if self.callbacks.len() == self.count {
            self.callbacks.pop()
        } else {
            None
        }
    }
}

/// A queue for spawning futures.
pub struct Queue {
    global: Arc<Global>,
}

impl UnwindSafe for Queue {}
impl RefUnwindSafe for Queue {}

impl Queue {
    /// Creates a new queue for spawning futures.
    pub fn new() -> Queue {
        Queue {
            global: Arc::new(Global {
                queue: ConcurrentQueue::unbounded(),
                shards: RwLock::new(Slab::new()),
                notified: AtomicBool::new(true),
                sleepers: Mutex::new(Sleepers {
                    count: 0,
                    callbacks: Vec::new(),
                }),
            }),
        }
    }

    /// Spawns a future onto this queue.
    ///
    /// Returns a [`Task`] handle for the spawned future.
    pub fn spawn<T: Send + 'static>(
        &self,
        future: impl Future<Output = T> + Send + 'static,
    ) -> Task<T> {
        let global = self.global.clone();

        // The function that schedules a runnable task when it gets woken up.
        let schedule = move |runnable| {
            if WORKER.is_set() {
                WORKER.with(|w| {
                    if Arc::ptr_eq(&global, &w.global) {
                        if let Err(err) = w.shard.push(runnable) {
                            global.queue.push(err.into_inner()).unwrap();
                        }
                    } else {
                        global.queue.push(runnable).unwrap();
                    }
                });
            } else {
                global.queue.push(runnable).unwrap();
            }

            global.notify();
        };

        // Create a task, push it into the queue by scheduling it, and return its `Task` handle.
        let (runnable, handle) = async_task::spawn(future, schedule, ());
        runnable.schedule();
        Task(Some(handle))
    }

    /// Registers a new worker.
    ///
    /// The worker will automatically deregister itself when dropped.
    pub fn worker(&self, notify: impl Fn() + Send + Sync + 'static) -> Worker {
        let mut shards = self.global.shards.write().unwrap();
        let vacant = shards.vacant_entry();

        // Create a worker and put its stealer handle into the executor.
        let worker = Worker {
            key: vacant.key(),
            global: Arc::new(self.global.clone()),
            shard: Arc::new(ConcurrentQueue::bounded(512)),
            local: Arc::new(ConcurrentQueue::unbounded()),
            callback: Callback::new(notify),
            sleeping: Cell::new(false),
            ticker: Cell::new(0),
            _marker: PhantomData,
        };
        vacant.insert(worker.shard.clone());

        worker
    }
}

impl Default for Queue {
    fn default() -> Queue {
        Queue::new()
    }
}

/// A worker for running tasks and spawning thread-local futures.
pub struct Worker {
    /// The ID of this worker obtained during registration.
    key: usize,

    /// The global queue.
    global: Arc<Arc<Global>>,

    /// A shard of the global queue.
    shard: Arc<ConcurrentQueue<Runnable>>,

    /// Local queue for `!Send` tasks.
    local: Arc<ConcurrentQueue<Runnable>>,

    /// Callback invoked to wake this worker up.
    callback: Callback,

    /// Set to `true` when in sleeping state.
    ///
    /// States a worker can be in:
    /// 1) Woken.
    /// 2a) Sleeping and unnotified.
    /// 2b) Sleeping and notified.
    sleeping: Cell<bool>,

    /// Bumped every time a task is run.
    ticker: Cell<usize>,

    /// Make sure the type is `!Send` and `!Sync`.
    _marker: PhantomData<Rc<()>>,
}

impl UnwindSafe for Worker {}
impl RefUnwindSafe for Worker {}

impl Worker {
    /// Spawns a thread-local future onto this executor.
    ///
    /// Returns a [`Task`] handle for the spawned future.
    pub fn spawn_local<T: 'static>(&self, future: impl Future<Output = T> + 'static) -> Task<T> {
        let local = self.local.clone();
        let callback = self.callback.clone();
        let id = thread_id();

        // The function that schedules a runnable task when it gets woken up.
        let schedule = move |runnable| {
            if thread_id() == id && WORKER.is_set() {
                WORKER.with(|w| {
                    if Arc::ptr_eq(&local, &w.local) {
                        w.local.push(runnable).unwrap();
                    } else {
                        local.push(runnable).unwrap();
                    }
                });
            } else {
                local.push(runnable).unwrap();
            }

            callback.call();
        };

        // Create a task, push it into the queue by scheduling it, and return its `Task` handle.
        let (runnable, handle) = async_task::spawn_local(future, schedule, ());
        runnable.schedule();
        Task(Some(handle))
    }

    /// Moves the worker into sleeping and unnotified state.
    ///
    /// Returns `false` if the worker was already sleeping and unnotified.
    fn sleep(&self) -> bool {
        let mut sleepers = self.global.sleepers.lock().unwrap();

        if self.sleeping.get() {
            // Already sleeping, check if notified.
            if !sleepers.update(&self.callback) {
                return false;
            }
        } else {
            // Move to sleeping state.
            sleepers.insert(&self.callback);
        }

        self.global
            .notified
            .swap(sleepers.is_notified(), Ordering::SeqCst);

        self.sleeping.set(true);
        true
    }

    /// Moves the worker into woken state.
    ///
    /// Returns `false` if the worker was already woken.
    fn wake(&self) -> bool {
        if self.sleeping.get() {
            let mut sleepers = self.global.sleepers.lock().unwrap();
            sleepers.remove(&self.callback);

            self.global
                .notified
                .swap(sleepers.is_notified(), Ordering::SeqCst);
        }

        self.sleeping.replace(false)
    }

    /// Runs a single task and returns `true` if one was found.
    pub fn tick(&self) -> bool {
        loop {
            match self.search() {
                None => {
                    // Move to sleeping and unnotified state.
                    if !self.sleep() {
                        // If already sleeping and unnotified, return.
                        return false;
                    }
                }
                Some(r) => {
                    // Wake up.
                    self.wake();

                    // Notify another worker now to pick up where this worker left off, just in
                    // case running the task takes a long time.
                    self.global.notify();

                    // Bump the ticker.
                    let ticker = self.ticker.get();
                    self.ticker.set(ticker.wrapping_add(1));

                    // Steal tasks from the global queue to ensure fair task scheduling.
                    if ticker % 64 == 0 {
                        steal(&self.global.queue, &self.shard);
                    }

                    // Run the task.
                    WORKER.set(self, || r.run());

                    return true;
                }
            }
        }
    }

    /// Finds the next task to run.
    fn search(&self) -> Option<Runnable> {
        if self.ticker.get() % 2 == 0 {
            // On even ticks, look into the local queue and then into the shard.
            if let Ok(r) = self.local.pop().or_else(|_| self.shard.pop()) {
                return Some(r);
            }
        } else {
            // On odd ticks, look into the shard and then into the local queue.
            if let Ok(r) = self.shard.pop().or_else(|_| self.local.pop()) {
                return Some(r);
            }
        }

        // Try stealing from the global queue.
        steal(&self.global.queue, &self.shard);
        if let Ok(r) = self.shard.pop() {
            return Some(r);
        }

        // Try stealing from other shards.
        let shards = self.global.shards.read().unwrap();

        // Pick a random starting point in the iterator list and rotate the list.
        let n = shards.len();
        let start = fastrand::usize(..n);
        let iter = shards.iter().chain(shards.iter()).skip(start).take(n);

        // Remove this worker's shard.
        let iter = iter.filter(|(key, _)| *key != self.key);
        let iter = iter.map(|(_, shard)| shard);

        // Try stealing from each shard in the list.
        for shard in iter {
            steal(shard, &self.shard);
            if let Ok(r) = self.shard.pop() {
                return Some(r);
            }
        }

        None
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        // Wake and unregister the worker.
        self.wake();
        self.global.shards.write().unwrap().remove(self.key);

        // Re-schedule remaining tasks in the shard.
        while let Ok(r) = self.shard.pop() {
            r.schedule();
        }
        // Notify another worker to start searching for tasks.
        self.global.notify();

        // TODO(stjepang): Close the local queue and empty it.
        // TODO(stjepang): Cancel all remaining tasks.
    }
}

/// Steals some from one queue into another.
fn steal<T>(src: &ConcurrentQueue<T>, dest: &ConcurrentQueue<T>) {
    // Half of `src`'s length rounded up.
    let mut count = (src.len() + 1) / 2;

    if count > 0 {
        // Don't steal more than fits into the queue.
        if let Some(cap) = dest.capacity() {
            count = count.min(cap - dest.len());
        }

        // Steal tasks.
        for _ in 0..count {
            if let Ok(t) = src.pop() {
                assert!(dest.push(t).is_ok());
            } else {
                break;
            }
        }
    }
}

/// Same as `std::thread::current().id()`, but more efficient.
fn thread_id() -> ThreadId {
    thread_local! {
        static ID: ThreadId = thread::current().id();
    }

    ID.try_with(|id| *id)
        .unwrap_or_else(|_| thread::current().id())
}

/// A cloneable callback function.
#[derive(Clone)]
struct Callback(Arc<Box<dyn Fn() + Send + Sync>>);

impl Callback {
    fn new(f: impl Fn() + Send + Sync + 'static) -> Callback {
        Callback(Arc::new(Box::new(f)))
    }

    fn call(&self) {
        (self.0)();
    }
}

impl PartialEq for Callback {
    fn eq(&self, other: &Callback) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Callback {}
