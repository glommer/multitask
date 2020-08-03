//! An executor for running async tasks.

#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations, rust_2018_idioms)]

use std::cell::{Cell, RefCell};
use std::fmt;
use std::future::Future;
use std::marker::PhantomData;
use std::panic::{RefUnwindSafe, UnwindSafe};
use std::pin::Pin;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::task::{Context, Poll};

use cache_padded::CachePadded;
use concurrent_queue::ConcurrentQueue;
use thread_local::ThreadLocal;

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
/// If a task panics, the panic will be thrown by the [`Ticker::tick()`] invocation that polled it.
///
/// # Examples
///
/// ```
/// use blocking::block_on;
/// use multitask::Executor;
/// use std::thread;
///
/// let ex = Executor::new();
///
/// // Spawn a future onto the executor.
/// let task = ex.spawn(async {
///     println!("Hello from a task!");
///     1 + 2
/// });
///
/// // Run an executor thread.
/// thread::spawn(move || {
///     let (p, u) = parking::pair();
///     let ticker = ex.ticker(move || u.unpark());
///     loop {
///         if !ticker.tick() {
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
    /// use multitask::Executor;
    /// use std::time::Duration;
    ///
    /// let ex = Executor::new();
    ///
    /// // Spawn a deamon future.
    /// ex.spawn(async {
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
    /// use multitask::Executor;
    /// use std::thread;
    /// use std::time::Duration;
    ///
    /// let ex = Executor::new();
    ///
    /// // Spawn a deamon future.
    /// let task = ex.spawn(async {
    ///     loop {
    ///         println!("Even though I'm in an infinite loop, you can still cancel me!");
    ///         Timer::new(Duration::from_secs(1)).await;
    ///     }
    /// });
    ///
    /// // Run an executor thread.
    /// thread::spawn(move || {
    ///     let (p, u) = parking::pair();
    ///     let ticker = ex.ticker(move || u.unpark());
    ///     loop {
    ///         if !ticker.tick() {
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

/// A single-threaded executor.
#[derive(Debug)]
pub struct LocalExecutor {
    /// The task queue.
    queue: Arc<ConcurrentQueue<Runnable>>,

    /// Callback invoked to wake the executor up.
    callback: Callback,

    /// Make sure the type is `!Send` and `!Sync`.
    _marker: PhantomData<Rc<()>>,
}

impl UnwindSafe for LocalExecutor {}
impl RefUnwindSafe for LocalExecutor {}

impl LocalExecutor {
    /// Creates a new single-threaded executor.
    ///
    /// # Examples
    ///
    /// ```
    /// use multitask::LocalExecutor;
    ///
    /// let (p, u) = parking::pair();
    /// let ex = LocalExecutor::new(move || u.unpark());
    /// ```
    pub fn new(notify: impl Fn() + Send + Sync + 'static) -> LocalExecutor {
        LocalExecutor {
            queue: Arc::new(ConcurrentQueue::unbounded()),
            callback: Callback::new(notify),
            _marker: PhantomData,
        }
    }

    /// Spawns a thread-local future onto this executor.
    ///
    /// Returns a [`Task`] handle for the spawned future.
    ///
    /// # Examples
    ///
    /// ```
    /// use multitask::LocalExecutor;
    ///
    /// let (p, u) = parking::pair();
    /// let ex = LocalExecutor::new(move || u.unpark());
    ///
    /// let task = ex.spawn(async { println!("hello") });
    /// ```
    pub fn spawn<T: 'static>(&self, future: impl Future<Output = T> + 'static) -> Task<T> {
        let queue = self.queue.clone();
        let callback = self.callback.clone();

        // The function that schedules a runnable task when it gets woken up.
        let schedule = move |runnable| {
            queue.push(runnable).unwrap();
            callback.call();
        };

        // Create a task, push it into the queue by scheduling it, and return its `Task` handle.
        let (runnable, handle) = async_task::spawn_local(future, schedule, ());
        runnable.schedule();
        Task(Some(handle))
    }

    /// Runs a single task and returns `true` if one was found.
    ///
    /// # Examples
    ///
    /// ```
    /// use multitask::LocalExecutor;
    ///
    /// let (p, u) = parking::pair();
    /// let ex = LocalExecutor::new(move || u.unpark());
    ///
    /// assert!(!ex.tick());
    /// let task = ex.spawn(async { println!("hello") });
    ///
    /// // This prints "hello".
    /// assert!(ex.tick());
    /// ```
    pub fn tick(&self) -> bool {
        if let Ok(r) = self.queue.pop() {
            r.run();
            true
        } else {
            false
        }
    }
}

impl Drop for LocalExecutor {
    fn drop(&mut self) {
        // TODO(stjepang): Close the local queue and empty it.
        // TODO(stjepang): Cancel all remaining tasks.
    }
}

/// State shared between [`Executor`] and [`Ticker`].
#[derive(Debug)]
struct Global {
    /// The global queue.
    queue: ConcurrentQueue<Runnable>,

    /// Shards of the global queue created by tickers.
    shards: RwLock<Vec<Arc<ConcurrentQueue<Runnable>>>>,

    /// Set to `true` when a sleeping ticker is notified or no tickers are sleeping.
    notified: AtomicBool,

    /// A list of sleeping tickers.
    sleepers: Mutex<Sleepers>,

    current: CachePadded<ThreadLocal<RefCell<Option<(Arc<ConcurrentQueue<Runnable>>, Callback)>>>>,
}

impl Global {
    /// Notifies a sleeping ticker.
    #[inline]
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

/// A list of sleeping tickers.
#[derive(Debug)]
struct Sleepers {
    /// Number of sleeping tickers (both notified and unnotified).
    count: usize,

    /// Callbacks of sleeping unnotified tickers.
    ///
    /// A sleeping ticker is notified when its callback is missing from this list.
    callbacks: Vec<Callback>,
}

impl Sleepers {
    /// Inserts a new sleeping ticker.
    fn insert(&mut self, callback: &Callback) {
        self.count += 1;
        self.callbacks.push(callback.clone());
    }

    /// Re-inserts a sleeping ticker's callback if it was notified.
    ///
    /// Returns `true` if the ticker was notified.
    fn update(&mut self, callback: &Callback) -> bool {
        if self.callbacks.iter().all(|cb| cb != callback) {
            self.callbacks.push(callback.clone());
            true
        } else {
            false
        }
    }

    /// Removes a previously inserted sleeping ticker.
    fn remove(&mut self, callback: &Callback) {
        self.count -= 1;
        for i in (0..self.callbacks.len()).rev() {
            if &self.callbacks[i] == callback {
                self.callbacks.remove(i);
                return;
            }
        }
    }

    /// Returns `true` if a sleeping ticker is notified or no tickers are sleeping.
    fn is_notified(&self) -> bool {
        self.count == 0 || self.count > self.callbacks.len()
    }

    /// Returns notification callback for a sleeping ticker.
    ///
    /// If a ticker was notified already or there are no tickers, `None` will be returned.
    fn notify(&mut self) -> Option<Callback> {
        if self.callbacks.len() == self.count {
            self.callbacks.pop()
        } else {
            None
        }
    }
}

/// A multi-threaded executor.
#[derive(Debug)]
pub struct Executor {
    global: Arc<Global>,
}

impl UnwindSafe for Executor {}
impl RefUnwindSafe for Executor {}

impl Executor {
    /// Creates a new multi-threaded executor.
    ///
    /// # Examples
    ///
    /// ```
    /// use multitask::Executor;
    ///
    /// let ex = Executor::new();
    /// ```
    pub fn new() -> Executor {
        Executor {
            global: Arc::new(Global {
                queue: ConcurrentQueue::unbounded(),
                shards: RwLock::new(Vec::new()),
                notified: AtomicBool::new(true),
                sleepers: Mutex::new(Sleepers {
                    count: 0,
                    callbacks: Vec::new(),
                }),
                current: CachePadded::new(ThreadLocal::new()),
            }),
        }
    }

    /// Spawns a future onto this executor.
    ///
    /// Returns a [`Task`] handle for the spawned future.
    ///
    /// # Examples
    ///
    /// ```
    /// use multitask::Executor;
    ///
    /// let ex = Executor::new();
    /// let task = ex.spawn(async { println!("hello") });
    /// ```
    pub fn spawn<T: Send + 'static>(
        &self,
        future: impl Future<Output = T> + Send + 'static,
    ) -> Task<T> {
        let global = self.global.clone();

        // The function that schedules a runnable task when it gets woken up.
        let schedule = move |runnable| {
            if let Some(current) = global.current.get() {
                if let Some((shard, callback)) = &*current.borrow() {
                    shard.push(runnable).unwrap();
                    callback.call();
                    return;
                }
            }

            global.queue.push(runnable).unwrap();
            global.notify();
        };

        // Create a task, push it into the queue by scheduling it, and return its `Task` handle.
        let (runnable, handle) = async_task::spawn(future, schedule, ());
        runnable.schedule();
        Task(Some(handle))
    }

    /// Creates a new ticker for executing tasks.
    ///
    /// In a multi-threaded executor, each executor thread will create its own ticker and then keep
    /// calling [`Ticker::tick()`] in a loop.
    ///
    /// # Examples
    ///
    /// ```
    /// use blocking::block_on;
    /// use multitask::Executor;
    /// use std::thread;
    ///
    /// let ex = Executor::new();
    ///
    /// // Create two executor threads.
    /// for _ in 0..2 {
    ///     let (p, u) = parking::pair();
    ///     let ticker = ex.ticker(move || u.unpark());
    ///     thread::spawn(move || {
    ///         loop {
    ///             if !ticker.tick() {
    ///                 p.park();
    ///             }
    ///         }
    ///     });
    /// }
    ///
    /// // Spawn a future and wait for one of the threads to run it.
    /// let task = ex.spawn(async { 1 + 2 });
    /// assert_eq!(block_on(task), 3);
    /// ```
    pub fn ticker(&self, notify: impl Fn() + Send + Sync + 'static) -> Ticker {
        // Create a ticker and put its stealer handle into the executor.
        let ticker = Ticker {
            global: Arc::new(self.global.clone()),
            shard: Arc::new(ConcurrentQueue::bounded(512)),
            callback: Callback::new(notify),
            sleeping: Cell::new(false),
            ticks: Cell::new(0),
        };
        self.global
            .shards
            .write()
            .unwrap()
            .push(ticker.shard.clone());
        *self.global.current.get_or_default().borrow_mut() =
            Some((ticker.shard.clone(), ticker.callback.clone()));
        ticker
    }
}

impl Default for Executor {
    fn default() -> Executor {
        Executor::new()
    }
}

/// Runs tasks in a multi-threaded executor.
#[derive(Debug)]
pub struct Ticker {
    /// The global queue.
    global: Arc<Arc<Global>>,

    /// A shard of the global queue.
    shard: Arc<ConcurrentQueue<Runnable>>,

    /// Callback invoked to wake this ticker up.
    callback: Callback,

    /// Set to `true` when in sleeping state.
    ///
    /// States a ticker can be in:
    /// 1) Woken.
    /// 2a) Sleeping and unnotified.
    /// 2b) Sleeping and notified.
    sleeping: Cell<bool>,

    /// Bumped every time a task is run.
    ticks: Cell<usize>,
}

impl UnwindSafe for Ticker {}
impl RefUnwindSafe for Ticker {}

impl Ticker {
    /// Moves the ticker into sleeping and unnotified state.
    ///
    /// Returns `false` if the ticker was already sleeping and unnotified.
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

    /// Moves the ticker into woken state.
    ///
    /// Returns `false` if the ticker was already woken.
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

                    // Notify another ticker now to pick up where this ticker left off, just in
                    // case running the task takes a long time.
                    self.global.notify();

                    // Bump the ticker.
                    let ticks = self.ticks.get();
                    self.ticks.set(ticks.wrapping_add(1));

                    // Steal tasks from the global queue to ensure fair task scheduling.
                    if ticks % 64 == 0 {
                        steal(&self.global.queue, &self.shard);
                    }

                    // Run the task.
                    r.run();

                    return true;
                }
            }
        }
    }

    /// Finds the next task to run.
    fn search(&self) -> Option<Runnable> {
        if let Ok(r) = self.shard.pop() {
            return Some(r);
        }

        // Try stealing from the global queue.
        if let Ok(r) = self.global.queue.pop() {
            steal(&self.global.queue, &self.shard);
            return Some(r);
        }

        // Try stealing from other shards.
        let shards = self.global.shards.read().unwrap();

        // Pick a random starting point in the iterator list and rotate the list.
        let n = shards.len();
        let start = fastrand::usize(..n);
        let iter = shards.iter().chain(shards.iter()).skip(start).take(n);

        // Remove this ticker's shard.
        let iter = iter.filter(|shard| !Arc::ptr_eq(shard, &self.shard));

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

impl Drop for Ticker {
    fn drop(&mut self) {
        // Wake and unregister the ticker.
        self.wake();
        self.global
            .shards
            .write()
            .unwrap()
            .retain(|shard| !Arc::ptr_eq(shard, &self.shard));
        self.global.current.get_or_default().borrow_mut().take();

        // Re-schedule remaining tasks in the shard.
        while let Ok(r) = self.shard.pop() {
            r.schedule();
        }
        // Notify another ticker to start searching for tasks.
        self.global.notify();

        // TODO(stjepang): Cancel all remaining tasks.
    }
}

/// Steals some items from one queue into another.
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

impl fmt::Debug for Callback {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("<callback>").finish()
    }
}
