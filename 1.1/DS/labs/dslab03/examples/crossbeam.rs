// Instead of `std::sync::mpsc`, we will use an external crate here.
// We can freely import dependencies declared in `Cargo.toml` as below.
// Cargo takes care of providing the dependency and making it visible to the compiler.
// Compiling manually with `rustc` would be way more challenging.
use crossbeam_channel::{bounded, unbounded, Receiver, RecvTimeoutError, SendTimeoutError, Sender};
use std::time::Duration;

fn create_crossbeam_channel() {
    // Creating a channel results in two objects, not one.
    // This pattern is common in Rust as it fits the strict ownership requirements.
    // The `Sender` interface allows of inserting values into the channel.
    // It is `Clone` and `Send` to allow multiple producers into the channel.
    // The `Receiver` interface allows in turn for receiving values from the channel,
    // (optionally) blocking until a value is actually received.
    // Unlike `mpsc`, it also can be cloned to allow multiple reception points --
    // the cloning does not duplicate values the channel,
    // each sent value can be received only once.
    // You may think of the channel as a queue (backed by a ring buffer),
    // for which the put and pop operations are largely independent.
    // The objects are typically named `tx` and `rx` in Rust:
    let (tx, rx): (Sender<u8>, Receiver<u8>) = unbounded();

    // Alternatively:
    // let (tx, rx) = std::sync::mpsc::channel::<u8>();


    // Sending to an *unbounded* does not block:
    tx.send(7).unwrap();

    // Receiving blocks until a value arrives, or until all Sender objects of
    // this channel are dropped:
    let val = rx.recv().unwrap();

    println!("Received value: '{}'", val);
}

fn channel_ends_dropped() {
    {
        let (tx, rx): (Sender<u8>, Receiver<u8>) = unbounded();
        drop(tx);

        // When all Sender references of a channel are dropped,
        // the channel returns an error on an attempt to receive:
        assert!(rx.recv().is_err());
    }
    {
        let (tx, rx) = unbounded();
        drop(rx);

        // When all Receiver references of a channel are dropped,
        // the channel returns an error on an attempt to send:
        assert!(tx.send(0).is_err());
    }
}

fn timeout_receiving() {
    let (_tx, rx): (Sender<u8>, Receiver<u8>) = unbounded();

    // If required, the receiving operation can time out after a specified duration:
    assert_eq!(
        Err(RecvTimeoutError::Timeout),
        rx.recv_timeout(Duration::from_millis(20))
    );
}

fn communication_between_threads() {
    let (tx, rx) = unbounded();
    let tx_clone = tx.clone();

    // The required end can be moved to the other thread:
    std::thread::spawn(move || {
        tx.send(42).unwrap();
    });

    // Receive the value in the main thread:
    assert_eq!(Ok(42), rx.recv());

    // The main thread can still send new values using the cloned end:
    tx_clone.send(7).unwrap();

    // Receive the value:
    assert_eq!(Ok(7), rx.recv());
}

fn bounded_channel() {
    // There are also channels of limited capacity.
    // The maximal number of values such a channel can store is specified
    // when the channel is created:
    let (tx, _rx) = bounded(1);

    tx.send(7).unwrap();

    // An attempt to send more values than the capacity results
    // in blocking the thread until some value is received.
    // The following line would block:
    // tx.send(7).unwrap();
    // But, we can also specify a timeout here.
    // The value is given back on error.
    assert_eq!(
        Err(SendTimeoutError::Timeout(5)),
        tx.send_timeout(5, Duration::from_millis(20))
    );
}

fn zero_size_channel() {
    // It is possible to create a bounded channel of 0 capacity:
    let (tx, rx) = bounded(0);

    // As the channel cannot store any value, the sending operation will block
    // until a different thread executes the receive operation:
    std::thread::spawn(move || {
        tx.send(7).unwrap();
    });

    // The other thread is suspended until the message is received:
    rx.recv().unwrap();
}

fn iter_channel() {
    let (tx, rx) = bounded(1);

    std::thread::spawn(move || {
        tx.send(42).unwrap();
        std::thread::sleep(Duration::from_millis(20));
        tx.send(47).unwrap();
        // tx is implicitly dropped
    });

    // We can also read all messages in a loop.
    // Calls to `.next()` on the iterator will block until all senders are dropped.
    for val in rx.iter() {
        println!("Received value: '{}'", val);
    }
}

fn main() {
    create_crossbeam_channel();
    channel_ends_dropped();
    timeout_receiving();
    communication_between_threads();
    bounded_channel();
    zero_size_channel();
    iter_channel();
}
