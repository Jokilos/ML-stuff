#[cfg(test)]
mod tests {
    use crate::solution::Fibonacci;
    use ntest::timeout;

    #[test]
    #[timeout(100)]
    fn fibonacci_smoke_test() {
        assert_eq!(Fibonacci::fibonacci(3), 2);
    }

    #[test]
    #[timeout(100)]
    fn iterator_smoke_test() {
        let mut fib = Fibonacci::new();
        assert_eq!(fib.nth(3), Some(2));
    }

    #[test]
    fn collect() {
        let fib = Fibonacci::new();
        //let vec: Vec<u128> = fib.take(10).collect();
        println!("{:?}", fib.take(10).collect::<Vec<u128>>());
    }
}
