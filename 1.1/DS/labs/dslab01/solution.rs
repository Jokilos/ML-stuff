#[derive(Debug)]
pub enum Counter{
    First,
    Second,
    Next,
}

#[derive(Debug)]
pub struct Fibonacci {
    // Add here any fields you need.
    n1 : u128,
    n0 : u128,
    ctr : Counter,
}

impl Fibonacci {
    /// Create new `Fibonacci`.
    pub fn new() -> Fibonacci {
        Fibonacci {n0: 0, n1: 1, ctr: Counter::First}
    }

    /// Calculate the n-th Fibonacci number.
    ///
    /// This shall not change the state of the iterator.
    /// The calculations shall wrap around at the boundary of u8.
    /// The calculations might be slow (recursive calculations are acceptable).
    pub fn fibonacci(n: usize) -> u8 {
        match n {
            0 | 1 => n as u8,
            _ => Fibonacci::fibonacci(n - 2).wrapping_add(Fibonacci::fibonacci(n - 1)), 
        }
    }
}

impl Iterator for Fibonacci {
    type Item = u128;

    /// Calculate the next Fibonacci number.
    ///
    /// The first call to `next()` shall return the 0th Fibonacci number (i.e., `0`).
    /// The calculations shall not overflow and shall not wrap around. If the result
    /// doesn't fit u128, the sequence shall end (the iterator shall return `None`).
    /// The calculations shall be fast (recursive calculations are **un**acceptable).
    fn next(&mut self) -> Option<Self::Item> {
        match self.ctr{
            Counter::First => {
                self.ctr = Counter::Second;
                Some(0)
            },

            Counter::Second => {
                self.ctr = Counter::Next;
                Some(1)
            },

            Counter::Next => {
                let n1: u128 = self.n1;
                let new_n1: Option<u128> = self.n0.checked_add(self.n1);
                self.n0 = n1;

                match new_n1 {
                    Some(value) => self.n1 = value,
                    _ => {},
                };

                new_n1
            },
        }
    }
}
