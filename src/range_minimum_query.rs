//! Range minimum query on an ordered array H.

// ToDo: Do I need to record the original array?
#[derive(Debug, Clone)]
pub struct RangeMinimumQuery {
    large_blocks: Vec<usize>,
    small_blocks: Vec<usize>,
    block_size: usize,
}

impl RangeMinimumQuery {
    pub fn new<T: Ord + Copy + Clone + std::fmt::Debug>(input: &[T]) -> Self {
        todo!()
    }
}
