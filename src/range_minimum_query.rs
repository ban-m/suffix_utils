//! Range minimum query on an ordered array H.

use std::fmt::Debug;
#[derive(Debug, Clone)]
pub struct LinearRangeMinimumQuery<T: Ord + Eq + Copy + Clone + Debug> {
    data: Vec<T>,
    large_blocks: SparseTableRangeMinimumQuery<T>,
    small_blocks: Vec<BitTable>,
    bucket_size: usize,
}

pub trait RangeMinimumQuery {
    fn min(&self, start: usize, end: usize) -> usize;
}

impl<T: Ord + Eq + Copy + Clone + Debug> RangeMinimumQuery for LinearRangeMinimumQuery<T> {
    fn min(&self, start: usize, end: usize) -> usize {
        assert!(start < end);
        let start_bucket_idx = start / self.bucket_size;
        let end_bucket_idx = end / self.bucket_size - (end % self.bucket_size == 0) as usize;
        if start_bucket_idx == end_bucket_idx {
            let offset = start_bucket_idx * self.bucket_size;
            let inner_start = start - offset;
            let inner_end = end - offset;
            assert!(inner_end - inner_start <= self.bucket_size);
            let inner_rmq = self.small_blocks[start_bucket_idx].min(inner_start, inner_end);
            inner_rmq + offset
        } else {
            let start_rmq = {
                let offset = start_bucket_idx * self.bucket_size;
                let inner_start = start - offset;
                self.small_blocks[start_bucket_idx].min(inner_start, self.bucket_size) + offset
            };
            let end_rmq = {
                let offset = end_bucket_idx * self.bucket_size;
                let inner_end = end - offset;
                self.small_blocks[end_bucket_idx].min(0, inner_end) + offset
            };
            if self.data.len() <= end_rmq {
                println!("{},{},{}", self.data.len(), start, end);
                println!("{end_bucket_idx}\t{end_rmq}");
            }
            let (rmq, value) = match self.data[start_rmq] <= self.data[end_rmq] {
                true => (start_rmq, self.data[start_rmq]),
                false => (end_rmq, self.data[end_rmq]),
            };
            if start_bucket_idx + 1 == end_bucket_idx {
                rmq
            } else {
                let inner_idx = self.large_blocks.min(start_bucket_idx + 1, end_bucket_idx);
                // Search the exact location.
                let offset = self.bucket_size * inner_idx;
                let inner_rmq = self.small_blocks[inner_idx].min(0, self.bucket_size) + offset;
                let inner_value = self.data[inner_rmq];
                match inner_value <= value {
                    true => inner_rmq,
                    false => rmq,
                }
            }
        }
    }
}

// We fix the size of the bucket to 64 (for efficient bitvector allocations).
// Usually, it does not harm the performance(?)
impl<'a, T: Ord + Eq + Copy + Clone + Debug> LinearRangeMinimumQuery<T> {
    pub fn new(input: &'a [T]) -> Self {
        let bucket_size = 64;
        let bucket_num = input.len() / bucket_size + 1;
        let large_blocks: Vec<_> = (0..bucket_num)
            .map(|idx| {
                let start = bucket_size * idx;
                let end = (bucket_size * (idx + 1)).min(input.len());
                *input[start..end].iter().min().unwrap()
            })
            .collect();
        let large_blocks = SparseTableRangeMinimumQuery::new(&large_blocks);
        let small_blocks: Vec<_> = input
            .chunks(bucket_size)
            .map(|bucket| BitTable::new(bucket))
            .collect();
        Self {
            data: input.to_vec(),
            large_blocks,
            small_blocks,
            bucket_size,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SparseTableRangeMinimumQuery<T: Ord + Eq + Copy + Clone + Debug> {
    data: Vec<T>,
    table: Vec<Vec<usize>>,
}

impl<T: Ord + Eq + Copy + Clone + Debug> RangeMinimumQuery for SparseTableRangeMinimumQuery<T> {
    fn min(&self, start: usize, end: usize) -> usize {
        assert!(start < end);
        let diff = (64 - ((end - start) as u64).leading_zeros()) as usize - 1;
        let former = self.table[start][diff];
        let latter = self.table[end - (1 << diff)][diff];
        if self.data[former] <= self.data[latter] {
            former
        } else {
            latter
        }
    }
}

impl<T: Ord + Eq + Copy + Clone + Debug> SparseTableRangeMinimumQuery<T> {
    pub fn new(input: &[T]) -> Self {
        let len = input.len() as u64;
        let msb = (64 - len.leading_zeros()) as usize - 1;
        // table[i][p] = RMQ(input[i..i+2^p])
        let mut table: Vec<_> = (0..input.len()).map(|i| vec![i]).collect();
        for p in 0..msb {
            for i in 0..input.len() {
                let former = table[i][p];
                let latter = (i + (1 << p)).min(input.len() - 1);
                let latter = table[latter][p];
                if input[former] <= input[latter] {
                    table[i].push(former);
                } else {
                    table[i].push(latter)
                }
            }
        }
        Self {
            data: input.to_vec(),
            table,
        }
    }
}

// Bit tables. Used for small Range Minimum Query.
#[derive(Debug, Clone)]
pub struct BitTable {
    recorded_rmq: Vec<u64>,
}

impl RangeMinimumQuery for BitTable {
    fn min(&self, start: usize, end: usize) -> usize {
        let mask = !((1 << start) - 1);
        let probe = mask & self.recorded_rmq[end];
        probe.trailing_zeros() as usize
    }
}

impl BitTable {
    pub fn new<T: Ord + Eq + Copy + Clone + Debug>(input: &[T]) -> Self {
        let nearest_leq: Vec<Option<usize>> = get_nearest_smaller_or_equal(input);
        let recorded_rmq = get_recorded_bits(&nearest_leq);
        Self { recorded_rmq }
    }
}

fn get_nearest_smaller_or_equal<T: Ord + Eq + Copy + Clone + Debug>(
    input: &[T],
) -> Vec<Option<usize>> {
    assert!(input.len() <= 64);
    let mut stack = vec![];
    let mut nearest_smaller = vec![];
    'outer: for (i, x) in input.iter().enumerate() {
        while let Some(&(j, y)) = stack.last() {
            if y <= x {
                nearest_smaller.push(Some(j));
                stack.push((i, x));
                continue 'outer;
            } else {
                stack.pop();
            }
        }
        nearest_smaller.push(None);
        stack.push((i, x));
    }
    assert_eq!(input.len(), nearest_smaller.len());
    nearest_smaller
}

// Return bit vector b where (b[j] >> i) & 1 == 1 if min(input[i..j]) is i.
fn get_recorded_bits(nearest_smaller_or_equal: &[Option<usize>]) -> Vec<u64> {
    let mut bit_vector = vec![0];
    for (j, &prev_smaller) in nearest_smaller_or_equal.iter().enumerate() {
        match prev_smaller {
            None => bit_vector.push(1 << j),
            Some(p) => {
                let next = bit_vector[p + 1] | (1 << p) | (1 << j);
                bit_vector.push(next);
            }
        }
    }
    assert_eq!(nearest_smaller_or_equal.len() + 1, bit_vector.len());
    bit_vector
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand::SeedableRng;
    use rand_xoshiro::Xoroshiro128PlusPlus;
    #[test]
    fn nearest_smaller() {
        let input = vec![1, 2, 3, 4];
        let nearest_sm = get_nearest_smaller_or_equal(&input);
        assert_eq!(nearest_sm, vec![None, Some(0), Some(1), Some(2)]);
        let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(394820);
        let len = 60;
        let input: Vec<_> = (0..len).map(|_| rng.gen_range(0..len)).collect();
        let nearest_sm = get_nearest_smaller_or_equal(&input);
        for (i, &sm) in nearest_sm.iter().enumerate() {
            let x = input[i];
            let answer = input[..i]
                .iter()
                .enumerate()
                .filter_map(|(j, &y)| (y <= x).then_some(j))
                .max();
            assert_eq!(answer, sm);
        }
    }
    #[test]
    fn recorded_bits() {
        let input = vec![1, 2, 3, 1, 1];
        let nearest_sm = get_nearest_smaller_or_equal(&input);
        let bits = get_recorded_bits(&nearest_sm);
        for b in bits.iter() {
            println!("{b:5b}");
        }
        for (j, bit) in bits.iter().enumerate().skip(1) {
            for (i, y) in input[..j].iter().enumerate() {
                let min = input[i..j].iter().min().unwrap();
                assert_eq!((bit >> i) & 1 == 1, y == min);
            }
        }
        let len = 63;
        let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(394820);
        let input: Vec<_> = (0..len).map(|_| rng.gen_range(0..len)).collect();
        let nearest_sm = get_nearest_smaller_or_equal(&input);
        let bits = get_recorded_bits(&nearest_sm);
        for (j, bit) in bits.iter().enumerate().skip(1) {
            for (i, y) in input[..j].iter().enumerate() {
                let min = input[i..j].iter().min().unwrap();
                assert_eq!((bit >> i) & 1 == 1, y == min);
            }
        }
    }
    #[test]
    fn construct() {
        let input = vec![1, 2, 3, 4, 5, 6];
        let _ = LinearRangeMinimumQuery::new(&input);
        let _ = SparseTableRangeMinimumQuery::new(&input);
        let _ = BitTable::new(&input);
    }
    #[test]
    fn check_st_table() {
        let input = vec![1, 2, 3, 4];
        let st = SparseTableRangeMinimumQuery::new(&input);
        assert_eq!(st.table[0], vec![0, 0, 0]);
        assert_eq!(st.table[1], vec![1, 1, 1]);
    }
    macro_rules! order_1 {
        ($ty:ty) => {
            let input: Vec<usize> = vec![1, 2, 3, 4, 5];
            let rmq = <$ty>::new(&input);
            for i in 0..input.len() {
                assert_eq!(i, rmq.min(i, i + 1));
                assert_eq!(0, rmq.min(0, i + 1));
            }
            assert_eq!(2, rmq.min(2, 5));
        };
    }
    macro_rules! order_random {
        ($ty:ty, $len:expr) => {
            let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(348203);
            let input: Vec<_> = (0..$len).map(|_| rng.gen_range(0..1000)).collect();
            let rmq = <$ty>::new(&input);
            for _ in 0..100 {
                let start = rng.gen_range(0..$len - 1);
                let end = rng.gen_range(1..$len - start) + start;
                let min = *input[start..end].iter().min().unwrap();
                let is_ok = input
                    .iter()
                    .enumerate()
                    .take(end)
                    .skip(start)
                    .filter_map(|(i, &x)| (x == min).then_some(i))
                    .any(|i| i == rmq.min(start, end));
                assert!(is_ok);
            }
        };
    }
    #[test]
    fn linear_rmq_test() {
        let len = 150;
        let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(348203);
        let input: Vec<_> = (0..len).map(|_| rng.gen_range(0..1000)).collect();
        // let mut input: Vec<_> = (0..len).map(|_| rng.gen_range(0..1000)).collect();
        // input.sort();
        // input.dedup();
        // let len = input.len();
        let rmq = LinearRangeMinimumQuery::new(&input);
        println!("{input:?}");
        for _ in 0..100 {
            let start = rng.gen_range(0..len - 1);
            let end = rng.gen_range(1..len - start) + start;
            assert!(start < end);
            let min = *input[start..end].iter().min().unwrap();
            let is_ok = input
                .iter()
                .enumerate()
                .take(end)
                .skip(start)
                .filter_map(|(i, &x)| (x == min).then_some(i))
                .any(|i| i == rmq.min(start, end));
            if !is_ok {
                let min_idx: Vec<_> = input
                    .iter()
                    .enumerate()
                    .take(end)
                    .skip(start)
                    .filter_map(|(i, &x)| (x == min).then_some(i))
                    .collect();
                let rmq = rmq.min(start, end);
                let rmq_value = input[rmq];
                println!("{start},{end},{min},{min_idx:?},{rmq},{rmq_value}");
            }
            assert!(is_ok);
        }
    }
    #[test]
    fn order_test() {
        order_1![LinearRangeMinimumQuery<usize>];
        order_1![SparseTableRangeMinimumQuery<usize>];
        order_1![BitTable];
        order_random!(LinearRangeMinimumQuery<usize>, 10_000);
        order_random!(SparseTableRangeMinimumQuery<usize>, 1_000);
        order_random!(BitTable, 60);
    }
}
