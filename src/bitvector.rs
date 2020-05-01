//! BitVector.
const SELECT_BUCKET: usize = 100;
#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq)]
pub struct BitVec {
    inner: Vec<u64>,
    bucket: Vec<u64>,
    // select_true[i] = Position of the 100*i-th true.
    select_true: Vec<usize>,
    select_false: Vec<usize>,
}

impl std::convert::AsRef<[u64]> for BitVec {
    fn as_ref(&self) -> &[u64] {
        self.inner.as_ref()
    }
}

impl BitVec {
    pub fn get(&self, index: usize) -> bool {
        let block = index / 64;
        let index = index % 64;
        let probe = 0b1 << index;
        (self.inner[block] & probe != 0)
    }
    pub fn new(xs: &[bool]) -> Self {
        let mut inner = vec![0; xs.len() / 64 + 1];
        for (idx, _) in xs.iter().enumerate().filter(|x| *x.1) {
            let bucket = idx / 64;
            let index = idx % 64;
            let probe = 0b1 << index;
            inner[bucket] |= probe;
        }
        let (_, bucket) = inner.iter().map(|&x: &u64| x.count_ones() as u64).fold(
            (0, vec![]),
            |(acc, mut bc), count| {
                bc.push(acc);
                (acc + count, bc)
            },
        );
        let (mut select_true, mut select_false) = (vec![0], vec![0]);
        let (mut pos, mut neg) = (0, 0);
        for (idx, &b) in xs.iter().enumerate() {
            if b {
                pos += 1;
                if pos % SELECT_BUCKET == 0 {
                    select_true.push(idx);
                }
            } else {
                neg += 1;
                if neg % SELECT_BUCKET == 0 {
                    select_false.push(idx);
                }
            }
        }
        Self {
            inner,
            bucket,
            select_true,
            select_false,
        }
    }
    pub fn rank(&self, x: bool, i: usize) -> usize {
        if x {
            let idx = i / 64;
            let rem = i % 64;
            let mask = (0b1 << rem) - 1;
            self.bucket[idx] as usize + (self.inner[idx] & mask).count_ones() as usize
        } else {
            i - self.rank(true, i)
        }
    }
    /// Return the i-th x. Note that the i begins one.
    /// In other words, self.rank(true, 0) would
    /// return zero and self.rank(true,1) would
    /// return the position of the first true.
    pub fn select(&self, x: bool, i: usize) -> usize {
        if i == 0 {
            return 0;
        }
        let block = {
            let compare = |position| {
                let count: usize = if x {
                    self.bucket[position] as usize
                } else {
                    64 * position - self.bucket[position] as usize
                };
                count.cmp(&i)
            };
            let chunk = i / SELECT_BUCKET;
            let (mut s, mut e) = if x {
                let s = self.select_true[chunk] / 64;
                let e = if chunk + 1 < self.select_true.len() {
                    self.select_true[chunk + 1] / 64
                } else {
                    self.bucket.len() - 1
                };
                (s, e)
            } else {
                let s = self.select_false[chunk] / 64;
                let e = if chunk + 1 < self.select_false.len() {
                    self.select_false[chunk + 1] / 64
                } else {
                    self.bucket.len() - 1
                };
                (s, e)
            };
            use std::cmp::Ordering::*;
            match compare(e) {
                Less => e,
                Equal | Greater => {
                    while e - s > 1 {
                        let center = (s + e) / 2;
                        match compare(center) {
                            std::cmp::Ordering::Less => s = center,
                            _ => e = center,
                        }
                    }
                    s
                }
            }
        };
        let mut occs_so_far = if x {
            self.bucket[block] as usize
        } else {
            64 * block - self.bucket[block] as usize
        };
        let window = if x {
            self.inner[block]
        } else {
            !self.inner[block]
        };
        let mut cursor = 0;
        while occs_so_far < i && cursor < 64 {
            occs_so_far += ((window & (1 << cursor)) != 0) as usize;
            cursor += 1;
        }
        if occs_so_far == i {
            block * 64 + cursor as usize - 1
        } else {
            self.inner.len() * 64
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn works() {
        assert!(true);
    }
    #[test]
    fn initialize() {
        let test = vec![true];
        dbg!(BitVec::new(&test));
        let test = vec![true, true, false];
        dbg!(BitVec::new(&test));
        let test = vec![false, false, false];
        dbg!(BitVec::new(&test));
        let test: Vec<_> = (0..1000000).map(|i| i % 232 == 0).collect();
        dbg!(BitVec::new(&test));
    }
    #[test]
    fn check() {
        let even_true: Vec<_> = (0..1000).map(|i| i % 2 == 0).collect();
        let bv = BitVec::new(&even_true);
        for (idx, &b) in even_true.iter().enumerate() {
            assert_eq!(bv.get(idx), b);
        }
        // Rank check
        for idx in 0..even_true.len() {
            // True query.
            let rank = even_true[..idx].iter().filter(|&&b| b).count();
            let rank_bv = bv.rank(true, idx);
            assert_eq!(rank, rank_bv, "{}\t{}\t{}", idx, rank, rank_bv);
            // False query
            let rank = even_true[..idx].iter().filter(|&&b| !b).count();
            let rank_bv = bv.rank(false, idx);
            assert_eq!(rank, rank_bv, "{}\t{}\t{}", idx, rank, rank_bv);
        }
        // Check select query.
        let number_of_true = even_true.iter().filter(|&&b| b).count();
        let number_of_false = even_true.len() - number_of_true;
        for i in 1..number_of_true {
            let mut acc = 0;
            let mut pos = 0;
            while acc < i {
                acc += even_true[pos] as usize;
                pos += 1;
            }
            let pos_bv = bv.select(true, i);
            assert_eq!(pos_bv, pos - 1, "{}\t{}\t{}", i, pos_bv, pos - 1);
        }
        for i in 1..number_of_false {
            let mut acc = 0;
            let mut pos = 0;
            while acc < i {
                acc += !even_true[pos] as usize;
                pos += 1;
            }
            let pos_bv = bv.select(false, i);
            assert_eq!(pos_bv, pos - 1, "{}\t{}\t{}", i, pos_bv, pos - 1);
        }
    }
}
