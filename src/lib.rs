use std::fmt::Debug;

use range_minimum_query::RangeMinimumQuery;

#[macro_use]
extern crate serde;
extern crate num;
pub mod bitvector;
pub mod range_minimum_query;
pub mod suffix_array;
pub mod suffix_tree;
#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

/// Struct to carry out longest common prefix between two prefix of two strings.
/// Specifically, for two input string `xs` and `ys`,
/// This data structure answers the following query in O(1) time.
/// - Input: `i` (an index of `xs`) and `j` (an index of `ys`).
/// - Output: `LCP(&xs[i..], &ys[j..])`
#[derive(Debug, Clone)]
pub struct LongestCommonPrefix {
    range_minimum_query: range_minimum_query::LinearRangeMinimumQuery<usize>,
    inverse_suffix_array: Vec<usize>,
    lcp_array: Vec<usize>,
    xslen: usize,
}

impl LongestCommonPrefix {
    pub fn new<T: Clone + Eq + Ord + Debug>(xs: &[T], ys: &[T], alphabet: &[T]) -> Self {
        let xslen = xs.len();
        // First, concatenate them.
        let alphabet: Vec<(_, u8)> = alphabet
            .iter()
            .enumerate()
            .map(|(i, t)| (t, i as u8 + 1))
            .collect();
        let to_lex = |c: &T| alphabet.iter().find(|r| r.0 == c).unwrap().1;
        let xs = xs.iter().map(to_lex);
        let ys = ys.iter().map(to_lex);
        let input: Vec<_> = xs.chain(std::iter::once(0)).chain(ys).collect();
        let alphabet: Vec<_> = (0..alphabet.len() as u8 + 1).collect();
        let sa = suffix_array::SuffixArray::new(&input, &alphabet);
        let inverse_sa = sa.inverse();
        let lcp_array = suffix_array::longest_common_prefix(&input, &sa, &inverse_sa);
        let rmq = range_minimum_query::LinearRangeMinimumQuery::new(&lcp_array);
        Self {
            xslen,
            range_minimum_query: rmq,
            inverse_suffix_array: inverse_sa,
            lcp_array,
        }
    }
    pub fn lcp(&self, xs_start: usize, ys_start: usize) -> usize {
        let x_suffix_rank = self.inverse_suffix_array[xs_start];
        let y_suffix_rank = self.inverse_suffix_array[self.xslen + 1 + ys_start];
        let (start, end) = match x_suffix_rank < y_suffix_rank {
            true => (x_suffix_rank, y_suffix_rank),
            false => (y_suffix_rank, x_suffix_rank),
        };
        let min_exact_match_idx = self.range_minimum_query.min(start + 1, end + 1);
        self.lcp_array[min_exact_match_idx]
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    use rand_xoshiro::Xoroshiro128PlusPlus;
    #[test]
    fn test_lcp() {
        let alphabet = b"ACGT";
        let xs = b"ACGTTTTT";
        let ys = b"ACGTCTCC";
        let lcp = LongestCommonPrefix::new(xs, ys, alphabet);
        for i in 0..4 {
            assert_eq!(lcp.lcp(i, i), 4 - i);
        }
        let len = 4;
        for i in 0..len {
            for j in 0..len {
                let match_len = lcp.lcp(i, j);
                let answer = std::iter::zip(&xs[i..], &ys[j..])
                    .take_while(|(x, y)| x == y)
                    .count();
                assert_eq!(match_len, answer, "{},{}", i, j);
            }
        }
    }
    #[test]
    fn test_lcp_random() {
        let len = 150;
        let alphabet_size = 10;
        let alphabet: Vec<_> = (0..alphabet_size).collect();
        //        let alphabet = b"ACGT";
        let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(348203);
        let xs: Vec<_> = (0..len)
            .filter_map(|_| alphabet.choose(&mut rng))
            .copied()
            .collect();
        let ys: Vec<_> = (0..len)
            .filter_map(|_| alphabet.choose(&mut rng))
            .copied()
            .collect();
        let lcp = LongestCommonPrefix::new(&xs, &ys, &alphabet);
        for i in 0..len {
            for j in 0..len {
                let match_len = lcp.lcp(i, j);
                let answer = std::iter::zip(&xs[i..], &ys[j..])
                    .take_while(|(x, y)| x == y)
                    .count();
                assert_eq!(match_len, answer, "{},{}", i, j);
            }
        }
    }
}
