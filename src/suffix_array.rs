//! Suffix Array module.
//! # Overview
//! Currently, we have only one struct, SuffixArray. It is just an array of usize, i.e., Vec<usize>, associated with the input string Vec<u8>
//! ,each element of which is the rank of the suffix starting from the index.
//! For example, assume the return value is SuffixArray sa for the input xs. Then, if sa[12] = 34, the rank of the suffix xs[34..] is 12.
//! Quite easy, huh?
use std::cmp::Ord;
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize, Default)]
pub struct SuffixArray<T: Ord + Clone + Eq> {
    inner: Vec<usize>,
    resource_type: std::marker::PhantomData<T>,
}

impl<T: Clone + Ord + Eq> std::convert::AsRef<[usize]> for SuffixArray<T> {
    fn as_ref(&self) -> &[usize] {
        self.inner.as_ref()
    }
}

impl<T: Clone + Ord + Eq> SuffixArray<T> {
    pub fn search(&self, input: &[T], query: &[T]) -> Option<usize> {
        let max = input.len();
        self.inner
            .binary_search_by(|&suf| {
                let end = (suf + query.len()).min(max);
                let suffix = &input[suf..end];
                suffix.cmp(query)
            })
            .ok()
            .map(|e| self.inner[e])
    }
    pub fn new(input: &[T], alphabet: &[T]) -> Self {
        // Renaming input into alphabetically order.
        let alphabet: Vec<(usize, T)> = alphabet.iter().cloned().enumerate().collect();
        let mut input: Vec<u64> = input
            .iter()
            .map(|x| {
                alphabet
                    .iter()
                    .filter(|c| &c.1 == x)
                    .map(|c| c.0 as u64 + 1)
                    .nth(0)
                    .expect("the input contains character not in the alphabet.")
            })
            .collect();
        input.push(0);
        // alphabet + sentinel character.
        let inner = Self::induced_sorting(input, alphabet.len() + 1);
        Self {
            inner,
            resource_type: std::marker::PhantomData,
        }
    }
    pub fn new_naive(input: &[T], alphabet: &[T]) -> Self {
        let alphabet: Vec<(usize, T)> = alphabet.iter().cloned().enumerate().collect();
        let mut input: Vec<u64> = input
            .iter()
            .map(|x| {
                alphabet
                    .iter()
                    .filter(|c| &c.1 == x)
                    .map(|c| c.0 as u64 + 1)
                    .nth(0)
                    .expect("the input contains character not in the alphabet.")
            })
            .collect();
        input.push(0);
        let mut inner: Vec<_> = (0..input.len()).collect();
        inner.sort_by(|&i, &j| input[i..].cmp(&input[j..]));
        Self {
            inner,
            resource_type: std::marker::PhantomData,
        }
    }
    fn induced_sorting(input: Vec<u64>, alphabet_size: usize) -> Vec<usize> {
        let lms_position = Self::first_induce(&input, alphabet_size);
        let sa_of_lms = Self::construct_sa_of_lms(&lms_position, &input);
        eprintln!("{:?}", sa_of_lms);
        Self::second_induce(&input, alphabet_size, &sa_of_lms)
    }
    fn first_induce(input: &[u64], alphabet_size: usize) -> Vec<(usize, usize)> {
        let (is_large, is_lms) = Self::determin_large_small(input);
        // If input.len(), it is empty.
        let mut approx_sa: Vec<usize> = vec![input.len(); input.len()];
        let mut bucket = Self::count_bucket(input, alphabet_size);
        // Set Small suffices.
        let mut position_of_samll_suffix = vec![false; input.len()];
        for (idx, (&c, &is_large)) in input.iter().zip(is_large.iter()).enumerate() {
            if !is_large {
                approx_sa[bucket[c as usize]] = idx;
                position_of_samll_suffix[bucket[c as usize]] = true;
                if bucket[c as usize] > 0 {
                    bucket[c as usize] -= 1;
                } else {
                    assert_eq!(c, 0);
                }
            }
        }
        // Forward path.
        let mut bucket = Self::count_bucket_front(input, alphabet_size);
        for position in 0..input.len() {
            // If approx_sa[pos] == 0, the previous suffix is 0, and it is already added.
            if approx_sa[position] != input.len() && approx_sa[position] > 0 {
                let prev_position = approx_sa[position] - 1;
                if is_large[prev_position] {
                    let c = input[prev_position];
                    approx_sa[bucket[c as usize]] = prev_position;
                    bucket[c as usize] += 1;
                }
            }
        }
        // Remove all the element of small suffices.
        approx_sa
            .iter_mut()
            .zip(position_of_samll_suffix)
            .filter(|&(_, b)| b)
            .for_each(|(x, _)| {
                *x = input.len();
            });
        // Reverse path.
        let mut bucket = Self::count_bucket(input, alphabet_size);
        for position in (0..input.len()).rev() {
            if approx_sa[position] != input.len() && approx_sa[position] > 0 {
                let prev_position = approx_sa[position] - 1;
                if !is_large[prev_position] {
                    let c = input[prev_position];
                    approx_sa[bucket[c as usize]] = prev_position;
                    bucket[c as usize] -= 1
                }
            }
        }
        approx_sa[0] = input.len() - 1;
        approx_sa
            .into_iter()
            .filter(|&x| is_lms[x])
            .map(|start| {
                let mut end = start + 1;
                while end < input.len() && !is_lms[end] {
                    end += 1;
                }
                (start, (end + 1).min(input.len()))
            })
            .collect()
    }
    fn count_bucket(input: &[u64], alphabet_size: usize) -> Vec<usize> {
        let mut bucket = vec![0; alphabet_size];
        for &x in input {
            bucket[x as usize] += 1;
        }
        bucket
            .into_iter()
            .fold((vec![], -1), |(mut bucket, mut acc), count| {
                acc += count as i64;
                assert!(acc >= 0);
                bucket.push(acc as usize);
                (bucket, acc)
            })
            .0
    }
    fn count_bucket_front(input: &[u64], alphabet_size: usize) -> Vec<usize> {
        let mut bucket = vec![0; alphabet_size];
        for &x in input {
            bucket[x as usize] += 1;
        }
        bucket
            .into_iter()
            .fold((vec![], 0), |(mut bucket, acc), count| {
                bucket.push(acc);
                (bucket, acc + count)
            })
            .0
    }
    fn construct_sa_of_lms(lms_position: &[(usize, usize)], input: &[u64]) -> Vec<usize> {
        let mut current_idx = 0;
        let new_array: Vec<_> = {
            let max = input.len() as u64;
            let mut result = vec![max; input.len()];
            for w in lms_position.windows(2) {
                let &(s1, t1) = &w[0];
                let &(s2, t2) = &w[1];
                current_idx += if input[s1..t1] == input[s2..t2] { 0 } else { 1 };
                result[s2] = current_idx as u64;
            }
            // The last chatacter is always fastest.
            result[input.len() - 1] = 0;
            result.into_iter().filter(|&e| e != max).collect()
        };
        if current_idx + 1 == new_array.len() {
            // They all different. Just sort by naive approach.
            let mut suffix_array = vec![0; new_array.len()];
            for (idx, rank) in new_array.into_iter().enumerate() {
                suffix_array[rank as usize] = idx;
            }
            suffix_array
        } else {
            Self::induced_sorting(new_array, current_idx + 1)
        }
    }
    fn second_induce(input: &[u64], alphabet_size: usize, sa_of_lms: &[usize]) -> Vec<usize> {
        let (is_large, is_lms) = Self::determin_large_small(input);
        let mut suffix_array = vec![input.len(); input.len()];
        // First fill the lmss.
        let mut bucket = Self::count_bucket(input, alphabet_size);
        let (lms_positions, _): (Vec<_>, Vec<&bool>) =
            is_lms.iter().enumerate().filter(|b| *b.1).unzip();
        assert_eq!(lms_positions.len(), sa_of_lms.len());
        let mut small_suf_position = vec![false; input.len()];
        for &index in sa_of_lms.iter().rev() {
            let position = lms_positions[index];
            let c = input[position];
            suffix_array[bucket[c as usize]] = position;
            small_suf_position[bucket[c as usize]] = true;
            if bucket[c as usize] > 0 {
                bucket[c as usize] -= 1;
            } else {
                assert_eq!(c, 0);
            }
        }
        // Then, induce sort for large suffices.
        let mut bucket = Self::count_bucket_front(input, alphabet_size);
        for i in 0..input.len() {
            if suffix_array[i] != input.len() && suffix_array[i] > 0 {
                let prev_position = suffix_array[i] - 1;
                if is_large[prev_position] {
                    let c = input[prev_position];
                    suffix_array[bucket[c as usize]] = prev_position;
                    bucket[c as usize] += 1;
                }
            }
        }
        // Remove all lms.
        suffix_array
            .iter_mut()
            .zip(small_suf_position)
            .filter(|&(_, b)| b)
            .for_each(|(x, _)| *x = input.len());
        // Reverse path.
        let mut bucket = Self::count_bucket(input, alphabet_size);
        for i in (0..input.len()).rev() {
            if suffix_array[i] != input.len() && suffix_array[i] > 0 {
                let prev_position = suffix_array[i] - 1;
                if !is_large[prev_position] {
                    let c = input[prev_position];
                    suffix_array[bucket[c as usize]] = prev_position;
                    bucket[c as usize] -= 1;
                }
            }
        }
        suffix_array[0] = input.len() - 1;
        suffix_array
    }
    fn determin_large_small(input: &[u64]) -> (Vec<bool>, Vec<bool>) {
        let mut is_large = vec![false; input.len()];
        for i in (0..input.len() - 1).rev() {
            is_large[i] = input[i + 1] < input[i] || (input[i + 1] == input[i] && is_large[i + 1]);
        }
        let mut is_lms = vec![false; input.len()];
        for i in 1..is_large.len() {
            is_lms[i] = (i == is_large.len() - 1) || (!is_large[i] && is_large[i - 1]);
        }
        (is_large, is_lms)
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
    fn naive() {
        let input = b"GTCCCGATGTCATGTCAGGA";
        let alphabet = b"ACGT";
        let result = SuffixArray::new_naive(input, alphabet);
        let answer = vec![
            20, 19, 16, 11, 6, 15, 10, 2, 3, 4, 18, 5, 17, 13, 8, 0, 14, 9, 1, 12, 7,
        ];
        assert_eq!(answer, result.inner);
    }
    #[test]
    fn sa_is() {
        let input = b"GTCCCGATGTCATGTCAGGA";
        let alphabet = b"ACGT";
        let result = SuffixArray::new(input, alphabet);
        let answer = vec![
            20, 19, 16, 11, 6, 15, 10, 2, 3, 4, 18, 5, 17, 13, 8, 0, 14, 9, 1, 12, 7,
        ];
        assert_eq!(answer, result.inner);
    }
    #[test]
    fn suffix_array_search() {
        let input = b"GTCCCGATGTCATGTCAGGA";
        let alphabet = b"ACGT";
        let result = SuffixArray::new(input, alphabet);
        let query = b"GTCCC";
        assert_eq!(result.search(input, query), Some(0));
        let query = b"CCCGATGTCATGTCAGGA";
        assert_eq!(result.search(input, query), Some(2));
        let query = b"CCCGATGTCTGTCAGGA";
        assert_eq!(result.search(input, query), None);
        let query = b"CCCCCC";
        assert_eq!(result.search(input, query), None);
    }
}
