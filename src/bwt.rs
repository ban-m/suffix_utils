use crate::suffix_array::SuffixArray;

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize, Default)]
pub struct BWT<T: Ord + Clone + Eq> {
    inner: Vec<T>,
    blank_position: usize,
}

impl<T: Clone + Ord + Eq> std::convert::AsRef<[T]> for BWT<T> {
    fn as_ref(&self) -> &[T] {
        self.inner.as_ref()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize, Default)]
pub struct Occ<T: Ord + Clone + Eq> {
    count: Vec<usize>,
}

impl<T: Ord + Clone + Eq> Occ<T> {
    pub fn new(input: &[T], alphabet: &[T]) -> Self {
        let mut count = vec![0; alphabet];
        for x in input.iter() {
            let idx = alphabet.binary_search(x).unwrap();
            count[idx] += 1;
        }
        let mut acc = 0;
        let count = count
            .into_iter()
            .map(|c| {
                let x = 1 + acc;
                acc += c;
                x
            })
            .collect();
        Self { count }
    }
}

impl<T: Ord + Clone + Eq> std::ops::Index<usize> for BWT<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.inner.index(index)
    }
}

impl<T: Clone + Ord + Eq + Debug> BWT<T> {
    pub fn new(input: &[T], alphabet: &[T]) -> Self {
        let sa = SuffixArray::new(input, alphabet);
        let mut blank_position = 0;
        let inner: Vec<_> = sa
            .as_ref()
            .iter()
            .enumerate()
            .map(|(idx, pos_at_text)| {
                if pos_at_text == 0 {
                    blank_position = idx;
                    input[0].clone()
                } else {
                    input[pos_at_text - 1].clone()
                }
            })
            .collect();
        Self {
            inner,
            blank_position,
        }
    }
}
