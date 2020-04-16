extern crate rand;
extern crate rand_xoshiro;
extern crate suffix_utils;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoroshiro128StarStar;
#[test]
fn test_works() {
    assert!(true);
}

#[test]
fn sa_is_works() {
    let mut rng: Xoroshiro128StarStar = SeedableRng::seed_from_u64(329);
    let alphabet = b"ACGT";
    let seq: Vec<_> = (0..1000)
        .filter_map(|_| alphabet.choose(&mut rng))
        .copied()
        .collect();
    let naive = suffix_utils::suffix_array::SuffixArray::new_naive(&seq, alphabet);
    let sa_is = suffix_utils::suffix_array::SuffixArray::new(&seq, alphabet);
    assert_eq!(naive, sa_is);
}
