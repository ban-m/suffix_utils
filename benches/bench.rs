#![feature(test)]
extern crate rand;
extern crate rand_xoshiro;
extern crate test;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoroshiro128Plus;

fn dataset(
    seed: u64,
    template_len: usize,
    test_num: usize,
    test_max: usize,
) -> (Vec<u8>, Vec<Vec<u8>>) {
    let mut rng: Xoroshiro128Plus = SeedableRng::seed_from_u64(seed);
    let mut template: Vec<u8> = (0..template_len)
        .map(|_| rng.gen_range(0, std::u8::MAX))
        .collect();
    template.push(0);
    let tests: Vec<Vec<_>> = (0..test_num)
        .map(|_| {
            let len = rng.gen_range(1, test_max);
            (0..len).map(|_| rng.gen_range(0, std::u8::MAX)).collect()
        })
        .collect();
    (template, tests)
}

const TEMPLATE_LEN: usize = 10_000;
const TEST_NUM: usize = 100;
const MAX_LEN: usize = 100;
const SEED: u64 = 12910489034;
use test::bench::Bencher;
#[bench]
fn naive_check(b: &mut Bencher) {
    let (reference, queries) = dataset(SEED, TEMPLATE_LEN, TEST_NUM, MAX_LEN);
    b.iter(|| {
        test::bench::black_box(
            queries
                .iter()
                .filter(|query| {
                    reference
                        .windows(query.len())
                        .any(|w| w == query.as_slice())
                })
                .count(),
        )
    });
}

#[bench]
fn suffix_array_check(b: &mut Bencher) {
    let (reference, queries) = dataset(SEED, TEMPLATE_LEN, TEST_NUM, MAX_LEN);
    use suffix_utils::suffix_array::SuffixArray;
    let alphabet: Vec<u8> = (0..=std::u8::MAX).collect();
    let sa = SuffixArray::new_naive(&reference, &alphabet);
    b.iter(|| {
        test::bench::black_box(
            queries
                .iter()
                .filter(|query| sa.search(&reference, query).is_some())
                .count(),
        )
    });
}

#[test]
fn random_check() {
    let (reference, queries) = dataset(SEED, TEMPLATE_LEN, TEST_NUM, MAX_LEN);
    use suffix_utils::suffix_array::SuffixArray;
    let alphabet: Vec<u8> = (0..=std::u8::MAX).collect();
    let sa = SuffixArray::new_naive(&reference, &alphabet);
    for query in queries {
        let have = reference
            .windows(query.len())
            .any(|w| w == query.as_slice());
        assert_eq!(have, sa.search(&reference, &query).is_some());
    }
}

const BV_LEN: usize = 100_000_000;

#[bench]
fn random_vb_rank(b: &mut Bencher) {
    let mut rng: Xoroshiro128Plus = SeedableRng::seed_from_u64(12908320);
    let bitvec: Vec<_> = (0..BV_LEN).map(|_| rng.gen()).collect();
    let bv = suffix_utils::bitvector::BitVec::new(&bitvec);
    for (idx, &b) in bitvec.iter().enumerate() {
        assert_eq!(bv.get(idx), b);
    }
    b.iter(|| test::black_box(bv.rank(true, 12090)));
}

#[bench]
fn random_vb_select(b: &mut Bencher) {
    let mut rng: Xoroshiro128Plus = SeedableRng::seed_from_u64(12908320);
    let bitvec: Vec<_> = (0..BV_LEN).map(|_| rng.gen()).collect();
    let bv = suffix_utils::bitvector::BitVec::new(&bitvec);
    for (idx, &b) in bitvec.iter().enumerate() {
        assert_eq!(bv.get(idx), b);
    }
    b.iter(|| test::black_box(bv.select(true, 1090)));
}

#[bench]
fn random_naive_rank(b: &mut Bencher) {
    let mut rng: Xoroshiro128Plus = SeedableRng::seed_from_u64(12908320);
    let bitvec: Vec<_> = (0..BV_LEN).map(|_| rng.gen()).collect();
    b.iter(|| test::black_box(bitvec[..12090].iter().filter(|&&b| b).count()));
}

#[bench]
fn random_naive_select(b: &mut Bencher) {
    let mut rng: Xoroshiro128Plus = SeedableRng::seed_from_u64(12908320);
    let bitvec: Vec<bool> = (0..BV_LEN).map(|_| rng.gen()).collect();
    b.iter(|| {
        let i = 1090;
        let mut acc = 0;
        let mut pos = 0;
        while acc < i {
            acc += bitvec[pos] as usize;
            pos += 1;
        }
        test::black_box(pos)
    });
}
