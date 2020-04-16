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
