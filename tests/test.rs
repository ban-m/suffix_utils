extern crate rand;
extern crate rand_xoshiro;
extern crate suffix_utils;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoroshiro128StarStar;

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

const TEMPLATE_LEN: usize = 10_000;
const TEST_NUM: usize = 100;
const MAX_LEN: usize = 100;
const SEED: u64 = 12910489034;
fn dataset(
    seed: u64,
    template_len: usize,
    test_num: usize,
    test_max: usize,
) -> (Vec<u8>, Vec<Vec<u8>>) {
    let mut rng: Xoroshiro128StarStar = SeedableRng::seed_from_u64(seed);
    let template: Vec<u8> = (0..template_len)
        .map(|_| rng.gen_range(0..std::u8::MAX))
        .collect();
    let tests: Vec<Vec<_>> = (0..test_num)
        .map(|_| {
            let len = rng.gen_range(1..test_max);
            (0..len).map(|_| rng.gen_range(0..std::u8::MAX)).collect()
        })
        .collect();
    (template, tests)
}

#[test]
fn random_check() {
    for i in 0..20 {
        let (reference, queries) = dataset(SEED + i as u64, TEMPLATE_LEN, TEST_NUM, MAX_LEN);
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
}

#[test]
fn random_check_lcp() {
    for i in 0..20 {
        let (reference, _queries) = dataset(SEED + i as u64, TEMPLATE_LEN, TEST_NUM, MAX_LEN);
        use suffix_utils::suffix_array;
        use suffix_utils::suffix_array::SuffixArray;
        let alphabet: Vec<u8> = (0..=std::u8::MAX).collect();
        let sa = SuffixArray::new_naive(&reference, &alphabet);
        let isa = sa.inverse();
        let lcp = suffix_array::longest_common_prefix(&reference, &sa, &isa);
        for i in 2..lcp.len() {
            let x = &reference[sa[i - 1]..sa[i - 1] + lcp[i]];
            let y = &reference[sa[i]..sa[i] + lcp[i]];
            assert_eq!(x, y);
            let l = lcp[i];
            assert!(
                sa[i] + l >= reference.len()
                    || sa[i - 1] + l >= reference.len()
                    || reference[sa[i] + l] != reference[sa[i - 1] + l]
            )
        }
    }
}

#[test]
fn random_check_suffix_tree() {
    for i in 0..2 {
        let (reference, _queries) = dataset(SEED + i as u64, TEMPLATE_LEN, TEST_NUM, MAX_LEN);
        use suffix_utils::suffix_tree::SuffixTree;
        let alphabet: Vec<u8> = (0..=std::u8::MAX).collect();
        let st = SuffixTree::new(&reference, &alphabet);
        let mut stack = vec![];
        stack.push(st.root_idx);
        let mut suffix = vec![];
        let mut arrived = vec![false; st.nodes.len()];
        let mut input = reference.to_vec();
        input.push(b'$');
        'dfs: while !stack.is_empty() {
            let node = *stack.last().unwrap();
            if !arrived[node] {
                arrived[node] = true;
            }
            for &(idx, _, _) in st.nodes[node].children.iter() {
                if !arrived[idx] {
                    let position = st.nodes[idx].position_at_text;
                    let length = st.nodes[idx].label_length_to_parent;
                    suffix.extend(input[position - length..position].iter().copied());
                    stack.push(idx);
                    continue 'dfs;
                }
            }
            let last = stack.pop().unwrap();
            if let Some(idx) = st.nodes[last].leaf_label {
                assert_eq!(suffix.as_slice(), &input[idx..]);
            }
            for _ in 0..st.nodes[last].label_length_to_parent {
                suffix.pop();
            }
        }
    }
}

#[test]
fn random_check_maximal_repeat() {
    for i in 0..2 {
        let (reference, _queries) = dataset(SEED + i as u64, TEMPLATE_LEN, TEST_NUM, MAX_LEN);
        use suffix_utils::suffix_tree::SuffixTree;
        let alphabet: Vec<u8> = (0..=std::u8::MAX).collect();
        let st = SuffixTree::new(&reference, &alphabet);
        for (starts, len) in st.maximul_repeat(&reference) {
            // Check repetitiveness.
            let subseq = &reference[starts[0]..starts[0] + len];
            assert!(starts.iter().all(|&s| &reference[s..s + len] == subseq));
            use std::collections::HashSet;
            // Check left-maximality.
            if starts.iter().all(|&s| s != 0) {
                let starts: HashSet<_> = starts.iter().map(|&s| reference[s - 1]).collect();
                assert!(starts.len() > 1);
            }
            // Check right-maximality.
            if starts.iter().all(|&s| s + len < reference.len()) {
                let ends: HashSet<_> = starts.iter().map(|&s| reference[s + len]).collect();
                assert!(ends.len() > 1);
            }
        }
    }
}

#[test]
fn random_check_bv() {
    let mut rng: Xoroshiro128StarStar = SeedableRng::seed_from_u64(12908320);
    for _ in 0..2 {
        let bitvec: Vec<_> = (0..100000).map(|_| rng.gen()).collect();
        let bv = suffix_utils::bitvector::BitVec::new(&bitvec);
        for (idx, &b) in bitvec.iter().enumerate() {
            assert_eq!(bv.get(idx), b);
        }
        // Rank check
        for idx in 0..bitvec.len() {
            // True query.
            let rank = bitvec[..idx].iter().filter(|&&b| b).count();
            let rank_bv = bv.rank(true, idx);
            assert_eq!(rank, rank_bv, "{}\t{}\t{}", idx, rank, rank_bv);
            // False query
            let rank = bitvec[..idx].iter().filter(|&&b| !b).count();
            let rank_bv = bv.rank(false, idx);
            assert_eq!(rank, rank_bv, "{}\t{}\t{}", idx, rank, rank_bv);
        }
        // Check select query.
        let number_of_true = bitvec.iter().filter(|&&b| b).count();
        let number_of_false = bitvec.len() - number_of_true;
        for i in 1..number_of_true {
            let mut acc = 0;
            let mut pos = 0;
            while acc < i {
                acc += bitvec[pos] as usize;
                pos += 1;
            }
            let pos_bv = bv.select(true, i);
            assert_eq!(pos_bv, pos - 1, "{}\t{}\t{}", i, pos_bv, pos - 1);
        }
        for i in 1..number_of_false {
            let mut acc = 0;
            let mut pos = 0;
            while acc < i {
                acc += !bitvec[pos] as usize;
                pos += 1;
            }
            let pos_bv = bv.select(false, i);
            assert_eq!(pos_bv, pos - 1, "{}\t{}\t{}", i, pos_bv, pos - 1);
        }
    }
}
