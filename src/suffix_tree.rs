//! Suffix Tree module.
//! # Overview
use super::suffix_array;
use super::suffix_array::SuffixArray;
use std::fmt::Debug;
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize, Default)]
pub struct SuffixTree<T: Ord + Clone + Eq + Debug> {
    pub root_idx: usize,
    pub nodes: Vec<Node>,
    resource_type: std::marker::PhantomData<T>,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize, Default)]
pub struct Node {
    pub parent: Option<usize>,
    pub label_length_to_parent: usize,
    // (node index, label length, the first character of the label)
    pub children: Vec<(usize, usize, u64)>,
    pub position_at_text: usize,
    // Leaf label. The position of the suffix.
    pub leaf_label: Option<usize>,
}

impl Node {
    pub fn new(idx: usize) -> Self {
        Self {
            parent: None,
            label_length_to_parent: 0,
            children: vec![],
            position_at_text: idx,
            leaf_label: None,
        }
    }
    pub fn set_leaf_label(&mut self, idx: usize) {
        self.leaf_label = Some(idx);
    }
    pub fn set_parent(&mut self, parent: usize, distance: usize) {
        self.parent = Some(parent);
        self.label_length_to_parent = distance;
    }
    pub fn push_child(&mut self, child: usize, len: usize, elm: u64) {
        let position = match self.children.binary_search_by_key(&elm, |x| x.2) {
            Ok(_) => panic!("{} was added to this node twice! {:?}", elm, self),
            Err(res) => res,
        };
        self.children.insert(position, (child, len, elm));
    }
    pub fn get_parent(&self) -> Option<(usize, usize)> {
        self.parent.map(|idx| (idx, self.label_length_to_parent))
    }
}

impl<T: Ord + Clone + Eq + Debug> SuffixTree<T> {
    pub fn new(input: &[T], alphabet: &[T]) -> Self {
        let suffix_array = SuffixArray::new(input, alphabet);
        let inverse_suffix_array = suffix_array.inverse();
        let lcp = suffix_array::longest_common_prefix(input, &suffix_array, &inverse_suffix_array);
        // Insert the root node, draw edge from the root to the first element in SA, as
        // It must be '$', the smallest suffix.
        let alphabet: Vec<(usize, T)> = alphabet.iter().cloned().enumerate().collect();
        let mut input: Vec<u64> = input
            .iter()
            .map(|x| {
                alphabet
                    .iter()
                    .filter(|c| &c.1 == x)
                    .map(|c| c.0 as u64 + 1)
                    .next()
                    .expect("the input contains character not in the alphabet.")
            })
            .collect();
        input.push(0);
        let mut nodes: Vec<_> = suffix_array
            .as_ref()
            .iter()
            .map(|&idx| {
                let mut n = Node::new(input.len());
                n.set_leaf_label(idx);
                n
            })
            .collect();
        let root_idx = input.len();
        let mut root = Node::new(0);
        root.push_child(0, 1, 0);
        nodes[0].set_parent(root_idx, 1);
        nodes.push(root);
        for i in 1..input.len() {
            let merge_len = lcp[i];
            let mut label_len = input.len() - suffix_array[i - 1];
            let mut parent = i - 1;
            // First, find the node v that the length of label l(v) is
            // less than or equal to the merge_len, i.e., |l(v)| <= merge_len.
            while label_len > merge_len {
                let (p, dist) = match nodes[parent].get_parent() {
                    Some(x) => x,
                    None => panic!("Parent:{},{:?}", parent, nodes[parent]),
                };
                parent = p;
                label_len -= dist;
            }
            // Next, merging. If label_len is the same as merge_len, it is easy.
            let position_at_text = suffix_array[i] + merge_len;
            let length_to_leaf = input.len() - position_at_text;
            if merge_len == label_len {
                nodes[parent].push_child(i, length_to_leaf, input[position_at_text]);
                nodes[i].set_parent(parent, length_to_leaf);
            } else {
                let mut new_node = Node::new(position_at_text);
                let (c_idx, l_len, ch) = nodes[parent].children.pop().unwrap();
                let new_branch_elm =
                    input[nodes[c_idx].position_at_text + merge_len - label_len - l_len];
                new_node.push_child(c_idx, l_len + label_len - merge_len, new_branch_elm);
                new_node.push_child(i, length_to_leaf, input[position_at_text]);
                new_node.set_parent(parent, merge_len - label_len);
                let new_node_idx = nodes.len();
                nodes[c_idx].set_parent(new_node_idx, l_len + label_len - merge_len);
                nodes[parent].push_child(new_node_idx, merge_len - label_len, ch);
                nodes[i].set_parent(new_node_idx, length_to_leaf);
                nodes.push(new_node);
            }
        }
        Self {
            root_idx,
            nodes,
            resource_type: std::marker::PhantomData,
        }
    }
    pub fn maximul_repeat(&self, input: &[T]) -> Vec<(Vec<usize>, usize)> {
        let mut depth = vec![0; self.nodes.len()];
        let mut leaves_under_subtree = vec![vec![]; self.nodes.len()];
        let mut prev = vec![None; self.nodes.len()];
        let mut stack = vec![self.root_idx];
        let mut arrived = vec![false; self.nodes.len()];
        let mut current_depth = 0;
        'dfs: while !stack.is_empty() {
            let node = *stack.last().unwrap();
            if !arrived[node] {
                arrived[node] = true;
            }
            for &(idx, label_len, _) in self.nodes[node].children.iter() {
                if !arrived[idx] {
                    stack.push(idx);
                    current_depth += label_len;
                    continue 'dfs;
                }
            }
            let node = stack.pop().unwrap();
            depth[node] = current_depth;
            leaves_under_subtree[node] = if let Some(idx) = self.nodes[node].leaf_label {
                vec![idx]
            } else {
                self.nodes[node]
                    .children
                    .iter()
                    .flat_map(|&(idx, _, _)| leaves_under_subtree[idx].iter())
                    .copied()
                    .collect()
            };
            prev[node] = match self.nodes[node].leaf_label {
                Some(idx) if idx == 0 => None,
                Some(idx) => Some(&input[idx - 1]),
                None => {
                    let children = &self.nodes[node].children;
                    let first = &prev[children[0].0];
                    if children.iter().any(|&(idx, _, _)| prev[idx].is_none())
                        || children.iter().any(|&(idx, _, _)| &prev[idx] != first)
                    {
                        None
                    } else {
                        *first
                    }
                }
            };
            current_depth -= self.nodes[node].label_length_to_parent;
        }
        prev.into_iter()
            .zip(leaves_under_subtree)
            .zip(depth)
            .filter(|&((prev, ref lv), _)| prev.is_none() && lv.len() > 1)
            .map(|((_, lv), depth)| (lv, depth))
            .collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn works() {
        let input = b"CAGTAGCTGACTGATCAGTC";
        let alphabet = b"ACGT";
        eprintln!("{}", String::from_utf8_lossy(input));
        let sa = SuffixArray::new(input, alphabet);
        for (rank, &idx) in sa.as_ref().iter().enumerate().skip(1) {
            eprintln!("{:2}\t{}", rank, String::from_utf8_lossy(&input[idx..]));
        }
        let st = SuffixTree::new(input, alphabet);
        let mut stack = vec![];
        stack.push(st.root_idx);
        let mut suffix = vec![];
        let mut arrived = vec![false; st.nodes.len()];
        let mut input = input.to_vec();
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
    #[test]
    fn enumerate_max() {
        let input = b"CAGTAGCTGACTGATCAGTC";
        let alphabet = b"ACGT";
        eprintln!("{}", String::from_utf8_lossy(input));
        let st = SuffixTree::new(input, alphabet);
        for (starts, len) in st.maximul_repeat(input) {
            // Check repetitiveness.
            let subseq = &input[starts[0]..starts[0] + len];
            assert!(starts.iter().all(|&s| &input[s..s + len] == subseq));
            use std::collections::HashSet;
            // Check left-maximality.
            if starts.iter().all(|&s| s != 0) {
                let starts: HashSet<_> = starts.iter().map(|&s| input[s - 1]).collect();
                assert!(starts.len() > 1);
            }
            // Check right-maximality.
            if starts.iter().all(|&s| s + len < input.len()) {
                let ends: HashSet<_> = starts.iter().map(|&s| input[s + len]).collect();
                assert!(ends.len() > 1);
            }
        }
    }
}
