#[macro_use]
extern crate serde;
extern crate num;
pub mod suffix_array;
pub mod suffix_tree;
#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
