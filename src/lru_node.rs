use std::cell::RefCell;
use std::fmt;

use std::rc::Rc;


/// The LRUNode struct represents a node in the LRU cache.
/// Each node has an identifier `id`, a position `pos`, and links to
/// the previous and next nodes in the LRU list.
#[derive(Clone, Eq)]
pub struct LruNode<T = u32>
    where
        T: PartialEq + Eq,
{
    pub id: u32,
    pub data: Option<T>,
    pub prev: Option<Rc<RefCell<LruNode<T>>>>,
    pub next: Option<Rc<RefCell<LruNode<T>>>>,
}


impl<T> LruNode<T>
    where
        T: PartialEq + Eq,
{
    pub fn new(id: u32, data: Option<T>) -> Self {
        Self {
            id,
            data,
            prev: None,
            next: None,
        }
    }

    pub fn new_rc_refcell(id: u32, data: Option<T>) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self::new(id, data)))
    }
}

impl<T> fmt::Debug for LruNode<T>
    where
        T: PartialEq + Eq{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LRUNode")
            .field("id", &self.id)
            // .field("data", &self.data)
            // You can optionally include more fields here, but avoid
            // including 'prev' and 'next' to prevt circular references
            .finish()
    }
}

impl<T> PartialEq for LruNode<T>
    where
        T: PartialEq + Eq
{
    fn eq(&self, other: &Self) -> bool {
        if self.id != other.id || self.data != other.data {
            return false;
        }
        // Define a helper function for comparing Option<Rc<RefCell<LRUNode>>>
        // Define `compare_optional_nodes` as a closure inside `eq`
        let compare_optional_nodes = |a: &Option<Rc<RefCell<LruNode<T>>>>,
                                      b: &Option<Rc<RefCell<LruNode<T>>>>| {
            match (a, b) {
                (Some(a_rc), Some(b_rc)) => {
                    // Safely borrow the contents of the Rc<RefCell<LRUNode>>.
                    let a_borrowed = a_rc.borrow();
                    let b_borrowed = b_rc.borrow();
                    // Compare the id and data of the borrowed LRUNode.
                    a_borrowed.id == b_borrowed.id && a_borrowed.data == b_borrowed.data
                }
                (None, None) => true,
                _ => false,
            }
        };
        // Use the helper function to compare prev and next fields
        compare_optional_nodes(&self.prev, &other.prev) &&
            compare_optional_nodes(&self.next, &other.next)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_node_initialization() {
        let node = LruNode::new_rc_refcell(1u32, Some(0u32));
        let node_borrow = node.borrow();

        assert_eq!(node_borrow.id, 1);
        assert_eq!(node_borrow.data.unwrap(), 0);
        assert!(node_borrow.prev.is_none());
        assert!(node_borrow.next.is_none());
    }

    #[test]
    fn test_lru_node_linking() {
        let node1 = LruNode::new_rc_refcell(1u32, Some(0u32));
        let node2 = LruNode::new_rc_refcell(2u32, Some(1u32));

        // Link node1 -> node2
        node1.borrow_mut().next = Some(node2.clone());
        node2.borrow_mut().prev = Some(node1.clone());

        // Check forward link
        {
            let node1_borrowed = node1.borrow();
            let node1_next = &node1_borrowed.next.as_ref().unwrap().borrow();
            assert_eq!(node1_next.id, 2);
            assert_eq!(node1_next.data.unwrap(), 1);
        }

        // Check backward link
        {
            let node2_borrowed = node2.borrow();
            let node2_prev = &node2_borrowed.prev.as_ref().unwrap().borrow();
            assert_eq!(node2_prev.id, 1);
            assert_eq!(node2_prev.data.unwrap(), 0);
        }
    }


    #[test]
    fn test_lru_node_equality_no_links() {
        let node1 = LruNode::new_rc_refcell(1, Some(0)).borrow().clone();
        let node2 = LruNode::new_rc_refcell(1, Some(0)).borrow().clone();

        assert_eq!(node1, node2);
    }

    #[test]
    fn test_lru_node_inequality_different_ids_or_pos() {
        let node1 = LruNode::new_rc_refcell(1, Some(0)).borrow().clone();
        let node2 = LruNode::new_rc_refcell(2, Some(0)).borrow().clone();
        let node3 = LruNode::new_rc_refcell(1, Some(1)).borrow().clone();

        assert_ne!(node1, node2);
        assert_ne!(node1, node3);
    }

    #[test]
    fn test_lru_node_equality_with_links() {
        let node1 = LruNode::new_rc_refcell(1, Some(0));
        let node2 = LruNode::new_rc_refcell(1, Some(0));
        let next_node = LruNode::new_rc_refcell(2, Some(1));

        node1.borrow_mut().next = Some(next_node.clone());
        node2.borrow_mut().next = Some(next_node.clone());

        let node1_borrowed = node1.borrow().clone();
        let node2_borrowed = node2.borrow().clone();

        assert_eq!(node1_borrowed, node2_borrowed);
    }

    #[test]
    fn test_lru_node_with_string() {
        let node = LruNode::new_rc_refcell(1u32, Some("test".to_string()));
        let node_borrow = node.borrow();

        // Access the fields directly
        assert_eq!(node_borrow.id, 1);
        assert_eq!(node_borrow.data.as_ref().unwrap(), "test");
    }


}
