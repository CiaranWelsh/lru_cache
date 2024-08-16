use std::fmt;

use std::sync::Arc;
use parking_lot::Mutex;



/// The LRUNode struct represents a node in the LRU cache.
/// Each node has an identifier `id`, a position `pos`, and links to
/// the previous and next nodes in the LRU list.
#[derive(Clone)]
pub struct ThreadSafeLruNode<V = u32>
    where
        V: PartialEq + Eq,
{
    pub id: u32,
    pub data: Option<V>,
    pub prev: Option<Arc<Mutex<ThreadSafeLruNode<V>>>>,
    pub next: Option<Arc<Mutex<ThreadSafeLruNode<V>>>>,
}


impl<V> ThreadSafeLruNode<V>
    where
        V: PartialEq + Eq,
{
    pub fn new(id: u32, data: Option<V>) -> Self {
        Self {
            id,
            data,
            prev: None,
            next: None,
        }
    }

    pub fn new_arc_mutex(id: u32, data: Option<V>) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self::new(id, data)))
    }
}

impl<V> fmt::Debug for ThreadSafeLruNode<V>
    where
        V: PartialEq + Eq {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LRUNode")
            .field("id", &self.id)
            // .field("data", &self.data)
            // You can optionally include more fields here, but avoid
            // including 'prev' and 'next' to prevt circular references
            .finish()
    }
}

impl<V> PartialEq for ThreadSafeLruNode<V>
    where
        V: PartialEq + Eq
{
    fn eq(&self, other: &Self) -> bool {
        if self.id != other.id || self.data != other.data {
            return false;
        }
        // Define a helper function for comparing Option<Rc<RefCell<LRUNode>>>
        // Define `compare_optional_nodes` as a closure inside `eq`
        let compare_optional_nodes = |a: &Option<Arc<Mutex<ThreadSafeLruNode<V>>>>,
                                      b: &Option<Arc<Mutex<ThreadSafeLruNode<V>>>>| {
            match (a, b) {
                (Some(a_arc_mutex), Some(b_arc_mutex)) => {
                    // Safely borrow the contents of the Rc<RefCell<LRUNode>>.
                    let a_clone = a_arc_mutex.clone();
                    let b_clone = b_arc_mutex.clone();
                    let alocked = a_clone.lock();
                    let blocked = b_clone.lock();
                    // Compare the id and data of the borrowed LRUNode.
                    alocked.id == blocked.id && alocked.data == blocked.data
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
        let node = ThreadSafeLruNode::new(1u32, Some(0u32));

        assert_eq!(node.id, 1);
        assert_eq!(node.data.unwrap(), 0);
        assert!(node.prev.is_none());
        assert!(node.next.is_none());
    }



    #[test]
    fn test_lru_node_equality_no_links() {
        let node1 = ThreadSafeLruNode::new_arc_mutex(1, Some(0));
        let node1_clone = node1.clone();
        let node1_locked = node1_clone.lock();
        let node2 = ThreadSafeLruNode::new_arc_mutex(1, Some(0));
        let node2_clone = node2.clone();
        let node2_locked = node2_clone.lock();

        assert_eq!(*node1_locked, *node2_locked);
    }

    #[test]
    fn test_lru_node_inequality_different_ids_or_pos() {
        let node1 = ThreadSafeLruNode::new_arc_mutex(1, Some(0));
        let node1_clone = node1.clone();
        let node1_locked = node1_clone.lock();

        let node2 = ThreadSafeLruNode::new_arc_mutex(2, Some(0));
        let node2_clone = node2.clone();
        let node2_locked = node2_clone.lock();

        let node3 = ThreadSafeLruNode::new_arc_mutex(1, Some(1));
        let node3_clone = node3.clone();
        let node3_locked = node3_clone.lock();

        assert_ne!(*node1_locked, *node2_locked);
        assert_ne!(*node1_locked, *node3_locked);
    }

    #[test]
    fn test_lru_node_with_string() {
        let node = ThreadSafeLruNode::new(1u32, Some("test".to_string()));

        // Access the fields directly
        assert_eq!(node.id, 1);
        assert_eq!(node.data.as_ref().unwrap(), "test");
    }
}
