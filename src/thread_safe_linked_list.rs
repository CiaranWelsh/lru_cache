use std::fmt;
use std::fmt::Debug;

use std::sync::Arc;
use parking_lot::Mutex;
use crate::thread_safe_lru_node::ThreadSafeLruNode;

/// # LRUCache
///
/// The `LRUCache` struct represents a Least Recently Used (LRU) cache.
///
/// ## Overview
///
/// The LRU Cache has a maximum capacity (`cap`) that defines how many items it can hold.
/// Once the capacity is reached, the least recently used item is evicted to make room for new items.
/// The current number of items in the cache is stored in `size`.
///
/// The cache is backed by two data structures:
/// - An `LRULinkedList` (`node_list`), which is used to maintain the LRU order of items.
/// - A `HashMap` (`node_hash`), which allows for quick look-up of items using their keys.
///
/// ## Fields
///
/// - `cap: u32` - The maximum number of items the cache can hold.
/// - `size: u32` - The current number of items in the cache.
/// - `node_list: LRULinkedList` - The linked list that maintains the LRU order of items.
/// - `node_hash: HashMap<u32, Rc<RefCell<LRUNode>>>` - The hash map for quick look-up of items.
///
/// ## Note
///
/// The `LRUCache` is implemented using `Rc` and `RefCell` for shared ownership and mutability of `LRUNode`.
/// This allows the linked list and hash map to share references to the same nodes.
///
#[derive(Clone)]
pub struct ThreadSafeLruLinkedList<V = u32>
    where
        V: PartialEq + Eq,
{
    head: Arc<Mutex<ThreadSafeLruNode<V>>>,
    tail: Arc<Mutex<ThreadSafeLruNode<V>>>,
    size: usize,
}

impl<V> ThreadSafeLruLinkedList<V>
    where
        V: PartialEq + Eq,
{
    /// Creates a new empty LRULinkedList with sentinel head and rear nodes.
    pub fn new() -> Self {
        let head = Arc::new(Mutex::new(ThreadSafeLruNode::new(u32::MAX - 1, None)));
        let rear = Arc::new(Mutex::new(ThreadSafeLruNode::new(u32::MAX, None)));

        // Link the head to the rear and the rear to the head
        {
            let mut head_lock = head.lock();
            let mut rear_lock = rear.lock();

            head_lock.next = Some(rear.clone());
            rear_lock.prev = Some(head.clone());
        }

        Self { head, tail: rear, size: 0 }
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0usize
    }

    /// get the most recently used node without modifying order
    pub fn get_most_recently_used_node(&self) -> Arc<Mutex<ThreadSafeLruNode<V>>> {
        // Lock the head to access its 'next' node
        let head_lock = self.head.lock();

        // Clone the Arc from the 'next' field in the head node, which points to the first actual node in the list
        let first_node_arc = head_lock.next.as_ref().expect("No first node").clone();

        first_node_arc
    }

    /// get the most recently used node without modifying order
    pub fn get_least_recently_used_node(&self) -> Arc<Mutex<ThreadSafeLruNode<V>>> {
        let tail_lock = self.tail.lock();
        let last_node_arc = tail_lock.prev.as_ref().expect("No last node").clone();
        last_node_arc
    }


    /// Adds a new node with the given `id` and `data` to the head of the list.
    /// Increments the size of the list and returns a reference to the new node.
    pub fn add_node_to_head(&mut self, id: u32, data: V) -> Arc<Mutex<ThreadSafeLruNode<V>>> {
        let new_node = ThreadSafeLruNode::<V>::new_arc_mutex(id, Some(data));
        self.size += 1;
        self.move_node_to_head(new_node.clone());
        new_node
    }

    /// add node to the rear of the linked list. 
    /// 
    /// Important: Using this method breaks the invariant of the LRUCache. This operation is
    /// permitted to allow the thread safe LRU override the ordering of eviction in situations 
    /// where eviction is rejected on grounds of ordering. Do not use this unless you have a clear 
    /// idea of what you are doing. 
    pub fn add_node_to_rear(&mut self, id: u32, data: V) -> Arc<Mutex<ThreadSafeLruNode<V>>> {
        let new_node = ThreadSafeLruNode::<V>::new_arc_mutex(id, Some(data));
        self.size += 1;
        self.move_node_to_rear(new_node.clone());
        new_node
    }

    /// Moves an existing node to the head of the list.
    /// The node is identified by the given `Rc<RefCell<LRUNode<V>>>` reference.
    pub fn move_node_to_head(&mut self, node: Arc<Mutex<ThreadSafeLruNode<V>>>) {
        /*
             head <-> node1 <-> node2 <-> ... <-> nodeX <-> nodeY <-> ... <-> rear
         step 1. remove from current position
         before removal:
             prevNode <-> nodeX <-> nextNode
         after removal
             prevNode <-> nextNode
         now
             nodeX (floating without pointers)
         we need to insert into the head position
         before insertion:
             head <-> firstNode
         after insertion:
             head <-> nodeX <-> firstNode
         */
        // Lock the node to check if it is already the head
        let is_head = {
            let next_head = self.head.lock().next.as_ref().unwrap().clone();
            Arc::ptr_eq(&next_head, &node)
        };
        if is_head {
            // The node is already the head, nothing to do.
            return;
        }

        // Lock the node and disconnect it from its current position

        let mut node_lock = node.lock();
        let prev_node_clone = node_lock.prev.clone(); // Clone the Arc for later use
        let prev_node_opt = node_lock.prev.take();
        let next_node_opt = node_lock.next.take();

        // If the node had a previous node, update the previous node's next pointer
        if let Some(prev_node) = prev_node_opt {
            let mut prev_node_lock = prev_node.lock();
            prev_node_lock.next = next_node_opt.clone();
        }

        // If the node had a next node, update the next node's prev pointer
        if let Some(next_node) = next_node_opt {
            let mut next_node_lock = next_node.lock();
            next_node_lock.prev = prev_node_clone; // Use the cloned Arc here
        }

        // Now, insert the node at the head of the list
        let mut head_lock = self.head.lock();
        let old_first_node = head_lock.next.replace(node.clone());

        node_lock.prev = Some(self.head.clone());
        node_lock.next = old_first_node.clone();

        if let Some(old_first_node) = old_first_node {
            let mut old_first_node_lock = old_first_node.lock();
            old_first_node_lock.prev = Some(node.clone());
        }
    }

    /// Moves an existing node to the rear of the list, just before the tail sentinel.
    pub fn move_node_to_rear(&mut self, node: Arc<Mutex<ThreadSafeLruNode<V>>>) {
        let is_tail = {
            let prev_tail = self.tail.lock().prev.as_ref().unwrap().clone();
            Arc::ptr_eq(&prev_tail, &node)
        };

        // If the node is already the last node (just before the tail), there's nothing to do.
        if is_tail {
            return;
        }

        // Disconnect the node from its current position
        let mut node_lock = node.lock();
        let next_node = node_lock.next.take();
        let prev_node = node_lock.prev.take();

        // If the node had a previous node, update the previous node's next pointer
        if let Some(prev_node_arc) = prev_node.clone() {
            let mut prev_node_lock = prev_node_arc.lock();
            prev_node_lock.next = next_node.clone();
        }

        // If the node had a next node, update the next node's prev pointer
        if let Some(next_node_arc) = next_node {
            let mut next_node_lock = next_node_arc.lock();
            next_node_lock.prev = prev_node;
        }

        // Now, insert the node just before the tail
        let mut tail_lock = self.tail.lock();
        let old_last_node = tail_lock.prev.replace(node.clone());

        node_lock.prev = old_last_node.clone();
        node_lock.next = Some(self.tail.clone());

        if let Some(old_last_node_arc) = old_last_node {
            let mut old_last_node_lock = old_last_node_arc.lock();
            old_last_node_lock.next = Some(node.clone());
        }
    }


    /// Removes the node at the rear of the list, decrementing the list size.
    /// Note: It doesn't remove the sentinel rear node.
    pub fn pop_back(&mut self) -> Option<Arc<Mutex<ThreadSafeLruNode<V>>>> {
        let mut rear_lock = self.tail.lock();
        let prev_to_rear_opt = rear_lock.prev.take(); // Take the prev to avoid double lock

        let prev_to_rear_arc = match prev_to_rear_opt {
            Some(arc) => arc,
            None => return None, // If there's no prev, it's empty or only has sentinel nodes
        };

        // If the previous node is the head, there's nothing to remove
        if Arc::ptr_eq(&prev_to_rear_arc, &self.head) {
            rear_lock.prev = Some(prev_to_rear_arc); // Restore the taken prev
            return None;
        }

        // Lock the previous to rear node and get the one before it
        let prev_to_prev_node;
        {
            let mut prev_to_rear_lock = prev_to_rear_arc.lock();
            prev_to_prev_node = prev_to_rear_lock.prev.take(); // Disconnect the node to be popped

            // No need to update next for prev_to_rear since it's going away
        }

        // Set the rear's prev pointer to the new last node
        rear_lock.prev = prev_to_prev_node.clone();

        // Update the next pointer of the node before the last to point to the rear
        if let Some(prev_to_prev_arc) = prev_to_prev_node {
            let mut prev_to_prev_lock = prev_to_prev_arc.lock();
            prev_to_prev_lock.next = Some(self.tail.clone());
        }

        // Decrement the size of the list
        self.size = self.size.saturating_sub(1);

        Some(prev_to_rear_arc)
    }


    /// Gets a reference to the node at the rear of the list (excluding the sentinel node).
    pub fn get_tail(&self) -> Option<Arc<Mutex<ThreadSafeLruNode<V>>>> {
        // Lock the rear's mutex to access its previous node
        let rear_lock = self.tail.lock();

        // Clone the Arc containing the previous node so we can return it
        // without holding on to the lock
        rear_lock.prev.clone()
    }

    /// Gets a reference to the node at the head of the list (excluding the sentinel node).
    pub fn get_head(&self) -> Option<Arc<Mutex<ThreadSafeLruNode<V>>>> {
        // Lock the head's mutex to access its next node
        let head_lock = self.head.lock();

        // Clone the Arc containing the next node so we can return it
        // without holding on to the lock
        head_lock.next.clone()
    }

    /// Returns the current size of the list (number of nodes excluding the sentinels).
    pub fn len(&self) -> usize {
        self.size
    }

    /// Creates an iterator over the LRULinkedList.
    ///
    /// The iterator starts from the node next to the head and ends at the node
    /// before the rear, thereby excluding the sentinel nodes.
    pub fn iter(&self) -> LRULinkedListIter<V> {
        LRULinkedListIter::<V>::new(&self)
    }

    /// Creates a mutable iterator over the LRULinkedList.
    ///
    /// The iterator starts from the node next to the head and ends at the node
    /// before the rear, thereby excluding the sentinel nodes.
    pub fn iter_mut(&self) -> LRULinkedListIterMut<V> {
        LRULinkedListIterMut::<V>::new(&self)
    }

    /// Peeks at the node at the specified position in the list without modifying the list order.
    ///
    /// # Parameters
    ///
    /// * `position` - The zero-based position of the node to peek at.
    ///
    /// # Returns
    ///
    /// * `Option<Arc<Mutex<ThreadSafeLruNode<V>>>>` - An `Option` containing an `Arc` pointing to the `Mutex` guarding the node at the specified position, or `None` if the position is out of bounds.
    pub fn peek_position(&self, position: usize) -> Option<Arc<Mutex<ThreadSafeLruNode<V>>>> {
        // Use the iterator to traverse up to the specified position.
        // The `nth` method on an iterator returns the nth item, consuming the iterator up to that point.
        self.iter().nth(position)
    }
}

pub struct LRULinkedListIter<V = u32>
    where
        V: PartialEq + Eq,
{
    current: Option<Arc<Mutex<ThreadSafeLruNode<V>>>>,
    rear: Arc<Mutex<ThreadSafeLruNode<V>>>,
}

/// Iterator for LRULinkedList. This iterator starts from the node
/// next to the head and ends at the node before the rear, thereby
/// excluding the sentinel nodes.
impl<V> LRULinkedListIter<V>
    where
        V: PartialEq + Eq,
{
    /// Creates a new iterator for a given `LRULinkedList`.
    ///
    /// # Arguments
    ///
    /// * `linked_list` - A reference to the `LRULinkedList` to iterate over.
    ///
    /// # Returns
    ///
    /// A new `LRULinkedListIter` instance.
    pub fn new(linked_list: &ThreadSafeLruLinkedList<V>) -> Self {
        let head_next = linked_list.head.lock().next.clone();
        Self {
            current: head_next,
            rear: linked_list.tail.clone(),
        }
    }
}

impl<V> Iterator for LRULinkedListIter<V>
    where
        V: PartialEq + Eq,
{
    type Item = Arc<Mutex<ThreadSafeLruNode<V>>>;

    /// Returns the next node in the LRULinkedList, or `None` if the iterator
    /// reaches the rear node.
    fn next(&mut self) -> Option<Self::Item> {
        let current = self.current.take(); // Take the current value out of the option

        current.and_then(|current_arc| {
            // Check if the current node is the rear sentinel node
            if Arc::ptr_eq(&current_arc, &self.rear) {
                return None;
            }

            // Lock the current node to get the next node
            let next_node = current_arc.lock().next.clone();

            // Set the iterator's current node to the next node
            self.current = next_node; // Now we can mutate self.current because current_arc is no longer borrowed

            // Return the current node that we took out
            Some(current_arc)
        })
    }
    
}

pub struct LRULinkedListIterMut<V = u32>
    where
        V: PartialEq + Eq,
{
    current: Option<Arc<Mutex<ThreadSafeLruNode<V>>>>,
    rear: Arc<Mutex<ThreadSafeLruNode<V>>>,
}

impl<V> LRULinkedListIterMut<V>
    where
        V: PartialEq + Eq,
{
    pub fn new(linked_list: &ThreadSafeLruLinkedList<V>) -> Self {
        let head_next = linked_list.head.lock().next.clone();
        Self {
            current: head_next,
            rear: linked_list.tail.clone(),
        }
    }
}

impl<V> Iterator for LRULinkedListIterMut<V>
    where
        V: PartialEq + Eq,
{
    type Item = Arc<Mutex<ThreadSafeLruNode<V>>>;

    fn next(&mut self) -> Option<Self::Item> {
        // Take the current value out of the option, replacing it with None
        let current = self.current.take();

        current.and_then(|current_arc| {
            if Arc::ptr_eq(&current_arc, &self.rear) {
                // If the current node is the rear, we've reached the end and return None.
                return None;
            }

            // Lock the current node to get the next node
            let next_node = current_arc.lock().next.clone();

            // Update the iterator's current node to the next node
            self.current = next_node; // No borrow checker issues here since current is not borrowed

            // Return the current node
            Some(current_arc)
        })
    }
}

impl<V> Debug for ThreadSafeLruLinkedList<V>
    where
        V: PartialEq + Eq + Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ParLRULinkedList {{ size: {}, nodes: [", self.size)?;

        let mut current = self.head.lock().next.clone();

        // Iterate through the nodes until we reach the rear sentinel node
        while let Some(node) = current {
            // Check if the node is the rear to stop printing
            if Arc::ptr_eq(&node, &self.tail) {
                break;
            }

            // Lock the node to access its contents
            let node_lock = node.lock();

            // Print the current node
            write!(f, "ParLruNode {{ id: {}, data: {:?} }}, ", node_lock.id, node_lock.data)?;

            // Move to the next node
            current = node_lock.next.clone();
        }

        write!(f, "] }}")
    }
}

impl<V> PartialEq for ThreadSafeLruLinkedList<V>
    where
        V: PartialEq + Eq,
{
    fn eq(&self, other: &Self) -> bool {
        // If the lists have different sizes, they cannot be equal.
        if self.size != other.size {
            return false;
        }

        // Start at the head of both lists.
        let mut my_current = self.head.lock().next.clone();
        let mut their_current = other.head.lock().next.clone();

        while let (Some(my_node_arc), Some(their_node_arc)) = (my_current.take(), their_current.take()) {
            // If either current node is the rear, then we've reached the end of both lists
            if Arc::ptr_eq(&my_node_arc, &self.tail) || Arc::ptr_eq(&their_node_arc, &other.tail) {
                break;
            }

            // Lock both nodes to compare their contents.
            let my_node_lock = my_node_arc.lock();
            let their_node_lock = their_node_arc.lock();

            // If the data in the nodes is not equal, the lists are not equal.
            if my_node_lock.data != their_node_lock.data {
                return false;
            }

            // Move to the next nodes in the list.
            my_current = my_node_lock.next.clone();
            their_current = their_node_lock.next.clone();
        }

        // If we've reached this point, the lists are equal.
        true
    }
}


impl<V> Eq for ThreadSafeLruLinkedList<V> where V: Eq {}


#[cfg(test)]
mod tests {
    use log::LevelFilter::Trace;
    use log::trace;
    use crate::init_logger::init_logger;
    use super::*;

    #[test]
    fn test_add_node_to_head() {
        init_logger(Trace);
        let mut list = ThreadSafeLruLinkedList::new();
        let node = list.add_node_to_head(1, 0);

        trace!("1");

        // Lock the node to access its data and immediately drop the lock
        {
            let node_lock = node.lock();
            assert_eq!(node_lock.id, 1);
            assert_eq!(node_lock.data, Some(0));
        }
        trace!("2");

        // Lock the head to access the next node and immediately drop the lock
        let head_next_arc = {
            let head_lock = list.head.lock();
            head_lock.next.as_ref().unwrap().clone()
        };

        trace!("3");
        // Lock the next node to access its data and immediately drop the lock
        {
            let head_next_lock = head_next_arc.lock();
            assert_eq!(head_next_lock.id, 1);
        }

        trace!("4");
        // Lock the rear to access the previous node and immediately drop the lock
        let tail_prev_arc = {
            let rear_lock = list.tail.lock();
            rear_lock.prev.as_ref().unwrap().clone()
        };

        trace!("5");
        // Lock the previous node to access its data and immediately drop the lock
        {
            let tail_prev_lock = tail_prev_arc.lock();
            assert_eq!(tail_prev_lock.id, 1);
        }
        trace!("6");
    }


    #[test]
    fn test_move_node_to_head() {
        let mut list = ThreadSafeLruLinkedList::new();
        let node1 = list.add_node_to_head(1, 0);
        let _node2 = list.add_node_to_head(2, 1);

        // Access the head node and check its value
        {
            let head_lock = list.head.lock();
            let head_next_temp = head_lock.next.as_ref().unwrap().clone();
            let head_next_lock = head_next_temp.lock();
            assert_eq!(head_next_lock.id, 2);
            assert_eq!(head_next_lock.data, Some(1));
        } // Locks are dropped here

        // Now move the first node to the head
        list.move_node_to_head(node1);

        // Access the head node again and check its new value
        {
            let head_lock = list.head.lock();
            let new_head_next_temp = head_lock.next.as_ref().unwrap().clone();
            let new_head_next_lock = new_head_next_temp.lock();
            assert_eq!(new_head_next_lock.id, 1);
            assert_eq!(new_head_next_lock.data, Some(0));
        } // Locks are dropped here
    }


    #[test]
    fn test_remove_rear_node() {
        let mut list = ThreadSafeLruLinkedList::new(); // Assuming ParLRULinkedList is the mutex version
        list.add_node_to_head(1, 0);
        list.add_node_to_head(2, 1);

        // Access the rear node and check its prev value
        assert_eq!(list.len(), 2);
        {
            let rear_lock = list.tail.lock();
            let rear_prev_node = rear_lock.prev.as_ref().unwrap().clone();
            let rear_prev_lock = rear_prev_node.lock();
            assert_eq!(rear_prev_lock.id, 1);
            assert_eq!(rear_prev_lock.data, Some(0));
        } // Locks are dropped here

        // Pop the rear node and check the list's state
        list.pop_back();
        assert_eq!(list.len(), 1);
        {
            let rear_lock = list.tail.lock();
            let new_rear_prev_node = rear_lock.prev.as_ref().unwrap().clone();
            let new_rear_prev_lock = new_rear_prev_node.lock();
            assert_eq!(new_rear_prev_lock.id, 2);
            assert_eq!(new_rear_prev_lock.data, Some(1));
        } // Locks are dropped here
    }

    #[test]
    fn test_linked_list_iterator() {
        let mut list = ThreadSafeLruLinkedList::new();
        list.add_node_to_head(1, 0);
        list.add_node_to_head(2, 1);
        list.add_node_to_head(3, 2);

        // Note: The iter() method should return an iterator that locks the nodes before yielding them.
        let mut ids: Vec<u32> = Vec::new();
        let mut current = list.head.lock().next.clone();
        while let Some(node) = current {
            if Arc::ptr_eq(&node, &list.tail) {
                break;
            }
            let node_lock = node.lock();
            ids.push(node_lock.id);
            current = node_lock.next.clone();
        }

        assert_eq!(ids, vec![3, 2, 1]);
    }


    #[test]
    fn test_linked_list_iterator_mut() {
        let mut list = ThreadSafeLruLinkedList::new();
        list.add_node_to_head(1, 0);
        list.add_node_to_head(2, 1);
        list.add_node_to_head(3, 2);

        // We cannot have a direct mutable iterator, but we can simulate the iteration.
        let mut ids: Vec<u32> = Vec::new();
        let mut current = list.head.lock().next.clone();
        while let Some(node) = current {
            if Arc::ptr_eq(&node, &list.tail) {
                break;
            }
            let mut node_lock = node.lock();
            ids.push(node_lock.id);
            node_lock.id += 1;  // Increment the id
            current = node_lock.next.clone();
        }

        assert_eq!(ids, vec![3, 2, 1]);

        // Re-iterate to verify that the node ids were actually modified
        let iter_verify = list.iter(); // Assuming iter() is properly implemented
        let modified_ids: Vec<u32> = iter_verify.map(|node| node.lock().id).collect();
        assert_eq!(modified_ids, vec![4, 3, 2]);
    }

    #[test]
    fn test_empty_linked_list_equality() {
        let list1 = ThreadSafeLruLinkedList::<u32>::new();
        let list2 = ThreadSafeLruLinkedList::<u32>::new();

        // The equality check should be defined to handle thread safety.
        assert_eq!(list1, list2);
    }

    #[test]
    fn test_linked_list_equality_same_elements() {
        let mut list1 = ThreadSafeLruLinkedList::new();
        let mut list2 = ThreadSafeLruLinkedList::new();

        list1.add_node_to_head(1, 0);
        list1.add_node_to_head(2, 1);

        list2.add_node_to_head(1, 0);
        list2.add_node_to_head(2, 1);

        // The equality check should be defined to handle thread safety.
        assert_eq!(list1, list2);
    }

    #[test]
    fn test_linked_list_inequality_different_elements() {
        let mut list1 = ThreadSafeLruLinkedList::new();
        let mut list2 = ThreadSafeLruLinkedList::new();

        list1.add_node_to_head(1, 0);
        list1.add_node_to_head(2, 1);

        list2.add_node_to_head(1, 0);
        list2.add_node_to_head(3, 2);

        assert_ne!(list1, list2);
    }

    #[test]
    fn test_linked_list_inequality_non_empty_and_empty() {
        let mut list1 = ThreadSafeLruLinkedList::new();
        let list2 = ThreadSafeLruLinkedList::new();

        list1.add_node_to_head(1, 0);

        assert_ne!(list1, list2);
    }


    #[test]
    fn test_pointers_after_adding_single_node() {
        // The `init_logger` function is not provided in the given context.
        // Assuming it's a function that initializes a logger.
        // init_logger(Trace);
        let mut list = ThreadSafeLruLinkedList::new();
        let _node = list.add_node_to_head(1, 10);

        // Lock the head and rear to access their next and prev
        let head_next = list.head.lock().next.as_ref().unwrap().clone();
        let rear_prev = list.tail.lock().prev.as_ref().unwrap().clone();

        // Test that the next pointer of the head and the prev pointer of the rear
        // both point to the same node (the one we just added).
        assert!(Arc::ptr_eq(&head_next, &rear_prev), "Head next should point to the same node as rear prev after adding a single node.");

        // For logging purposes, lock the head and rear again
        let head_lock = list.head.lock();
        let rear_lock = list.tail.lock();

        trace!("head.id {:?}", head_lock.id);
        trace!("head.data {:?}", head_lock.data);
        trace!("head.prev {:?}", head_lock.prev);
        trace!("head.next {:?}", head_lock.next);
        trace!("rear.id {:?}", rear_lock.id);
        trace!("rear.data {:?}", rear_lock.data);
        trace!("rear.prev {:?}", rear_lock.prev);
        trace!("rear.next {:?}", rear_lock.next);
    }

    #[test]
    fn test_pointers_after_adding_multiple_nodes() {
        // init_logger(Trace); // Initialize the logger, assuming it is set up.
        let mut list = ThreadSafeLruLinkedList::new();
        let _node1 = list.add_node_to_head(1, 10);
        let _node2 = list.add_node_to_head(2, 20);

        // Lock the head to access the first node
        let head_next_arc = list.head.lock().next.as_ref().unwrap().clone();
        let head_next_lock = head_next_arc.lock();

        // Check that the first node after head is the most recently added node
        assert_eq!(head_next_lock.id, 2, "Head next should be the most recently added node.");

        // Access the second node
        let second_node_arc = head_next_lock.next.as_ref().unwrap().clone();
        let second_node_lock = second_node_arc.lock();

        // Check that the second node is the one added first
        assert_eq!(second_node_lock.id, 1, "Second node should be the one added first.");

        // Log the head and rear node details
        let _head_lock = list.head.lock();
        let _rear_lock = list.tail.lock();

        // Using trace! would require a logger to be initialized.
        // trace!("head.id {:?}", head_lock.id);
        // trace!("head.data {:?}", head_lock.data);
        // trace!("head.prev {:?}", head_lock.prev);
        // trace!("head.next {:?}", head_lock.next);
        // trace!("rear.id {:?}", rear_lock.id);
        // trace!("rear.data {:?}", rear_lock.data);
        // trace!("rear.prev {:?}", rear_lock.prev);
        // trace!("rear.next {:?}", rear_lock.next);

        // Check that the nodes point to each other correctly
        assert!(Arc::ptr_eq(&head_next_arc, &second_node_lock.prev.as_ref().unwrap()), "Nodes should point to each other correctly.");
    }

    #[test]
    fn test_pointers_after_moving_node_to_head() {
        // init_logger(Trace);
        let mut list = ThreadSafeLruLinkedList::new();
        let node1 = list.add_node_to_head(1, 10);
        let _node2 = list.add_node_to_head(2, 20);
        let _node3 = list.add_node_to_head(3, 30);

        // Move the last node to the head and test pointers
        list.move_node_to_head(node1);

        // Lock the head to get the next node after moving
        let new_head_next_arc = list.head.lock().next.as_ref().unwrap().clone();
        let new_head_next_lock = new_head_next_arc.lock();
        // Lock the second node
        let new_second_node_arc = new_head_next_lock.next.as_ref().unwrap().clone();
        let new_second_node_lock = new_second_node_arc.lock();
        // Lock the last node
        let last_node_arc = list.tail.lock().prev.as_ref().unwrap().clone();
        let last_node_lock = last_node_arc.lock();

        // Check the id of the new head, second, and last node
        assert_eq!(new_head_next_lock.id, 1, "After moving, first node should be the moved node.");
        assert_eq!(new_second_node_lock.id, 3, "Second node should now be the previously first node.");
        assert_eq!(last_node_lock.id, 2, "Last node should be the one that was second.");

        // Log the details of the list structure
        // trace!("head.id {:?}", list.head.lock().id);
        // trace!("head.next {:?}", list.head.lock().next);
        // trace!("rear.prev {:?}", list.rear.lock().prev);
        // trace!("rear.id {:?}", list.rear.lock().id);

        // Check that the pointers are correctly set after moving the node
        assert!(Arc::ptr_eq(&new_head_next_arc, &new_second_node_lock.prev.as_ref().unwrap()), "Moved node should correctly point to the next node.");
        assert!(Arc::ptr_eq(&new_second_node_arc, &last_node_lock.prev.as_ref().unwrap()), "Second node should correctly point to the last node.");
    }

    #[test]
    fn test_pointers_after_popping_back() {
        let mut list = ThreadSafeLruLinkedList::new(); // Assuming ParLRULinkedList is the mutex version
        list.add_node_to_head(1, 10);
        list.add_node_to_head(2, 20);
        list.add_node_to_head(3, 30);

        // Pop back and test pointers
        list.pop_back();

        // Get the last node and check its ID
        let (last_node_id, last_node_next) = {
            let rear_lock = list.tail.lock();
            let last_node_arc = rear_lock.prev.as_ref().unwrap().clone();
            let last_node_lock = last_node_arc.lock();
            (last_node_lock.id, last_node_lock.next.clone())
        };

        assert_eq!(last_node_id, 2, "After popping, last node should be the one before the popped node.");

        // Check if the last node points to the rear
        let points_to_rear = last_node_next.is_some() && Arc::ptr_eq(&last_node_next.unwrap(), &list.tail);
        assert!(points_to_rear, "Last node should correctly point to the rear sentinel node.");
    }

    #[test]
    fn test_move_specific_node_to_rear() {
        init_logger(Trace);
        let mut list = ThreadSafeLruLinkedList::new();
        let _node1 = list.add_node_to_head(1, 10);
        let node2 = list.add_node_to_head(2, 20);
        let _node3 = list.add_node_to_head(3, 30);

        // Verify the initial rear of the list is node1 with values (1, 10)
        {
            let rear_prev = list.get_least_recently_used_node();
            let rear_prev_lock = rear_prev.lock();
            assert_eq!(rear_prev_lock.id, 1, "Initially, the rear node should be node1 with ID 1.");
            assert_eq!(rear_prev_lock.data, Some(10), "The data of the rear node should be 10.");
        }

        // Move node2 to the rear
        list.move_node_to_rear(node2);

        // Verify that node2 is now the rear node and the integrity of the list
        {
            let rear_prev = list.get_least_recently_used_node();
            let rear_prev_lock = rear_prev.lock();
            assert_eq!(rear_prev_lock.id, 2, "After moving, the rear node should be node2 with ID 2.");
            assert_eq!(rear_prev_lock.data, Some(20), "The data of the rear node should be 20.");

            // Ensure node3 is now the first node
            let head_next = list.get_most_recently_used_node();
            let head_next_lock = head_next.lock();
            assert_eq!(head_next_lock.id, 3, "The first node should now be node3 with ID 3.");

            // Ensure the previous node to node2 is node1
            let node2_prev = rear_prev_lock.prev.as_ref().unwrap().clone();
            let node2_prev_lock = node2_prev.lock();
            assert_eq!(node2_prev_lock.id, 1, "The previous node to node2 (now at rear) should be node1 with ID 1.");
        }

        // Log the current list structure for debugging purposes
        trace!("After moving node2 to the rear, the list order should be updated accordingly.");
    }

    #[test]
    fn test_peek_position() {
        init_logger(Trace);
        let mut ll = ThreadSafeLruLinkedList::<String>::new(); // Create a cache with some capacity.

        // Populate the cache with some test data.
        ll.add_node_to_head(1, "Data 1".to_string());
        ll.add_node_to_head(2, "Data 2".to_string());
        ll.add_node_to_head(3, "Data 3".to_string());

        trace!("cache: {:?}", ll);

        // Attempt to peek at different positions in the linked list.
        if let Some(node) = ll.peek_position(0) {
            let node_lock = node.lock();
            assert_eq!(node_lock.data.as_ref(), Some(&"Data 3".to_string()), "The first node should contain 'Data 3'");
        } else {
            panic!("Expected a node at position 0, but none was found.");
        }

        trace!("cache: {:?}", ll);
        
        fn check_position(ll: &ThreadSafeLruLinkedList<String>, position: usize, expected_data: &str){
            if let Some(node) = ll.peek_position(position) {
                let node_lock = node.lock();
                assert_eq!(node_lock.data.as_ref(), Some(&expected_data.to_string()), 
                           "Node at position {:?} should contain {:?} but contains {:?}", position, expected_data, node_lock.data.as_ref());
            } else {
                panic!("Expected a node at position 1, but none was found.");
            }
        }
        
        check_position(&ll, 0, "Data 3");
        check_position(&ll, 1, "Data 2");
        check_position(&ll, 2, "Data 1");
    }

}
