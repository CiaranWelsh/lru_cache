use std::cell::RefCell;
use std::fmt;
use std::fmt::Debug;

use std::rc::Rc;
use crate::lru_node::LruNode;

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
#[derive(Clone, Eq)]
pub struct LRULinkedList<T = u32>
    where
        T: PartialEq + Eq,
{
    head: Rc<RefCell<LruNode<T>>>,
    rear: Rc<RefCell<LruNode<T>>>,
    size: usize,
}

impl<T> LRULinkedList<T>
    where
        T: PartialEq + Eq,
{
    /// Creates a new empty LRULinkedList with sentinel head and rear nodes.
    pub fn new() -> Self {
        // Create the sentinel head and rear nodes.
        // Since these are sentinel nodes, the actual data is not important, hence we can use `None`.
        let head = Rc::new(RefCell::new(LruNode::new(u32::MAX - 1, None)));
        let rear = Rc::new(RefCell::new(LruNode::new(u32::MAX, None)));

        // Link the head to the rear and the rear to the head
        head.borrow_mut().next = Some(Rc::clone(&rear));
        rear.borrow_mut().prev = Some(Rc::clone(&head));

        // Initialize the list with a size of 0.
        Self {
            head,
            rear,
            size: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0usize
    }


    /// Adds a new node with the given `id` and `data` to the head of the list.
    /// Increments the size of the list and returns a reference to the new node.
    pub fn add_node_to_head(&mut self, id: u32, data: T) -> Rc<RefCell<LruNode<T>>> {
        let new_node = LruNode::<T>::new_rc_refcell(id, Some(data));
        self.size += 1;
        self.move_node_to_head(new_node.clone());
        new_node
    }

    /// Moves an existing node to the head of the list.
    /// The node is identified by the given `Rc<RefCell<LRUNode<T>>>` reference.
    pub fn move_node_to_head(&mut self, node: Rc<RefCell<LruNode<T>>>) {
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
        // Check if the node is already the head
        if Rc::ptr_eq(&self.head.borrow().next.as_ref().unwrap(), &node) {
            return;
        }

        let mut node_borrow = node.borrow_mut();

        // Disconnect the node from its current position
        if let Some(prev) = node_borrow.prev.take() {
            let mut prev_borrow = prev.borrow_mut();
            if let Some(next) = node_borrow.next.clone() {
                prev_borrow.next = Some(next.clone());
                let mut next_borrow = next.borrow_mut();
                next_borrow.prev = Some(prev.clone());
            } else {
                // If there's no next, then the node was the last one before rear, so update rear's prev
                prev_borrow.next = Some(self.rear.clone());
                self.rear.borrow_mut().prev = Some(prev.clone());
            }
        }

        // Move the node to the head
        let mut head_borrow = self.head.borrow_mut();
        let first_node = head_borrow.next.replace(node.clone());
        node_borrow.next = first_node.clone(); // Set node's next to the experiments first node
        node_borrow.prev = Some(self.head.clone()); // Set node's prev to the head

        // Update the experiments first node's prev to the moved node
        if let Some(first_node) = first_node {
            let mut first_node_borrow = first_node.borrow_mut();
            first_node_borrow.prev = Some(node.clone());
        }
    }


    /// Removes the node at the rear of the list, decrementing the list size.
    /// Note: It doesn't remove the sentinel rear node.
    pub fn pop_back(&mut self) -> Option<Rc<RefCell<LruNode<T>>>> {
        // trace!("head: {:?}", self.head);
        // trace!("rear: {:?}", self.rear);
        let rear_node = self.rear.borrow().prev.clone()?; // Get the last node if it exists
        // trace!("rear node: {:?}", rear_node);

        {
            let mut rear_node_borrow = rear_node.borrow_mut();

            // Update the list's rear to point to the new last node
            if let Some(new_rear) = rear_node_borrow.prev.clone() {
                let mut new_rear_borrow = new_rear.borrow_mut();
                new_rear_borrow.next = Some(self.rear.clone());
                self.rear.borrow_mut().prev = Some(new_rear.clone());
            } else {
                // If there is no new rear, it means the list is now empty
                self.rear.borrow_mut().prev = None;
            }

            // Disconnect the popped node from the list
            rear_node_borrow.prev = None;
            rear_node_borrow.next = None;
        }
        self.size = self.size.saturating_sub(1);

        Some(rear_node) // Return the popped node
    }

    /// Returns a reference to the node at the rear of the list (excluding the sentinel node),
    /// or `None` if the list is empty.
    pub fn get_rear(&self) -> Option<Rc<RefCell<LruNode<T>>>> {
        let rear_borrow = self.rear.borrow();
        rear_borrow.prev.clone()
    }
    pub fn get_head(&self) -> Option<Rc<RefCell<LruNode<T>>>> {
        let head_borrow = self.head.borrow();
        head_borrow.next.clone()
    }

    /// Returns the current size of the list (number of nodes excluding the sentinels).
    pub fn len(&self) -> usize {
        self.size
    }

    /// Creates an iterator over the LRULinkedList.
    ///
    /// The iterator starts from the node next to the head and ends at the node
    /// before the rear, thereby excluding the sentinel nodes.
    pub fn iter(&self) -> LRULinkedListIter<T> {
        LRULinkedListIter::<T>::new(&self)
    }

    /// Creates a mutable iterator over the LRULinkedList.
    ///
    /// The iterator starts from the node next to the head and ends at the node
    /// before the rear, thereby excluding the sentinel nodes.
    pub fn iter_mut(&self) -> LRULinkedListIterMut<T> {
        LRULinkedListIterMut::<T>::new(&self)
    }
}

pub struct LRULinkedListIter<T = u32>
    where
        T: PartialEq + Eq,
{
    current: Option<Rc<RefCell<LruNode<T>>>>,
    rear: Rc<RefCell<LruNode<T>>>,
}

/// Iterator for LRULinkedList. This iterator starts from the node
/// next to the head and ends at the node before the rear, thereby
/// excluding the sentinel nodes.
impl<T> LRULinkedListIter<T>
    where
        T: PartialEq + Eq,
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
    pub fn new(linked_list: &LRULinkedList<T>) -> Self {
        Self {
            current: linked_list.head.borrow().next.clone(),
            rear: linked_list.rear.clone(),
        }
    }
}

impl<T> Iterator for LRULinkedListIter<T>
    where
        T: PartialEq + Eq,
{
    type Item = Rc<RefCell<LruNode<T>>>;

    /// Returns the next node in the LRULinkedList, or `None` if the iterator
    /// reaches the rear node.
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(current) = &self.current {
            if Rc::ptr_eq(current, &self.rear) {
                return None;
            }
        }

        self.current.take().map(|current_node| {
            self.current = current_node.borrow().next.clone();
            current_node
        })
    }
}

pub struct LRULinkedListIterMut<T = u32>
    where
        T: PartialEq + Eq,
{
    current: Option<Rc<RefCell<LruNode<T>>>>,
    rear: Rc<RefCell<LruNode<T>>>,
}

impl<T> LRULinkedListIterMut<T>
    where
        T: PartialEq + Eq,
{
    pub fn new(linked_list: &LRULinkedList<T>) -> Self {
        Self {
            current: linked_list.head.borrow().next.clone(),
            rear: linked_list.rear.clone(),
        }
    }
}

impl<T> Iterator for LRULinkedListIterMut<T>
    where
        T: PartialEq + Eq,
{
    type Item = Rc<RefCell<LruNode<T>>>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(current) = &self.current {
            if Rc::ptr_eq(current, &self.rear) {
                return None;
            }
        }

        self.current.take().map(|current_node| {
            self.current = current_node.borrow().next.clone();
            current_node
        })
    }
}

impl<T> Debug for LRULinkedList<T>
    where
        T: PartialEq + Eq + Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LRULinkedList {{ ")?;
        write!(f, "<head> ")?;
        write!(f, "size: {}, ", self.size)?;

        // Print the head sentinel node without the RefCell.
        // let head_borrowed = self.head.borrow();
        // write!(f, "LRUNode {{ id: {}, data: {} }}, ", head_borrowed.id, head_borrowed.data)?;

        // Print the nodes in between the head and rear.
        write!(f, "nodes: [")?;
        for node in self.iter() {
            let node_borrowed = node.borrow();
            write!(f, "LRUNode {{ id: {}, data: {:?} }}, ", node_borrowed.id, node_borrowed.data)?;
        }
        write!(f, "], ")?;

        // Print the rear sentinel node without the RefCell.
        let _rear_borrowed = self.rear.borrow();
        write!(f, "<rear> ")?;
        // write!(f, "rear: LRUNode {{ id: {}, data: {} }}, ", rear_borrowed.id, rear_borrowed.data)?;

        // Print the size.

        write!(f, "}}")
    }
}

impl<T> PartialEq for LRULinkedList<T>
    where
        T: PartialEq + Eq,
{
    fn eq(&self, other: &Self) -> bool {
        // Compare sizes first for a quick exit
        if self.size != other.size {
            return false;
        }

        for (mine, yours) in self.iter().zip(other.iter()) {
            if *mine.borrow() != *yours.borrow() {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use log::LevelFilter::Trace;
    use log::trace;
    use crate::init_logger::init_logger;
    use super::*;

    #[test]
    fn test_add_node_to_head() {
        let mut list = LRULinkedList::new();
        let node = list.add_node_to_head(1u32, 0u32);
        let node_borrow = node.borrow();

        assert_eq!(node_borrow.id.clone(), 1u32);
        assert_eq!(node_borrow.data.unwrap(), 0);

        let head_borrow = list.head.borrow();
        let head_next_opt = head_borrow.next.as_ref().unwrap();
        let head_next = head_next_opt.borrow();
        assert_eq!(head_next.id.clone(), 1);

        let tail_borrow = list.rear.borrow();
        let tail_prev_opt = tail_borrow.prev.as_ref().unwrap();
        assert_eq!(tail_prev_opt.as_ref().borrow().id.clone(), 1);
    }

    #[test]
    fn test_move_node_to_head() {
        let mut list = LRULinkedList::new();
        let node1 = list.add_node_to_head(1u32, 0u32);
        let _node2 = list.add_node_to_head(2u32, 1u32);

        // access the head node and check its value
        assert_eq!(list.head.borrow().next.as_ref().unwrap().borrow().id.clone(), 2);
        assert_eq!(list.head.borrow().next.as_ref().unwrap().borrow().data.unwrap(), 1);

        list.move_node_to_head(node1.clone());

        assert_eq!(list.head.borrow().next.as_ref().unwrap().borrow().id.clone(), 1);
        assert_eq!(list.head.borrow().next.as_ref().unwrap().borrow().data.unwrap(), 0);
    }

    #[test]
    fn test_remove_rear_node() {
        let mut list = LRULinkedList::new();
        list.add_node_to_head(1u32, 0u32);
        list.add_node_to_head(2u32, 1u32);

        // access the rear node and check its prev value
        assert_eq!(list.len(), 2);
        assert_eq!(list.rear.borrow().prev.as_ref().unwrap().borrow().id.clone(), 1);
        assert_eq!(list.rear.borrow().prev.as_ref().unwrap().borrow().data.unwrap(), 0);

        list.pop_back();
        assert_eq!(list.len(), 1);
        assert_eq!(list.rear.borrow().prev.as_ref().unwrap().borrow().id.clone(), 2);
        assert_eq!(list.rear.borrow().prev.as_ref().unwrap().borrow().data.unwrap(), 1);
    }

    #[test]
    fn test_linked_list_iterator() {
        let mut list = LRULinkedList::new();
        list.add_node_to_head(1, 0);
        list.add_node_to_head(2, 1);
        list.add_node_to_head(3, 2);

        let iter = list.iter(); // Assuming you've implemented an `iter()` method for LRULinkedList
        let ids: Vec<u32> = iter.map(|node| node.borrow().id.clone()).collect();

        assert_eq!(ids, vec![3, 2, 1]);
    }

    #[test]
    fn test_linked_list_iterator_mut() {
        let mut list = LRULinkedList::new();
        list.add_node_to_head(1, 0);
        list.add_node_to_head(2, 1);
        list.add_node_to_head(3, 2);

        let mut iter = list.iter_mut(); // Assuming you've implemented an `iter_mut()` method for LRULinkedList
        let mut ids: Vec<u32> = Vec::new();

        while let Some(node) = iter.next() {
            let mut node_borrow = node.borrow_mut();
            ids.push(node_borrow.id.clone());
            node_borrow.id += 1;
        }

        assert_eq!(ids, vec![3, 2, 1]);

        // Verify that the node ids were actually modified
        let iter_verify = list.iter();
        let modified_ids: Vec<u32> = iter_verify.map(|node| node.borrow().id.clone()).collect();
        assert_eq!(modified_ids, vec![4, 3, 2]);
    }


    #[test]
    fn test_empty_linked_list_equality() {
        let list1 = LRULinkedList::<u32>::new();
        let list2 = LRULinkedList::<u32>::new();

        assert_eq!(list1, list2);
    }

    #[test]
    fn test_linked_list_equality_same_elements() {
        let mut list1 = LRULinkedList::new();
        let mut list2 = LRULinkedList::new();

        list1.add_node_to_head(1, 0);
        list1.add_node_to_head(2, 1);

        list2.add_node_to_head(1, 0);
        list2.add_node_to_head(2, 1);

        assert_eq!(list1, list2);
    }

    #[test]
    fn test_linked_list_inequality_different_elements() {
        let mut list1 = LRULinkedList::new();
        let mut list2 = LRULinkedList::new();

        list1.add_node_to_head(1, 0);
        list1.add_node_to_head(2, 1);

        list2.add_node_to_head(1, 0);
        list2.add_node_to_head(3, 2);

        assert_ne!(list1, list2);
    }

    #[test]
    fn test_linked_list_inequality_non_empty_and_empty() {
        let mut list1 = LRULinkedList::new();
        let list2 = LRULinkedList::new();

        list1.add_node_to_head(1, 0);

        assert_ne!(list1, list2);
    }


    #[test]
    fn test_pointers_after_adding_single_node() {
        init_logger(Trace);
        let mut list = LRULinkedList::new();
        let _node = list.add_node_to_head(1, 10);

        // Test pointers of the head and rear sentinel nodes
        let head_next = list.head.borrow().next.as_ref().unwrap().clone();
        let rear_prev = list.rear.borrow().prev.as_ref().unwrap().clone();

        let head = list.head.borrow();
        let rear = list.rear.borrow();

        trace!("head.id {:?}", head.id);
        trace!("head.data {:?}", head.data);
        trace!("head.prev {:?}", head.prev);
        trace!("head.next {:?}", head.next);
        trace!("rear.id {:?}", rear.id);
        trace!("rear.data {:?}", rear.data);
        trace!("rear.prev {:?}", rear.prev);
        trace!("rear.next {:?}", rear.next);

        assert!(Rc::ptr_eq(&head_next, &rear_prev), "Head next should point to the same node as rear prev after adding a single node.");
    }

    #[test]
    fn test_pointers_after_adding_multiple_nodes() {
        init_logger(Trace);
        let mut list = LRULinkedList::new();
        let _node1 = list.add_node_to_head(1, 10);
        let _node2 = list.add_node_to_head(2, 20);

        // Bind the results of `borrow` to variables
        let head_next_node = list.head.borrow().next.as_ref().unwrap().clone();
        let head_next = head_next_node.borrow();
        let second_node_node = head_next.next.as_ref().unwrap().clone();
        let second_node = second_node_node.borrow();

        assert_eq!(head_next.id, 2, "Head next should be the most recently added node.");
        assert_eq!(second_node.id, 1, "Second node should be the one added first.");

        let head = list.head.borrow();
        let rear = list.rear.borrow();

        trace!("head.id {:?}", head.id);
        trace!("head.data {:?}", head.data);
        trace!("head.prev {:?}", head.prev);
        trace!("head.next {:?}", head.next);
        trace!("rear.id {:?}", rear.id);
        trace!("rear.data {:?}", rear.data);
        trace!("rear.prev {:?}", rear.prev);
        trace!("rear.next {:?}", rear.next);


        assert!(Rc::ptr_eq(&head_next_node, &second_node.prev.as_ref().unwrap()), "Nodes should point to each other correctly.");
    }

    #[test]
    fn test_pointers_after_moving_node_to_head1() {
        init_logger(Trace);
        let mut list = LRULinkedList::new();
        println!("Initial list: {:?}", list);

        let node1 = list.add_node_to_head(1, 10);
        println!("After adding node1: {:?}", list);

        let _node2 = list.add_node_to_head(2, 20);
        println!("After adding node2: {:?}", list);

        let _node3 = list.add_node_to_head(3, 30);
        println!("After adding node3: {:?}", list);

        list.move_node_to_head(node1.clone());
        println!("After moving node1 to head: {:?}", list);

        // // Move the last node to the head and test pointers
        // // [1, 3, 2]
        // list.move_node_to_head(node1.clone());

        // Bind the results of `borrow` to variables
        let new_head_next_node = list.head.borrow().next.as_ref().expect("No next found").clone();
        let new_head_next_node_borrowed = new_head_next_node.borrow();
        let new_second_node = new_head_next_node_borrowed.next.as_ref().expect("No next found").clone();
        let new_second_node_borrowed = new_second_node.borrow();

        trace!("second node id: {:?}", new_second_node_borrowed.id);
        trace!("second node data: {:?}", new_second_node_borrowed.data);
        trace!("second node next: {:?}", new_second_node_borrowed.next);
        trace!("second node prev: {:?}", new_second_node_borrowed.prev);

        let rear_node_borrowed = list.rear.borrow();


        trace!("rear_node_borrowed id: {:?}",   rear_node_borrowed.id);
        trace!("rear_node_borrowed data: {:?}", rear_node_borrowed.data);
        trace!("rear_node_borrowed next: {:?}", rear_node_borrowed.next);
        trace!("rear_node_borrowed prev: {:?}", rear_node_borrowed.prev);


        let last_node = list.rear.borrow().prev.as_ref().expect("No prev found").clone();
        let last_node_borrowed = last_node.borrow();

        assert_eq!(new_head_next_node_borrowed.id, 1, "After moving, first node should be the moved node.");
        assert_eq!(new_second_node_borrowed.id, 3, "Second node should now be the previously first node.");
        assert_eq!(last_node_borrowed.id, 2, "Last node should be the one that was second before the move.");
        assert!(Rc::ptr_eq(&new_head_next_node, &new_second_node_borrowed.prev.as_ref().unwrap()), "Moved node should correctly point to the next node.");
        assert!(Rc::ptr_eq(&new_second_node, &last_node_borrowed.prev.as_ref().unwrap()), "Second node should correctly point to the last node.");
    }

    #[test]
    fn test_pointers_after_moving_node_to_head() {
        let mut list = LRULinkedList::new();
        let node1 = list.add_node_to_head(1, 10);
        let _node2 = list.add_node_to_head(2, 20);
        let _node3 = list.add_node_to_head(3, 30);

        // Move the last node to the head and test pointers
        list.move_node_to_head(node1.clone());

        // Bind the results of `borrow` to variables
        let new_head_next_node = list.head.borrow().next.as_ref().unwrap().clone();
        let new_head_next = new_head_next_node.borrow();
        let new_second_node_node = new_head_next.next.as_ref().unwrap().clone();
        let new_second_node = new_second_node_node.borrow();
        let last_node_node = list.rear.borrow().prev.as_ref().unwrap().clone();
        let last_node = last_node_node.borrow();

        assert_eq!(new_head_next.id, 1, "After moving, first node should be the moved node.");
        assert_eq!(new_second_node.id, 3, "Second node should now be the previously first node.");
        assert_eq!(last_node.id, 2, "Last node should be the one that was second.");
        assert!(Rc::ptr_eq(&new_head_next_node, &new_second_node.prev.as_ref().unwrap()), "Moved node should correctly point to the next node.");
        assert!(Rc::ptr_eq(&new_second_node_node, &last_node.prev.as_ref().unwrap()), "Second node should correctly point to the last node.");
    }

    #[test]
    fn test_pointers_after_popping_back() {
        let mut list = LRULinkedList::new();
        list.add_node_to_head(1, 10);
        list.add_node_to_head(2, 20);
        list.add_node_to_head(3, 30);

        // Pop back and test pointers
        let _popped_node = list.pop_back();

        // Bind the results of `borrow` to variables
        let last_node_node = list.rear.borrow().prev.as_ref().unwrap().clone();
        let last_node = last_node_node.borrow();
        let second_to_last_node_node = last_node.prev.as_ref().unwrap().clone();
        let _second_to_last_node = second_to_last_node_node.borrow();

        assert_eq!(last_node.id, 2, "After popping, last node should be the one before the popped node.");
        assert!(Rc::ptr_eq(&second_to_last_node_node, &last_node.prev.as_ref().unwrap()), "Second to last node should correctly point to the last node.");
    }
}
