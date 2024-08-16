use std::collections::HashMap;

use std::rc::Rc;
use std::cell::RefCell;
use std::fmt;
use std::fmt::Debug;
use crate::lru_linked_list::LRULinkedList;
use crate::lru_node::LruNode;

#[derive(Clone)]
pub struct LruCache<V>
    where
        V: PartialEq + Eq + Clone + Debug + Default,
{
    list: LRULinkedList<V>,
    map: HashMap<u32, Rc<RefCell<LruNode<V>>>>,
    capacity: usize,
}

impl<V> LruCache<V>
    where
        V: PartialEq + Eq + Clone + Debug + Default
{
    pub fn new(capacity: usize) -> Self {
        LruCache {
            list: LRULinkedList::new(),
            map: HashMap::with_capacity(capacity),
            capacity,
        }
    }

    pub fn get_clone(&mut self, key: &u32) -> Option<V> {
        if let Some(node) = self.map.get(key) {
            let val = node.borrow().data.clone().unwrap();
            // trace!("Val : {:?}", val);
            // Move the node to the front of the list to mark it as recently used
            // trace!("list before mode: {:?}", self.list);
            self.list.move_node_to_head(node.clone());
            // trace!("list after mode: {:?}", self.list);
            Some(val)
        } else {
            None
        }
    }

    pub fn get_mut<F>(&mut self, key: &u32, mut f: F) -> Option<()>
        where
            F: FnMut(&mut V),
    {
        if let Some(node) = self.map.get(key) {
            // Move the node to the front of the list to mark it as recently used
            self.list.move_node_to_head(node.clone());

            // Execute the closure with a mutable reference to the node's data
            let mut node_borrow = node.borrow_mut();
            if let Some(ref mut data) = node_borrow.data {
                f(data);
            }

            Some(())
        } else {
            None
        }
    }


    pub fn put(&mut self, key: u32, value: V) -> Option<V> {
        let mut evicted_value: Option<V> = None;

        if let Some(node) = self.map.get(&key) {
            // If the key exists, update the value and move the node to the front
            node.borrow_mut().data = Some(value);
            self.list.move_node_to_head(Rc::clone(node));
        } else {
            // Check if the cache is at capacity
            if self.map.len() == self.capacity {
                // Evict the least recently used item
                // trace!("LRU list: {:?}", self.list);
                if let Some(lru) = self.list.pop_back() {
                    // trace!("lru is: {:?}", lru);
                    let lru_id = lru.borrow().id;
                    if let Some(node) = self.map.remove(&lru_id) {
                        evicted_value = node.borrow_mut().data.take();
                    }
                }
            }
            // Insert the new node at the front of the list and in the map
            let new_node = self.list.add_node_to_head(key, value);
            self.map.insert(key, new_node);
        }
        evicted_value
    }

    /// Removes and returns the least recently used item from the cache.
    pub fn pop(&mut self) -> Option<(u32, V)> {
        // Attempt to pop the last node from the list
        if let Some(lru_node_rc) = self.list.pop_back() {
            let mut lru_node = lru_node_rc.borrow_mut();

            // Remove the node from the map
            let evicted_key = lru_node.id;
            self.map.remove(&evicted_key);

            // Take the data out of the node to return it
            let evicted_value = lru_node.data.take();

            // Return the key and value if we have a value
            if let Some(value) = evicted_value {
                Some((evicted_key, value))
            } else {
                None
            }
        } else {
            // If the list is empty, return None
            None
        }
    }

    pub fn get_processed_data(&mut self, key: u32) -> V {
        if let Some(node) = self.map.get(&key) {
            let mut node_borrow = node.borrow_mut();
            if let Some(ref mut data) = node_borrow.data {
                return std::mem::take(data);  // This empties the VecDeque in the cache and returns it
            }
        }
        // Return an empty V if there's no data for the key
        V::default()
    }
}

impl<V> Debug for LruCache<V>
    where
        V: Debug + PartialEq + Eq + Clone + Debug + Default,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LRUCache")
            .field("capacity", &self.capacity)
            .field("size", &self.map.len())
            .field("items", &self.list) // Assuming LRULinkedList also implements Debug
            .finish()
    }
}


#[cfg(test)]
mod tests {
    use log::LevelFilter::Trace;
    use log::trace;
    use crate::init_logger::init_logger;
    use super::*;

    #[test]
    fn test_new_cache() {
        let cache: LruCache<String> = LruCache::new(2);
        assert_eq!(cache.capacity, 2);
        assert!(cache.map.is_empty());
        assert_eq!(0, cache.list.len());
    }

    #[test]
    fn test_put_and_get() {
        let mut cache = LruCache::new(2);
        assert_eq!(cache.get_clone(&1), None);
        assert!(cache.list.is_empty());
        assert!(cache.map.is_empty());

        cache.put(1, "one".to_string());
        assert_eq!(cache.get_clone(&1), Some("one".to_string()));
        assert_eq!(1, cache.list.len());
        assert_eq!(1, cache.map.len());

        cache.put(2, "two".to_string());
        assert_eq!(cache.get_clone(&2), Some("two".to_string()));
        assert_eq!(2, cache.list.len());
        assert_eq!(2, cache.map.len());
    }

    #[test]
    fn checl_cache_size_5() {
        init_logger(Trace);
        let mut cache = LruCache::new(5);
        cache.put(6, "six".to_string());
        cache.put(5, "five".to_string());
        cache.put(4, "four".to_string());
        cache.put(3, "three".to_string());
        cache.put(2, "two".to_string());

        trace!("cache: {:?}", cache);
        let popped = cache.put(1, "one".to_string());
        assert_eq!(popped.unwrap(), "six");
    }

    #[test]
    fn checl_cache_size_2() {
        init_logger(Trace);
        let mut cache = LruCache::new(2);
        cache.put(3, "three".to_string());
        cache.put(2, "two".to_string());

        trace!("cache: {:?}", cache);
        let popped = cache.put(1, "one".to_string());
        assert_eq!(popped.unwrap(), "three");
    }

    #[test]
    fn check_cache_size_2_after_get() {
        init_logger(Trace);
        let mut cache = LruCache::new(2);
        cache.put(3, "three".to_string());
        cache.put(2, "two".to_string());
        trace!("cache before get 3: {:?}", cache);
        cache.get_clone(&3);
        trace!("cache after get 3: {:?}", cache);

        let popped = cache.put(1, "one".to_string());
        assert_eq!(popped.unwrap(), "two");
    }

    #[test]
    fn test_put_eviction() {
        init_logger(Trace);
        let mut cache = LruCache::new(2);
        // order: 1
        trace!("cache: {:?}", cache);
        cache.put(1, "one".to_string());
        trace!("cache: {:?}", cache);
        // order: 2, 1
        cache.put(2, "two".to_string());
        trace!("cache: {:?}", cache);
        // order: 1, 2
        assert_eq!(cache.get_clone(&1), Some("one".to_string()));
        trace!("cache: {:?}", cache);

        // now we put 3, it takes the place of 1 and 2 is popped
        let popped = cache.put(3, "three".to_string());
        assert!(popped.is_some());
        if let Some(number_2) = popped {
            assert_eq!(number_2, "two".to_string());
        }
        // assert_eq!(cache.get(&1), Some("one".to_string()));
        // assert_eq!(cache.get(&2), None);
        // assert_eq!(cache.get(&3), Some("three".to_string()));

        // cache.put(4, "four".to_string());
        // assert_eq!(cache.get(&1), None);
        // assert_eq!(cache.get(&3), Some("three".to_string()));
        // assert_eq!(cache.get(&4), Some("four".to_string()));
    }

    #[test]
    fn test_lru_eviction_order() {
        let mut cache = LruCache::new(2);
        cache.put(1, "one".to_string());
        // [1]
        cache.put(2, "two".to_string());
        // [2, 1]
        assert_eq!(cache.get_clone(&1), Some("one".to_string())); // This should make key 1 the most recently used
        // [1, 2]

        let two_popped = cache.put(3, "three".to_string()); // This should evict key 2
        // [3, 1] ==> 2
        assert_eq!(two_popped.unwrap(), "two");
        assert_eq!(cache.get_clone(&2), None);
        assert_eq!(cache.get_clone(&1), Some("one".to_string()));
        // [1, 3]
        assert_eq!(cache.get_clone(&3), Some("three".to_string()));
        // [3, 1]

        let one_popped = cache.put(2, "two".to_string()); // This should evict key 1
        // [2, 3] => 1
        assert_eq!(one_popped.unwrap(), "one");
        assert_eq!(cache.get_clone(&1), None);
        // [2, 3]

        assert_eq!(cache.get_clone(&2), Some("two".to_string()));
        assert_eq!(cache.get_clone(&3), Some("three".to_string()));
    }


    #[test]
    fn test_get_mut_updates_value() {
        let mut cache = LruCache::new(2);
        cache.put(1, 10);
        assert_eq!(cache.get_clone(&1), Some(10));

        cache.get_mut(&1, |value| {
            *value += 5;
        });

        assert_eq!(cache.get_clone(&1), Some(15));
    }

    #[test]
    fn test_get_mut_nonexistent_key() {
        let mut cache = LruCache::new(2);
        cache.put(1, 10);

        assert!(cache.get_mut(&2, |_| ()).is_none());
    }

    #[test]
    fn test_pop_removes_least_recently_used() {
        let mut cache = LruCache::new(2);
        cache.put(1, "one".to_string());
        cache.put(2, "two".to_string());
        cache.get_clone(&1); // This will mark key 1 as recently used

        let evicted = cache.pop();
        assert_eq!(evicted, Some((2, "two".to_string()))); // Key 2 should be evicted since it's the LRU
    }

    #[test]
    fn test_pop_empty_cache() {
        let mut cache: LruCache<String> = LruCache::new(2);
        assert!(cache.pop().is_none());
    }

    #[test]
    fn test_put_and_pop_sequence() {
        let mut cache = LruCache::new(2);
        cache.put(1, "one".to_string());
        cache.put(2, "two".to_string());

        assert_eq!(cache.pop(), Some((1, "one".to_string()))); // Pops key 1 since it's the LRU
        assert_eq!(cache.get_clone(&2), Some("two".to_string()));

        cache.put(3, "three".to_string()); // Puts key 3 and evicts key 2 as it's now the LRU
        assert_eq!(cache.pop(), Some((2, "two".to_string())));
        assert_eq!(cache.get_clone(&3), Some("three".to_string()));
    }
}
