use std::collections::{HashMap, VecDeque};

use std::fmt;
use std::fmt::Debug;
use std::sync::Arc;
use log::trace;
use parking_lot::Mutex;
use crate::thread_safe_linked_list::ThreadSafeLruLinkedList;
use crate::thread_safe_lru_node::ThreadSafeLruNode;

#[derive(Clone)]
pub struct ThreadSafeLruCache<V>
    where
        V: PartialEq + Eq + Clone + Debug + Default,
{
    list: ThreadSafeLruLinkedList<V>,
    map: HashMap<u32, Arc<Mutex<ThreadSafeLruNode<V>>>>,
    capacity: usize,
    last_evicted_id: Option<u32>,
}

impl<V> ThreadSafeLruCache<V>
    where
        V: PartialEq + Eq + Clone + Debug + Default
{
    /// Creates a new instance of a thread-safe LRU cache with the specified capacity.
    ///
    /// # Parameters
    ///
    /// * `capacity` - The maximum number of items the cache can hold.
    ///
    /// # Returns
    ///
    /// * `ThreadSafeLruCache<V>` - A new instance of a thread-safe LRU cache with the specified capacity.
    pub fn new(capacity: usize) -> Self {
        ThreadSafeLruCache {
            list: ThreadSafeLruLinkedList::new(),
            map: HashMap::with_capacity(capacity),
            capacity,
            last_evicted_id: None,
        }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns a value from the cache, cloning it if it exists.
    ///
    /// # Parameters
    ///
    /// * `key` - The key associated with the value to retrieve.
    ///
    /// # Returns
    ///
    /// * `Option<V>` - The value associated with the key, if it exists, or `None` if the key does not exist.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it may return a reference to data that is owned by the cache,
    /// which may be invalidated or deallocated at any time.
    pub fn get_clone(&mut self, key: u32) -> Option<V> {
        if let Some(node_ref) = self.map.get(&key) {
            // Lock the node to access its data
            let node_lock = node_ref.lock();
            let val = node_lock.data.clone();

            // Since we're going to move the node to the head of the list,
            // clone the Arc to pass along to the list management functions.
            let node_ref_cloned = Arc::clone(node_ref);

            // Drop the lock before moving the node to the head to avoid deadlocks
            drop(node_lock);

            // Move the node to the front of the list to mark it as recently used
            self.list.move_node_to_head(node_ref_cloned);

            val
        } else {
            None
        }
    }


    pub fn get_mut<F>(&mut self, key: u32, f: F) -> Option<()>
        where
            F: FnOnce(&mut V),
    {
        if let Some(node_ref) = self.map.get(&key) {
            // Clone the Arc to pass along to the list management functions
            let node_ref_cloned = Arc::clone(node_ref);
            // Move the node to the front of the list to mark it as recently used
            self.list.move_node_to_head(node_ref_cloned);

            // Lock the node to access its data
            let mut node_lock = node_ref.lock();

            // Execute the closure with a mutable reference to the node's data
            if let Some(ref mut data) = node_lock.data {
                f(data); // f can only be called once, since it's FnOnce
            }

            Some(())
        } else {
            None
        }
    }


    pub fn put(&mut self, key: u32, value: V) -> Option<(u32, V)> {
        let mut evicted_item: Option<(u32, V)> = None;

        if let Some(node_ref) = self.map.get(&key) {
            // If the key exists, update the value and move the node to the front
            let mut node_lock = node_ref.lock();
            node_lock.data = Some(value);
            drop(node_lock); // Drop the lock before moving the node
            self.list.move_node_to_head(Arc::clone(node_ref));
        } else {
            // Check if the cache is at capacity
            if self.map.len() == self.capacity {
                // Evict the least recently used item
                if let Some(lru_ref) = self.list.pop_back() {
                    let lru_lock = lru_ref.lock();
                    let lru_key = lru_lock.id;
                    drop(lru_lock); // Drop the lock before removing the node from the map
                    if let Some(evicted_node_ref) = self.map.remove(&lru_key) {
                        let mut evicted_node_lock = evicted_node_ref.lock();
                        if let Some(evicted_value) = evicted_node_lock.data.take() {
                            evicted_item = Some((lru_key, evicted_value));
                        }
                    }
                }
            }
            // Insert the new node at the front of the list and in the map
            self.list.add_node_to_head(key, value);
            let most_recently_used = self.list.get_most_recently_used_node();
            self.map.insert(key, most_recently_used);
            // trace!("cache capacity: {:?}, map len: {:?}", self.capacity, self.map.len());
        }
        evicted_item
    }

    /// in this version of `put` checking is done to ensure that lru nodes are always
    /// evicted in linear sequential ordering.
    pub fn put_with_sequential_eviction(&mut self, key: u32, value: V) -> Option<(u32, V)> {
        let mut evicted_item: Option<(u32, V)> = None;

        if let Some(node_ref) = self.map.get(&key) {
            // If the key exists, update the value and move the node to the front
            let mut node_lock = node_ref.lock();
            node_lock.data = Some(value);
            drop(node_lock); // Drop the lock before moving the node
            self.list.move_node_to_head(Arc::clone(node_ref));
        } else {
            // Check if the cache is at capacity
            if self.map.len() == self.capacity {
                // Evict the least recently used item
                if let Some(popped_lru_node) = self.list.pop_back() {
                    let lru_guard = popped_lru_node.lock();
                    let lru_key = lru_guard.id;
                    // We popped from the linked list, make sure to also remove from the map.
                    let evicted_node_opt = self.map.remove(&lru_key);
                    drop(lru_guard); // Drop the lock before removing the node from the map

                    // some validation to ensure the eviction process is sequential in the frame id

                    let next_expected_eviction_id = if let Some(last_evicted_id) = self.last_evicted_id {
                        last_evicted_id + 1
                    } else {
                        0
                    };

                    let is_sequential = lru_key == next_expected_eviction_id;

                    if is_sequential {
                        // the next node for eviction is the next key in order
                        // trace!("is seq: self.map: {:?}", self.map);
                        if let Some(evicted_node_ref) = evicted_node_opt {
                            let mut evicted_node_lock = evicted_node_ref.lock();
                            if let Some(evicted_data) = evicted_node_lock.data.take() {
                                evicted_item = Some((lru_key, evicted_data));
                            }
                            // update the last evicted id
                            if let Some(ref mut last_evicted_id) = self.last_evicted_id {
                                *last_evicted_id += 1;
                            } else {
                                self.last_evicted_id = Some(0);
                            }
                        } else {
                            panic!("Tried to remove key from ThreadSafeLruCache on eviction but it \
                            does not exist. This cannot happen, but has, so there is a bug.")
                        }
                    } else {
                        // eviction rejection

                        // the key is not the next key in order, so we manually find and move the
                        // node with the correct key to the rear, manipulating the LRU order
                        // and recurse so that the next key to evict is the next key in order
                        // trace!("not seq: self.map: {:?}", self.map);
                        // we start by putting the popped value back and moving it to the rear.
                        // (as if not the expected least recently used value, then it'll be close to it)
                        // assert that we have capacity to put it back. This should just be logical
                        assert_eq!(self.capacity - 1, self.len(), "We have capacity {} and we popped one item, so we should have cache size of {}, but it is {}", self.capacity, self.capacity - 1, self.len());
                        // we do not expect an eviction
                        let mut popped_lru_node_guard = popped_lru_node.lock();
                        let data_of_popped_node = popped_lru_node_guard.data.take();
                        let id_of_popped_node = popped_lru_node_guard.id;
                        drop(popped_lru_node_guard);
                        drop(popped_lru_node);
                        self.list.add_node_to_rear(id_of_popped_node, data_of_popped_node.expect("Data is None, how can this be?"));
                        // and also remember to add the value back into the map
                        let shared_ref_to_rear_node = self.list.get_tail().expect("No tail? ");
                        self.map.insert(id_of_popped_node, shared_ref_to_rear_node);
                        assert_eq!(self.capacity, self.len());

                        // trace!("Self after eviction rejection: {:?}", self);
                        // trace!("Map after eviction rejection: {:?}", self.map);

                        // acquire the next node that we expect to evict.
                        if let Some(node_with_expected_id_ref) = self.map.get(&next_expected_eviction_id) {
                            // move it to the rear
                            self.list.move_node_to_rear(node_with_expected_id_ref.clone());

                            // trace!("Self after correction of eviction rejection: {:?}", self);

                            // and recurse with same input parameters. In the next call
                            // this function will go down the "is_sequential" path.
                            return self.put_with_sequential_eviction(key, value);
                        } else {
                            panic!("Data are not being processed in the correct order, likely because \
                            your data is sparse and your cache bin size is small. In this situation \
                            sequential data points can jump time bins and do not occur in sequential \
                            order. You can try increasing the time bin sizes. A side note, processing \
                            will be slower will too small time bin intervals and this situation is \
                            pathological");
                        }
                    } // if is_sequential
                } // list.pop_back()
            }
            // we now have an evicted node with a guarantee that it is next sequentially.
            // Insert the new node at the front of the list and in the map
            self.list.add_node_to_head(key, value);
            let most_recently_used = self.list.get_most_recently_used_node();
            self.map.insert(key, most_recently_used);

            // trace!("cache capacity: {:?}, map len: {:?}", self.capacity, self.map.len());
        }
        evicted_item
    }


    /// Removes and returns the least recently used item from the cache.
    pub fn pop(&mut self) -> Option<(u32, V)> {
        // Attempt to pop the last node from the list
        if let Some(lru_ref) = self.list.pop_back() {
            let mut lru_lock = lru_ref.lock();

            // Remove the node from the map
            let evicted_key = lru_lock.id;
            self.map.remove(&evicted_key);

            // Take the data out of the node to return it
            let evicted_value = lru_lock.data.take();

            // Drop the lock before returning the value
            drop(lru_lock);

            // Return the key and value if we have a value
            evicted_value.map(|value| (evicted_key, value))
        } else {
            // If the list is empty, return None
            None
        }
    }


    /// Empties the cache in the order of least recently used and returns the flushed items.
    pub fn flush(&mut self) -> VecDeque<(u32, V)> {
        let mut items = Vec::new();

        // Pop all items from the cache and collect them into a Vec.
        while let Some(item) = self.pop() {
            items.push(item);
        }

        // Sort the collected items. This assumes that the items implement the PartialOrd trait.
        // If you need a specific sorting order, you might adjust the sorting logic accordingly.
        items.sort_by_key(|item| item.0);

        VecDeque::from(items)
    }

    pub fn get_processed_data(&mut self, key: u32) -> V {
        if let Some(node_ref) = self.map.get(&key) {
            // Lock the node to access its data
            let mut node_lock = node_ref.lock();
            if let Some(data) = node_lock.data.take() { // This takes the data out of the node, leaving None in its place
                // Optionally, move the node to the front of the list
                self.list.move_node_to_head(Arc::clone(node_ref));
                return data; // Return the taken data
            }
        }
        // Return an empty V if there's no data for the key
        V::default()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

impl<V> Debug for ThreadSafeLruCache<V>
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
        let cache: ThreadSafeLruCache<String> = ThreadSafeLruCache::new(2);
        assert_eq!(cache.capacity, 2);
        assert!(cache.map.is_empty());
        assert_eq!(0, cache.list.len());
    }

    #[test]
    fn test_put_and_get() {
        let mut cache = ThreadSafeLruCache::new(2);
        assert_eq!(cache.get_clone(1), None);
        assert!(cache.list.is_empty());
        assert!(cache.map.is_empty());

        cache.put(1, "one".to_string());
        assert_eq!(cache.get_clone(1), Some("one".to_string()));
        assert_eq!(1, cache.list.len());
        assert_eq!(1, cache.map.len());

        cache.put(2, "two".to_string());
        assert_eq!(cache.get_clone(2), Some("two".to_string()));
        assert_eq!(2, cache.list.len());
        assert_eq!(2, cache.map.len());
    }

    #[test]
    fn checl_cache_size_5() {
        init_logger(Trace);
        let mut cache = ThreadSafeLruCache::new(5);
        cache.put(6, "six".to_string());
        cache.put(5, "five".to_string());
        cache.put(4, "four".to_string());
        cache.put(3, "three".to_string());
        cache.put(2, "two".to_string());

        trace!("cache: {:?}", cache);
        let (k, popped) = cache.put(1, "one".to_string()).unwrap();
        assert_eq!(popped, "six");
        assert_eq!(k, 6);
    }

    #[test]
    fn checl_cache_size_2() {
        init_logger(Trace);
        let mut cache = ThreadSafeLruCache::new(2);
        cache.put(3, "three".to_string());
        cache.put(2, "two".to_string());

        trace!("cache: {:?}", cache);
        let (k, popped) = cache.put(1, "one".to_string()).unwrap();
        assert_eq!(popped, "three");
        assert_eq!(k, 3);
    }

    #[test]
    fn check_cache_size_2_after_get() {
        init_logger(Trace);
        let mut cache = ThreadSafeLruCache::new(2);
        cache.put(3, "three".to_string());
        cache.put(2, "two".to_string());
        trace!("cache before get 3: {:?}", cache);
        cache.get_clone(3);
        trace!("cache after get 3: {:?}", cache);

        let (_k, popped) = cache.put(1, "one".to_string()).unwrap();
        assert_eq!(popped, "two");
    }

    #[test]
    fn test_put_eviction() {
        init_logger(Trace);
        let mut cache = ThreadSafeLruCache::new(2);
        // order: 1
        trace!("cache: {:?}", cache);
        cache.put(1, "one".to_string());
        trace!("cache: {:?}", cache);
        // order: 2, 1
        cache.put(2, "two".to_string());
        trace!("cache: {:?}", cache);
        // order: 1, 2
        assert_eq!(cache.get_clone(1), Some("one".to_string()));
        trace!("cache: {:?}", cache);

        // now we put 3, it takes the place of 1 and 2 is popped
        let (_k, popped) = cache.put(3, "three".to_string()).unwrap();
        assert_eq!(popped, "two".to_string());
    }

    #[test]
    fn test_lru_eviction_order() {
        let mut cache = ThreadSafeLruCache::new(2);
        cache.put(1, "one".to_string());
        // [1]
        cache.put(2, "two".to_string());
        // [2, 1]
        assert_eq!(cache.get_clone(1), Some("one".to_string())); // This should make key 1 the most recently used
        // [1, 2]

        let (_k, two_popped) = cache.put(3, "three".to_string()).unwrap(); // This should evict key 2
        // [3, 1] ==> 2
        assert_eq!(two_popped, "two");
        assert_eq!(cache.get_clone(2), None);
        assert_eq!(cache.get_clone(1), Some("one".to_string()));
        // [1, 3]
        assert_eq!(cache.get_clone(3), Some("three".to_string()));
        // [3, 1]

        let (_k, one_popped) = cache.put(2, "two".to_string()).unwrap(); // This should evict key 1
        // [2, 3] => 1
        assert_eq!(one_popped, "one");
        assert_eq!(cache.get_clone(1), None);
        // [2, 3]

        assert_eq!(cache.get_clone(2), Some("two".to_string()));
        assert_eq!(cache.get_clone(3), Some("three".to_string()));
    }


    #[test]
    fn test_get_mut_updates_value() {
        let mut cache = ThreadSafeLruCache::new(2);
        cache.put(1, 10);
        assert_eq!(cache.get_clone(1), Some(10));

        cache.get_mut(1, |value| {
            *value += 5;
        });

        assert_eq!(cache.get_clone(1), Some(15));
    }

    #[test]
    fn test_get_mut_nonexistent_key() {
        let mut cache = ThreadSafeLruCache::new(2);
        cache.put(1, 10);

        assert!(cache.get_mut(2, |_| ()).is_none());
    }

    #[test]
    fn test_pop_removes_least_recently_used() {
        let mut cache = ThreadSafeLruCache::new(2);
        cache.put(1, "one".to_string());
        cache.put(2, "two".to_string());
        cache.get_clone(1); // This will mark key 1 as recently used

        let evicted = cache.pop();
        assert_eq!(evicted, Some((2, "two".to_string()))); // Key 2 should be evicted since it's the LRU
    }

    #[test]
    fn test_pop_empty_cache() {
        let mut cache: ThreadSafeLruCache<String> = ThreadSafeLruCache::new(2);
        assert!(cache.pop().is_none());
    }

    #[test]
    fn test_put_and_pop_sequence() {
        let mut cache = ThreadSafeLruCache::new(2);
        cache.put(1, "one".to_string());
        cache.put(2, "two".to_string());

        assert_eq!(cache.pop(), Some((1, "one".to_string()))); // Pops key 1 since it's the LRU
        assert_eq!(cache.get_clone(2), Some("two".to_string()));

        cache.put(3, "three".to_string()); // Puts key 3 and evicts key 2 as it's now the LRU
        assert_eq!(cache.pop(), Some((2, "two".to_string())));
        assert_eq!(cache.get_clone(3), Some("three".to_string()));
    }

    #[test]
    fn test_sequential_eviction_when_already_ordered() {
        init_logger(Trace);
        let mut cache = ThreadSafeLruCache::new(3);

        // Populate the cache in a specific order
        cache.put_with_sequential_eviction(0, "zero".to_string());
        cache.put_with_sequential_eviction(1, "one".to_string());
        cache.put_with_sequential_eviction(2, "two".to_string());

        // Peek at the nodes to verify the current order is 3 - 2 - 1 (from most to least recently used)
        let first_peeked_data = cache.list.peek_position(0).unwrap().lock().data.clone();
        let second_peeked_data = cache.list.peek_position(1).unwrap().lock().data.clone();
        let third_peeked_data = cache.list.peek_position(2).unwrap().lock().data.clone();

        assert_eq!(first_peeked_data, Some("two".to_string()), "First node should be 'two'");
        assert_eq!(second_peeked_data, Some("one".to_string()), "Second node should be 'one'");
        assert_eq!(third_peeked_data, Some("zero".to_string()), "Third node should be 'zero'");

        // This should cause an eviction. Given the setup, item 1 should be evicted.
        let evicted = cache.put_with_sequential_eviction(3, "three".to_string());
        assert_eq!(evicted, Some((0, "zero".to_string())), "Item 1 should be evicted first in sequential order.");

        // Verify the state and order of the cache after the first eviction using peek_position
        let first_peeked_data = cache.list.peek_position(0).unwrap().lock().data.clone();
        let second_peeked_data = cache.list.peek_position(1).unwrap().lock().data.clone();
        let third_peeked_data = cache.list.peek_position(2).unwrap().lock().data.clone();

        assert_eq!(first_peeked_data, Some("three".to_string()), "After eviction, first node should be 'four'");
        assert_eq!(second_peeked_data, Some("two".to_string()), "After eviction, second node should be 'three'");
        assert_eq!(third_peeked_data, Some("one".to_string()), "After eviction, third node should be 'two'");
    }

    #[test]
    fn test_multiple_sequential_eviction_scenarios() {

        // already tested situation where one eviction occurs. 
        // do not test this stuff again
        init_logger(Trace);
        let mut cache = ThreadSafeLruCache::new(3);

        // Populate the cache with keys starting from 0, in a specific order
        cache.put_with_sequential_eviction(0, "zero".to_string());
        cache.put_with_sequential_eviction(1, "one".to_string());
        cache.put_with_sequential_eviction(2, "two".to_string());

        trace!("1.cache: {:?}", cache);

        // [2, 1, 0]

        // Manually adjust the LRU order, for example, by accessing item 0 to make it the most recently used
        // This simulates a scenario where the least recently used is not the next sequential expected value
        let _ = cache.get_clone(0);
        // Accessing the item to change its LRU position
        // [0, 2, 1]
        trace!("2.cache: {:?}", cache);

        // The expected eviction sequence should adjust automatically to maintain sequential order
        // Attempt to add a new item to trigger eviction and sequential adjustment
        let evicted = cache.put_with_sequential_eviction(3, "three".to_string());
        // [0, 2, 1] ==> [2, 1, 0] ==> [3, 2, 1]
        trace!("3.cache: {:?}", cache);
        trace!("evicted: {:?}", evicted);
        // Despite the manual LRU adjustment, the cache should still evict the item with key 0 ("zero")
        // since it's the next in the expected sequential order.
        assert_eq!(evicted, Some((0, "zero".to_string())), "Item with key 0 ('zero') should be evicted.");

        // H [3, 2, 1] T => [2, 3, 1]
        let _ = cache.get_clone(2);
        trace!("4.cache: {:?}", cache);

        // The expected eviction sequence should adjust automatically to maintain sequential order
        // Attempt to add a new item to trigger eviction and sequential adjustment
        let evicted = cache.put_with_sequential_eviction(4, "four".to_string());

        trace!("5.cache: {:?}", cache);
        trace!("evicted: {:?}", evicted);

        // Despite the manual LRU adjustment, the cache should still evict the item with key 0 ("zero")
        // since it's the next in the expected sequential order.
        assert_eq!(evicted, Some((1, "one".to_string())), "Item with key 1 ('one') should be evicted.");
    }
}


#[cfg(test)]
pub mod stress_test {
    use std::collections::VecDeque;
    use crate::thread_safe_lru_cache::ThreadSafeLruCache;

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::collections::HashSet;
        use log::LevelFilter::Trace;
        use log::trace;
        use rand::Rng;
        use crate::init_logger::init_logger;
        use crate::thread_safe_lru_cache::ThreadSafeLruCache;

        #[test]
        fn stress_test_thread_safe_lru_cache() {
            init_logger(Trace);
            let mut rng = rand::thread_rng();
            let capacity = 1000;
            let mut cache = ThreadSafeLruCache::new(capacity);

            // Step 1: Fill the cache to its capacity with sequential insertions.
            for i in 0..capacity {
                cache.put_with_sequential_eviction(i as u32, format!("value{}", i));
            }
            // Verify cache is filled to its capacity
            assert_eq!(cache.len(), capacity, "Cache should be filled to its capacity.");

            // Step 2: Randomly access a subset of the inserted items to adjust LRU.
            let mut accessed_keys = HashSet::new();
            for _ in 0..(capacity / 2) {
                let key = rng.gen_range(0..capacity) as u32;
                assert!(cache.get_clone(key).is_some(), "Accessed item should exist in cache.");
                accessed_keys.insert(key);
            }
            
            for key in accessed_keys.iter(){
                trace!("key: {:?}", key);
            }

            // Step 3: Insert additional items to trigger evictions and verify sequential eviction order.
            let expected_eviction_order = (0..capacity)
                .map(|x| x as u32)
                .collect::<Vec<_>>();
            let mut actual_eviction_order = Vec::new();
            for i in 0..capacity{
                let new_key = capacity+i;
                let new_value = format!("value{}", new_key);
                let evicted = cache.put_with_sequential_eviction(new_key as u32, new_value);
                assert!(evicted.is_some());
                actual_eviction_order.push(evicted.unwrap().0);
            }
            assert_eq!(expected_eviction_order, actual_eviction_order);
        }
    }


    #[test]
    fn flush_returns_sorted_data() {
        let mut cache = ThreadSafeLruCache::<String>::new(10);

        // Insert data into the cache in a non-sorted order
        cache.put(5, "Value5".to_string());
        cache.put(1, "Value1".to_string());
        cache.put(3, "Value3".to_string());
        cache.put(2, "Value2".to_string());
        cache.put(4, "Value4".to_string());

        // Now flush the cache and retrieve the flushed items
        let flushed_items: VecDeque<(u32, String)> = cache.flush();

        // Expected sorted order based on the keys
        let expected_order: VecDeque<(u32, String)> = VecDeque::from([
            (1, "Value1".to_string()),
            (2, "Value2".to_string()),
            (3, "Value3".to_string()),
            (4, "Value4".to_string()),
            (5, "Value5".to_string()),
        ]);

        // Verify the flushed items are in the expected sorted order
        assert_eq!(flushed_items, expected_order, "Flushed items are not in the expected sorted order.");
    }
}
