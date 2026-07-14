//! Process-lifetime, type-keyed cache for the generated table models.
//!
//! The free-function recognizers ([`crate::cls::classify`],
//! [`crate::unet`]) don't own their loaded network in a struct — they build the
//! generated Burn `Model<B>` on first use and keep it for the process lifetime, so
//! repeated calls only pay for the forward pass, not the (expensive) weight load.
//!
//! Because those functions are now generic over the Burn backend `B`, a single
//! `OnceLock<Model<Cpu>>` static no longer suffices: each concrete `B` needs its
//! own cached instance. This module provides that as a heterogeneous, `TypeId`-keyed
//! store. The stored values are downcast back to their concrete `Arc<T>` before use,
//! so **all model dispatch stays static** — the `dyn Any` here is purely storage
//! (a type-map), never a trait object over model behaviour.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// The type-keyed store. Each entry is an `Arc<T>` for exactly one concrete model
/// type `T`, erased to `Arc<dyn Any + Send + Sync>` for storage and downcast back on
/// retrieval.
type Store = Mutex<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>;

/// Returns the global model store, initialised on first use.
fn store() -> &'static Store {
    static STORE: OnceLock<Store> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Returns the cached instance of `T`, building it with `init` on first use.
///
/// The first successful `init` for a given `T` wins and is cached for the process
/// lifetime; a failed `init` is *not* cached, so a later call can retry (e.g. after
/// a network fetch recovers). Concurrent first calls may each build an instance;
/// the store keeps whichever inserts first and both are equivalent.
///
/// # Errors
///
/// Propagates any error from `init` (a weight fetch/load failure), unchanged.
pub(crate) fn get_or_try_init<T, E>(init: impl FnOnce() -> Result<T, E>) -> Result<Arc<T>, E>
where
    T: Any + Send + Sync,
{
    let key = TypeId::of::<T>();

    // Fast path: already cached.
    if let Ok(map) = store().lock() {
        if let Some(existing) = map.get(&key) {
            if let Ok(arc) = existing.clone().downcast::<T>() {
                return Ok(arc);
            }
        }
    }

    // Build outside the lock (weight load can be slow / do I/O).
    let arc: Arc<T> = Arc::new(init()?);

    if let Ok(mut map) = store().lock() {
        // If another thread inserted first, keep theirs (equivalent); else insert.
        let entry = map
            .entry(key)
            .or_insert_with(|| arc.clone() as Arc<dyn Any + Send + Sync>);
        if let Ok(shared) = entry.clone().downcast::<T>() {
            return Ok(shared);
        }
    }

    // Lock poisoned or downcast unexpectedly failed: return the freshly built
    // instance rather than error, so the caller still gets a working model.
    Ok(arc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caches_same_instance_per_type() {
        let a = get_or_try_init::<u64, ()>(|| Ok(7)).unwrap();
        let b = get_or_try_init::<u64, ()>(|| Ok(9)).unwrap();
        // Second init is ignored; both share the first-cached value.
        assert_eq!(*a, 7);
        assert_eq!(*b, 7);
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn failed_init_is_not_cached() {
        // A distinct type so this test is independent of the one above.
        struct Marker(u32);
        let first = get_or_try_init::<Marker, &str>(|| Err("boom"));
        assert!(first.is_err());
        let second = get_or_try_init::<Marker, &str>(|| Ok(Marker(3))).unwrap();
        assert_eq!(second.0, 3);
    }
}
