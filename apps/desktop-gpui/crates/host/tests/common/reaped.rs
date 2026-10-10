//! A child process the test owns: killed and reaped when it goes out of
//! scope, so a failing assertion never leaves a host running (included
//! with `#[path = "common/reaped.rs"] mod reaped;`).

use std::ops::{Deref, DerefMut};
use std::process::Child;

pub struct Reaped(pub Child);

impl Deref for Reaped {
    type Target = Child;

    fn deref(&self) -> &Child {
        &self.0
    }
}

impl DerefMut for Reaped {
    fn deref_mut(&mut self) -> &mut Child {
        &mut self.0
    }
}

impl Drop for Reaped {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
