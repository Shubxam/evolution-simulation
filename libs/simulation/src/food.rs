use crate::*; // imports all the modules, structs, functions, imports from lib.rs

#[derive(Debug)]
pub struct Food {
    pub(crate) position: na::Point2<f32>,
}

impl Food {
    pub fn random(rng: &mut dyn RngCore) -> Self {
        Self {
            position: rng.gen(),
        }
    }

    // getter function
    pub fn position(&self) -> na::Point2<f32> {
        self.position
    }
}
