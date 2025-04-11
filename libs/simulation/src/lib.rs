// this file handles the simulation logic

mod world;
mod animal;
mod food;
mod eye;

// pub because we want to use the structs in other files
pub use self::{animal::*, food::*, world::*, eye::*}; // imports all public modules from submodules to this crate

use ::nalgebra as na;
use rand::{Rng, RngCore};
pub struct Simulation {
    world: World,
}




impl Simulation {
    pub fn random(rng: &mut dyn RngCore) -> Self {
        Self {
            world: World::random(rng),
        }
    }

    // getter function
    pub fn world(&self) -> &World {
        &self.world
    }

    /// This function is called to advance the simulation by one step.
    /// It processes the movements of the animals and the collisions with the food.
    pub fn step(&mut self, rng: &mut dyn RngCore) {
        self.process_collisions(rng);
        self.process_movements();
    }

    /// This function processes the movements of the animals.
    /// It updates the position of each animal based on its rotation and speed.
    /// It also wraps the position of the animals around the screen.
    fn process_movements(&mut self) {
        for animal in &mut self.world.animals {
            animal.position += animal.rotation * na::Vector2::new(0.0, animal.speed);

            animal.position.x = na::wrap(animal.position.x, 0.0, 1.0);
            animal.position.y = na::wrap(animal.position.y, 0.0, 1.0);
        }
    }

    /// This function processes the collisions between the animals and the food.
    /// If an animal is close enough to a food, the food is moved to a new random position.
    /// The distance threshold for a collision is set to 0.01.
    fn process_collisions(&mut self, rng: &mut dyn RngCore) {
        for animal in &self.world.animals {
            for food in &mut self.world.foods {
                let distance = na::distance(&animal.position, &food.position);

                if distance <= 0.01 {
                    food.position = rng.gen();
                }
            }
        }
    }
}


