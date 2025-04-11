// This file handles the conversion of the simulation lib's structs to a format that can be invoked by JavaScript.

use lib_simulation as sim;
use rand::prelude::*;
use wasm_bindgen::prelude::*;

// not all functions are invokable by JavaScript, only the ones that are marked with #[wasm_bindgen]
#[wasm_bindgen] 
pub struct Simulation {
    rng: ThreadRng,
    sim: sim::Simulation,
}

#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct World {
    #[wasm_bindgen(getter_with_clone)]
    pub animals: Vec<Animal>,

    #[wasm_bindgen(getter_with_clone)]
    pub foods: Vec<Food>,
}

#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct Animal {
    pub x: f32,
    pub y: f32,
    pub rotation: f32,
}

#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct Food {
    pub x: f32,
    pub y: f32,
}


#[wasm_bindgen]
impl Simulation {
    #[wasm_bindgen(constructor)] // This attribute makes the new function available to JavaScript.
    pub fn new() -> Self {
        let mut rng = thread_rng();
        let sim = sim::Simulation::random(&mut rng);

        Self { rng: rng, sim: sim }
    }

    // This function converts the simulation crate's World to the World struct defined in this file.
    // this is a getter function
    pub fn world(&self) -> World {
        World::from(self.sim.world()) // takes input and converts it to World struct
    }

    pub fn step(&mut self) {
        self.sim.step(&mut self.rng); // returns nothing, hence didn't need to convert using From trait
    }
}

/// Implement the From trait to convert the simulation crate's World to the World struct defined in this file.
/// we dont implement any other function because we have getter functions in place to access the data.

impl From<&sim::World> for World {
    // this is a required method for the From trait. It converts a World from the simulation crate to a World from this file.
    fn from(world: &sim::World) -> Self {
        let animals = world.animals().iter().map(Animal::from).collect();
        let foods = world.foods().iter().map(Food::from).collect();

        Self { animals, foods }
    }
}

// This is a conversion from the simulation crate's Animal to the Animal struct defined in this file.

impl From<&sim::Animal> for Animal {
    fn from(animal: &sim::Animal) -> Self {
        Self {
            x: animal.position().x,
            y: animal.position().y,
            rotation: animal.rotation().angle(),
        }
    }
}

// in the from trait, you define how the fields of the incoming struct relate to the fields of the struct you are converting to.
impl From<&sim::Food> for Food {
    fn from(food: &sim::Food) -> Self {
        Self {
            x: food.position().x,
            y: food.position().y,
        }
    }
}
