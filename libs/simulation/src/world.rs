use crate::*;

#[derive(Debug)]
pub struct World {
    pub(crate) animals: Vec<Animal>,
    pub(crate) foods: Vec<Food>,
}

impl World {
    pub fn random(rng: &mut dyn RngCore) -> Self {
        // generate m animals
        let animals = (0..40).map(|_| Animal::random(rng)).collect();

        // generate n foods
        let foods = (0..60).map(|_| Food::random(rng)).collect();

        Self { animals, foods }
    }

    // getter function, allows us to get the state of world objects, e.g. location of food, animals, etc.
    pub fn animals(&self) -> &[Animal] {
        &self.animals
    }

    pub fn foods(&self) -> &[Food] {
        &self.foods
    }
}


