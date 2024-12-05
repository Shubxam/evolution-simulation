use rand::seq::SliceRandom;
use rand::{Rng, RngCore};
use std::ops::Index;

// this struct is generic over trait S
pub struct GeneticAlgorithm<S, C> {
    selection_method: S,
    crossover_method: C,
}

pub trait Individual {
    fn fitness(&self) -> f32;
    fn chromosome(&self) -> &Chromosome;
}

pub struct RouletteWheelSelection;

pub struct UniformCrossover;

#[derive(Clone, Debug)]
pub struct Chromosome{
    genes: Vec<f32>,
}

impl Chromosome {
    pub fn len(&self) -> usize {
        self.genes.len()
    }

    // function returns a type which implements Iterator trait generic over f32
    pub fn iter(&self) -> impl Iterator<Item = &f32> {
        self.genes.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.genes.iter_mut()
    }
}

// implementation of Index trait for Chromosome struct
impl Index<usize> for Chromosome {
    
    // type alias for Output associated type
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.genes[index]
    }
}

// implementation of FromIterator trait for Chromosome struct
// allowing to create a Chromosome from an iterator of f32 values
impl FromIterator<f32> for Chromosome {
    fn from_iter<T: IntoIterator<Item = f32>>(iter: T) -> Self {
        Self {
            genes: iter.into_iter().collect(),
        }
    }

}

// implementation of IntoIterator trait for Chromosome struct
// allowing to iterate over the genes of a Chromosome
impl IntoIterator for Chromosome {
    type Item = f32;
    type IntoIter = std::vec::IntoIter<f32>;

    fn into_iter(self) -> Self::IntoIter {
        self.genes.into_iter()
    }
}

/// A trait implementing selection method that selects individuals based on their fitness.
pub trait SelectionMethod {
    // lifetime specifier to tie input and output reference together
    // function works with any generic type I that implements Individual trait.
    fn select<'a, I>(&self, rng: &mut dyn RngCore, population: &'a [I]) -> &'a I
    where
        I: Individual;
}

pub trait CrossoverMethod {
    fn crossover(&self, rng: &mut dyn RngCore, parent_a: &Chromosome, parent_b: &Chromosome) -> Chromosome;
}

// Implementation of Selection Method trait for Type Roulette Wheel Selection.
impl SelectionMethod for RouletteWheelSelection {
    fn select<'a, I>(&self, rng: &mut dyn RngCore, population: &'a [I]) -> &'a I
    where
        I: Individual,
    {
        population
            .choose_weighted(rng, |individual| individual.fitness())
            .expect("got an empty population")
    }
}

impl CrossoverMethod for UniformCrossover {
    fn crossover(&self, rng: &mut dyn RngCore, parent_a: &Chromosome, parent_b: &Chromosome) -> Chromosome {
        assert_eq!(parent_a.len(), parent_b.len());

        parent_a
        .iter()
        .zip(parent_b.iter())
        .map(|(&a, &b)| if rng.gen_bool(0.5) {a} else {b})
        .collect()

    }
}

impl<S, C> GeneticAlgorithm<S, C> 
where 
    S: SelectionMethod, 
    C: CrossoverMethod,
{

    pub fn new(
        selection_method: S, crossover_method: C
    )-> Self {
        Self { 
            selection_method, 
            crossover_method, 
        }
    }


    pub fn evolve<I>(&self, rng: &mut dyn RngCore, population: &[I]) -> Vec<I>
    where
        I: Individual,
    {
        assert!(!population.is_empty());

        (0..population.len())
        .map(
            |_| {
                // Selection
                let parent_a = self.selection_method.select(rng, population).chromosome();
                let parent_b = self.selection_method.select(rng, population).chromosome();
                // Crossover
                let mut child = self.crossover_method.crossover(rng, parent_a, parent_b);
                // TODO Mutation
                todo!();
            }
        ).collect()
    }
}




#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::collections::BTreeMap;
    use std::iter::FromIterator;

    #[test]
    fn roulette_wheel_selection() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());
        let population = vec![
            TestIndividual::new(2.0),
            TestIndividual::new(1.0),
            TestIndividual::new(4.0),
            TestIndividual::new(3.0),
        ];

        let mut actual_histogram = BTreeMap::new();

        for _ in 0..1000 {
            // sample a random fitness score
            let fitness = RouletteWheelSelection
                .select(&mut rng, &population)
                .fitness() as i32;

            // increase the count
            *actual_histogram.entry(fitness).or_insert(0) += 1;
        }

        let expected_histogram = BTreeMap::from_iter([(1, 98), (2, 202), (3, 278), (4, 422)]);
        assert_eq!(actual_histogram, expected_histogram);
    }

    #[derive(Clone, Debug)]
    struct TestIndividual {
        fitness: f32,
    }

    impl TestIndividual {
        fn new(fitness: f32) -> Self {
            Self { fitness }
        }
    }

    impl Individual for TestIndividual {
        fn fitness(&self) -> f32 {
            self.fitness
        }
        fn chromosome(&self) -> &Chromosome {
            panic!("not supported for TestIndividual")
        }
    }

    #[test]
    fn uniform_crossover() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());
        let parent_a: Chromosome = (0..=100).map(|n| n as f32).collect();
        let parent_b: Chromosome = (0..=100).map(|n| -n as f32).collect();
        let child = UniformCrossover.crossover(&mut rng, &parent_a, &parent_b);

        let diff_a = child
                            .iter()
                            .zip(parent_a.iter())
                            .filter(|(c, p)| *c != *p)
                            .count();
        let diff_b = child
                            .iter()
                            .zip(parent_b.iter())
                            .filter(|(c, p)| *c != *p)
                            .count();

        assert_eq!(diff_a, 49);
        assert_eq!(diff_b, 51);
    }
}
