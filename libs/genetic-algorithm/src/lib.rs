use rand::seq::SliceRandom;
use rand::{Rng, RngCore};
use std::ops::Index;

// this struct is generic over trait S
pub struct GeneticAlgorithm<S, C, M> {
    selection_method: S,
    crossover_method: C,
    mutation_method: M,
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

pub trait MutationMethod {
    fn mutation(&self, rng: &mut dyn RngCore, child: &mut Chromosome);
}

#[derive(Debug, Clone)]
pub struct GaussianMutation {
    // probability of mutation for a gene
    // [0,1]
    chance: f32,
    // max Degree of mutation
    // [0,3]
    magnitude: f32,
}

impl GaussianMutation {
    pub fn new(chance: f32, magnitude: f32) -> Self {
        assert!(chance >= 0.0 && chance <= 1.0);
        Self {
            chance, magnitude
        }
    }
}

impl MutationMethod for GaussianMutation {
    fn mutation(&self, rng: &mut dyn RngCore, child: &mut Chromosome) {
        for gene in child.iter_mut(){
            if rng.gen_bool(self.chance as f64){
                let sign = if rng.gen_bool(0.5) {-1.0} else {1.0};
                *gene += sign * self.magnitude * rng.gen::<f32>();
            }
        }
    }
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

impl<S, C, M> GeneticAlgorithm<S, C, M> 
where 
    S: SelectionMethod, 
    C: CrossoverMethod,
    M: MutationMethod,
{

    pub fn new(
        selection_method: S, crossover_method: C, mutation_method: M,
    )-> Self {
        Self { 
            selection_method, 
            crossover_method, 
            mutation_method,
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
                self.mutation_method.mutation(rng, &mut child);
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
    mod gaussian_mutation{
        use super::*;
        fn actual(chance:f32, magnitude:f32) -> Vec<f32>{
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let mut child = vec![1.0, 2.0, 3.0, 4.0, 5.0].into_iter().collect();

            GaussianMutation::new(chance, magnitude).mutation(&mut rng, &mut child);
            child.into_iter().collect()
        }

        mod zero_chance{
            use approx::assert_relative_eq;
            fn actual(magnitude:f32) -> Vec<f32>{
                super::actual(0.0, magnitude)
            }
            mod zero_magnitude{
                use super::*;
                #[test]
                fn doesnot_change_original_chromosome(){
                    let actual = actual(0.0);
                    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];

                    assert_relative_eq!(actual.as_slice(), expected.as_slice());
                }
            }
            mod non_zero_magnitude{
                use super::*;
                #[test]
                fn doesnot_change_original_chromosome(){
                    let actual = super::actual(0.5);
                    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];

                    assert_relative_eq!(actual.as_slice(), expected.as_slice());
                }
            }
        }
        mod zero_to_1_chance{
            use approx::assert_relative_eq;
            fn actual(magnitude:f32) -> Vec<f32> {
                super::actual(0.3, magnitude)
            }
            mod zero_magnitude{
                use super::*;
                #[test]
                fn doesnot_change_original_chromosome(){
                    let actual = actual(0.0);
                    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];

                    assert_relative_eq!(actual.as_slice(), expected.as_slice());
                }
            }
            mod non_zero_magnitude{
                use super::*;
                #[test]
                fn might_change_the_original_chromosome(){
                    let actual = actual(2.1);
                    let expected = vec![1.0, 2.0, 2.0576246, 4.0, 5.0];

                    assert_relative_eq!(actual.as_slice(), expected.as_slice());
                }
            }

        }
        mod certain_chance{
            use approx::assert_relative_eq;
            fn actual(magnitude:f32) -> Vec<f32> {
                super::actual(1.0, magnitude)
            }
            mod zero_magnitude{
                use super::*;
                #[test]
                fn doesnot_change_original_chromosome(){
                    let actual = actual(0.0);
                    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];

                    assert_relative_eq!(actual.as_slice(), expected.as_slice());
                }
            }
            mod non_zero_magnitude{
                use super::*;
                #[test]
                fn changes_the_original_chromosome(){
                    let actual = actual(2.7);
                    let expected = vec![3.4544702, 2.6275227, 1.7883743, 3.7327676, 3.0489318];

                    assert_relative_eq!(actual.as_slice(), expected.as_slice());
                }
            }
        }
    }
}
