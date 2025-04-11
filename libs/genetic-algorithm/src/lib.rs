use rand::seq::SliceRandom;
use rand::{Rng, RngCore};
use std::ops::Index;


#[derive(Clone, Debug)]
pub struct Chromosome{
    genes: Vec<f32>,
}

impl Chromosome {
    pub fn len(&self) -> usize {
        self.genes.len()
    }

    // we create these 2 functions to allow the genes to be accessed and mutated from outside this module.
    // we do this because the field genes is private.
    
    // function returns a type which implements Iterator trait generic over f32
    pub fn iter(&self) -> impl Iterator<Item = &f32> {
        self.genes.iter()
    }
    
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.genes.iter_mut()
    }
}

// implements Index trait to enable indexing operations e.g. Chromosome[0]
impl Index<usize> for Chromosome {
    
    // type alias for Output associated type requirement of Index trait
    type Output = f32;
    
    // returns &f32
    fn index(&self, index: usize) -> &Self::Output {
        &self.genes[index]
    }
}

// allows for Chromosome to be created from an (vector) iterator of f32
impl FromIterator<f32> for Chromosome {

    // the argument must implement IntoIterator trait and contain item of type f32
    fn from_iter<T: IntoIterator<Item = f32>>(iter: T) -> Self {
        Self {
            genes: iter.into_iter().collect(),
        }
    }
    
}

// allows for Chromosome to be turned into an iterator (vector) of f32
impl IntoIterator for Chromosome {

    // type of elements in the iterator
    type Item = f32;

    // type annotation of iterator
    type IntoIter = std::vec::IntoIter<f32>;
    
    fn into_iter(self) -> Self::IntoIter {
        self.genes.into_iter()
    }
}


// selection is proportonal to fitness of individual
pub struct RouletteWheelSelection; // struct without any field

/// A trait implementing selection method that selects individuals based on their fitness.
pub trait SelectionMethod {
    // The lifetime 'a ensures safe borrowing by tying the lifetime of the returned reference to the input slice.
    // function works with any generic type I that implements Individual trait.
    fn select<'a, I>(&self, rng: &mut dyn RngCore, population: &'a [I]) -> &'a I
    where
    I: Individual;
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


pub struct UniformCrossover;

pub trait CrossoverMethod {
    // crossover method is used to combine two parent chromosomes to create a child chromosome
    fn crossover(&self, rng: &mut dyn RngCore, parent_a: &Chromosome, parent_b: &Chromosome) -> Chromosome;
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


pub trait MutationMethod {
    // mutation method is used to mutate a child chromosome
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
        // mutates each value in the child chromosome individually
        for gene in child.iter_mut(){
            if rng.gen_bool(self.chance as f64){
                let sign = if rng.gen_bool(0.5) {-1.0} else {1.0};
                // rng.gen::<f32>() generates a random number between 0.0 and 1.0
                *gene += sign * self.magnitude * rng.gen::<f32>();
            }
        }
    }
}

// this struct is generic over traits S, C, M
// S is SelectionMethod, C is CrossoverMethod, M is MutationMethod
pub struct GeneticAlgorithm<S, C, M> {
    selection_method: S,
    crossover_method: C,
    mutation_method: M,
}


impl<S, C, M> GeneticAlgorithm<S, C, M> 
where 
    S: SelectionMethod, 
    C: CrossoverMethod,
    M: MutationMethod,
{

    // constructor for GeneticAlgorithm
    pub fn new(
        selection_method: S, crossover_method: C, mutation_method: M,
    )-> Self {
        Self { 
            selection_method, 
            crossover_method, 
            mutation_method,
        }
    }


    // evolve method takes a population of individuals and returns a new population
    // in order for population to evolve, it calls evolve method once for each individual, which creates a new individual
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
                // Mutation
                self.mutation_method.mutation(rng, &mut child);

                I::create(child)
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

    #[derive(Clone, Debug, PartialEq)]
    pub enum TestIndividual {
        // tests that require access to chromosomes
        WithChromosome {chromosome: Chromosome},
        // tests that don't need access to chromosomes
        WithFitness {fitness: f32},
    }

    impl TestIndividual {
        fn new(fitness: f32) -> Self {
            Self::WithFitness { fitness }
        }
    }

    impl PartialEq for Chromosome{
        fn eq(&self, other: &Self) -> bool {
            approx::relative_eq!(self.genes.as_slice(), other.genes.as_slice())
        }
    }

    impl Individual for TestIndividual {
        fn create(chromosome: Chromosome) -> Self {
            Self::WithChromosome { chromosome }
        }
        fn fitness(&self) -> f32 {
            match self {
                Self::WithChromosome { chromosome } => {
                chromosome.iter().sum()
                },
                Self::WithFitness { fitness } => *fitness
            }
        }
        fn chromosome(&self) -> &Chromosome {
            match self {
                Self::WithChromosome { chromosome } => chromosome,
                Self::WithFitness { .. } => {
                    panic!("Not supported for TestIndividual::WithFitness")
                }
            }
        }
    }


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

    #[test]
    fn genetic_algorithm() {

        fn individual(genes: &[f32]) -> TestIndividual{
            TestIndividual::create(genes.iter().cloned().collect())
        }


        let mut rng = ChaCha8Rng::from_seed(Default::default());
        let ga = GeneticAlgorithm {
            selection_method: RouletteWheelSelection,
            crossover_method: UniformCrossover,
            mutation_method: GaussianMutation::new(0.5, 0.5),
        };
        let mut population = vec![
            individual(&[0.0, 0.0, 0.0]),
            individual(&[1.0, 1.0, 1.0]),
            individual(&[1.0, 2.0, 1.0]),
            individual(&[1.0, 2.0, 4.0]),
        ];

        // mutate population 10 times
        for _ in 0..10 {
            population = ga.evolve(&mut rng, &population);
        }

        let expected_population = vec![
            individual(&[1.6119734, 1.8159671, 0.31497368]),
            individual(&[1.0151604, 1.1331394, 0.8526902]),
            individual(&[2.1268358, 2.932069, 0.10471791]),
            individual(&[0.77124745, 1.1331394, 0.9507327]),
        ];

        assert_eq!(population, expected_population);
    }
}
