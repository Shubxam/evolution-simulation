use rand::{Rng, RngCore};

#[derive(Debug)]
pub struct Network{
    layers: Vec<Layer>,
}

#[derive(Debug)]
pub struct LayerTopology {
    pub neurons: usize,
}

impl Network {
    pub fn new(layers: Vec<Layer>) -> Self {
        Self {layers}
    }

    pub fn random(rng: &mut dyn RngCore, layers: &[LayerTopology]) -> Self {

        assert!(layers.len() > 1);

        let layers = layers.windows(2)
                            .map(|adjacent_layers| Layer::random(rng, adjacent_layers[0].neurons, adjacent_layers[1].neurons))
                            .collect();

        Self {layers: layers}
    }

    pub fn propogate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.layers
            .iter()
            .fold(inputs, |inputs: Vec<f32>, layer: &Layer| layer.propogate(inputs))
    }
}

#[derive(Debug)]
pub struct Layer{
    neurons: Vec<Neuron>
}

impl Layer {

    fn random(rng: &mut dyn RngCore, input_size: usize, output_size: usize) -> Self{
        let neurons = (0..output_size)
                            .map(|_output_neuron_id| Neuron::random(rng, input_size))
                            .collect();

        Self {neurons: neurons}

    }

    fn propogate(&self, inputs: Vec<f32>) -> Vec<f32> {

        self.neurons
            .iter()
            .map(|neuron| neuron.propogate(&inputs))
            .collect()

    }
}

#[derive(Debug)]
struct Neuron{
    bias: f32,
    weights: Vec<f32>
}

impl Neuron {

    fn random(rng: &mut dyn RngCore, input_size: usize) -> Self {
        
        let bias = rng.gen_range(-1.0..=1.0);
        let weights = (0..input_size)
            .map(|_| rng.gen_range(-1.0..=1.0))
            .collect();
        Self {bias, weights}
    }

    fn propogate(&self, inputs: &[f32]) -> f32 {

        assert_eq!(inputs.len(), self.weights.len());

        let output = self.weights
                                .iter()
                                .zip(inputs)
                                .map(|(weight, input)| weight * input)
                                .sum::<f32>();

        (self.bias + output).max(0.0)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use approx::assert_relative_eq;

    mod neuron_tests {
        use super::*;

        #[test]
        fn random(){
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let neuron = Neuron::random(&mut rng, 4);
    
            assert_relative_eq!(neuron.bias, -0.6255188);
            assert_relative_eq!(neuron.weights.as_slice(), [0.67383957, 0.8181262, 0.26284897, 0.5238807].as_ref());
        }
    
        #[test]
        fn propogate(){
            let neuron = Neuron {
                bias : 0.5,
                weights: vec![-0.3, 0.8]
            };
    
            // Ensure ReLU function works
            assert_relative_eq!(neuron.propogate(&[2.2, 0.9]), 0.56);
        }
    }

    mod layer_tests {
        use super::*;

        #[test]
        fn random(){
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let neurons_count = 3;
            let layer = Layer::random(&mut rng, 4, neurons_count);

            assert_eq!(layer.neurons.len(), neurons_count);
        }

        #[test]
        fn propogate(){
            let layer = Layer {
                neurons : vec![Neuron {bias: 0.5, weights: vec![-0.3, 0.8]},
                            Neuron {bias: 0.1, weights: vec![0.11, -1.5]},
                            Neuron {bias: -0.15, weights: vec![-0.11, 1.5]},
                        ]
            };

            // assert_relative_eq!(layer.propogate((&[2.2, 0.9]).to_vec()),
            //     vec![0.56, 0.0, 0.958]);
            assert_eq!(layer.propogate([2.2, 0.9].to_vec()).len(), 3);
        }

    }

    mod function_tests {
        use super::*;

        #[test]
        fn random(){
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let layer_struct = [LayerTopology{neurons: 2}, LayerTopology{neurons:3}, LayerTopology{neurons:2}];
            let network = Network::random(&mut rng, &layer_struct);
            assert_eq!(network.layers.len(), layer_struct.len() - 1);
        }

        // fn propogate(){
        //     todo!();
        // }
    }
}