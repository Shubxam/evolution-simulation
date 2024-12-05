use rand::{Rng, RngCore};

#[derive(Debug)]
/// Layer Topology Abstraction
/// Each layer will have a number of neurons
pub struct LayerTopology {
    pub neurons: usize,
}

#[derive(Debug)] 
/// Neural Network Abstraction
/// A neural network will have a vector of type Layer.
pub struct Network{
    layers: Vec<Layer>,
}

impl Network {
    /// create a new neural network object.
    pub fn new(layers: Vec<Layer>) -> Self {
        Self {layers}
    }

    /// create a new neural network object with random weights and biases.
    /// the layers count will be equal to the length of the layers vector minus 1.
    /// if zeroth layer has n neurons, that means the input size for first layer is n.
    /// > ex: if the layer topology is [3,2,3], it means that first layer will have 3 inputs for each neuron and 2 outputs in total, second layer will have 2 inputs for each neuron and 3 outputs.  
    /// So the network in whole will have 3 inputs and 3 outputs.
    pub fn random(rng: &mut dyn RngCore, layers: &[LayerTopology]) -> Self {

        assert!(layers.len() > 1);

        let layers = layers.windows(2)
                            .map(|adjacent_layers| Layer::random(rng, adjacent_layers[0].neurons, adjacent_layers[1].neurons))
                            .collect();

        Self {layers}
    }

    /// process inputs for neural network and returns a vector of outputs based on neurons in the last layer.
    pub fn propogate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.layers
            .iter()
            .fold(inputs, |inputs: Vec<f32>, layer: &Layer| layer.propogate(inputs))
    }
}

#[derive(Debug)]
/// Abstraction for Layer
/// Each layer holds a vector of neurons
pub struct Layer{
    neurons: Vec<Neuron>
}

impl Layer {

    /// create a new layer object with random neurons
    /// The number of neurons in the layer is equal to the output size
    /// The number of weights in each neuron is equal to the input size
    fn random(rng: &mut dyn RngCore, input_size: usize, output_size: usize) -> Self{
        let neurons = (0..output_size)
                            .map(|_output_neuron_id| Neuron::random(rng, input_size))
                            .collect();

        Self {neurons}

    }

    /// propogate the input through the layer
    /// The output is the sum of the product of the weights and inputs
    fn propogate(&self, inputs: Vec<f32>) -> Vec<f32> {

        self.neurons
            .iter()
            .map(|neuron| neuron.propogate(&inputs))
            .collect()

    }
}

#[derive(Debug)]
/// Abstraction for Neuron
/// Each neuron holds a bias and a vector of weights
/// The number of weights is equal to the input size
struct Neuron{
    bias: f32,
    weights: Vec<f32>
}

impl Neuron {

    /// create a new neuron object with random weights and bias
    fn random(rng: &mut dyn RngCore, input_size: usize) -> Self {
        
        let bias = rng.gen_range(-1.0..=1.0);
        let weights = (0..input_size)
            .map(|_| rng.gen_range(-1.0..=1.0))
            .collect();
        Self {bias, weights}
    }

    /// propogate the input through the neuron
    /// The output is the sum of the product of the weights and inputs
    /// The output is then passed through a ReLU function
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