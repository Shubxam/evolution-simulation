use rand::Rng;

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

    pub fn random(layers: &[LayerTopology]) -> Self {

        assert!(layers.len() > 1);

        let layers = layers.windows(2)
                            .map(|adjacent_layers| Layer::random(adjacent_layers[0].neurons, adjacent_layers[1].neurons))
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
struct Layer{
    neurons: Vec<Neuron>
}

impl Layer {

    fn random(input_size: usize, output_size: usize) -> Self{
        let neurons = (0..output_size)
                            .map(|_output_neuron_id| Neuron::random(input_size))
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

    fn random(input_size: usize) -> Self {

        let mut rng = rand::thread_rng();
        
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