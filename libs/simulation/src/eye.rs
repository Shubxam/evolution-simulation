use crate::*; 
use std::f32::consts::*;

const FOV_RANGE: f32 = 0.25; // how far in any direction the eye can see: 25% of the world
const FOV_ANGLE: f32 = PI + FRAC_PI_4; // 5pi / 4 = 225 degrees in radians
const CELLS: usize = 8; // number of cells (photoreceptors) in the eye


#[derive(Debug)]
pub struct Eye {
    fov_range: f32,
    fov_angle: f32,
    cells: usize,
}

impl Eye{
    pub fn process_vision() -> Vec<f32> {
        todo!()
    }
}