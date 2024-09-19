use ndarray::NDArray;
use tensor::Tensor;

pub mod ndarray;
pub mod tensor;

fn main() {
    let t1 = Tensor::new(NDArray::ones(vec![3, 4]));
    let t2 = Tensor::new(NDArray::ones(vec![3, 4]));

    let t3 = t1.add(&t2);
    t3.backward();
}
