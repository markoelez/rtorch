use ndarray::NDArray;
use tensor::Tensor;

pub mod ndarray;
pub mod tensor;

fn main() {
    // create tensors
    let a = Tensor::new(NDArray::new(vec![1, 2, 3], vec![3, 1]));
    let b = Tensor::new(NDArray::new(vec![4, 5, 6], vec![3, 1]));

    // c = a + b
    let c = a.add(&b);

    // backpropagation
    c.backward();

    // get gradients
    println!("Gradient of a: {:?}", a.grad.borrow().as_ref().unwrap().buf);
    println!("Gradient of b: {:?}", b.grad.borrow().as_ref().unwrap().buf);
}
