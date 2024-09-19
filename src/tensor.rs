use crate::ndarray::NDArray;
use std::cell::RefCell;
use std::rc::Rc;

pub struct Context {
    saved_tensors: Vec<Rc<Tensor>>,
    saved_data: Vec<NDArray>,
}

pub trait Function: 'static {
    fn forward(&self, ctx: &mut Context, inputs: &[&Tensor]) -> NDArray;
    fn backward(&self, ctx: &Context, grad_output: &NDArray) -> Vec<NDArray>;
}

pub struct Tensor {
    data: NDArray,
    grad: RefCell<Option<NDArray>>,
    ctx: RefCell<Option<(Rc<dyn Function>, Context)>>,
}

impl Tensor {
    pub fn new(data: NDArray) -> Rc<Self> {
        Rc::new(Self {
            data,
            grad: RefCell::new(None),
            ctx: RefCell::new(None),
        })
    }

    pub fn backward(&self) {
        unimplemented!()
    }

    // TODO: Add other methods
    pub fn add(&self, other: &Tensor) -> Tensor {
        unimplemented!()
    }
}

struct Add;

impl Function for Add {
    fn forward(&self, ctx: &mut Context, inputs: &[&Tensor]) -> NDArray {
        unimplemented!()
    }

    fn backward(&self, ctx: &Context, grad_output: &NDArray) -> Vec<NDArray> {
        unimplemented!()
    }
}
