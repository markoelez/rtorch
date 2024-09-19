use crate::ndarray::NDArray;
use std::{cell::RefCell, collections::HashSet, rc::Rc};

pub struct Context {
    saved_tensors: Vec<Rc<Tensor>>,
}

#[derive(Clone)]
pub enum Operation {
    Add,
    // TODO: other ops
}

impl Operation {
    fn forward(&self, ctx: &mut Context, inputs: &[Rc<Tensor>]) -> NDArray {
        match self {
            Operation::Add => {
                assert_eq!(inputs.len(), 2, "Add op requires two inputs!");
                let a = &inputs[0].data;
                let b = &inputs[1].data;
                ctx.saved_tensors = inputs.iter().cloned().collect();
                a + b
            }
        }
    }

    fn backward(&self, ctx: &Context, grad_output: &NDArray) -> Vec<NDArray> {
        match self {
            Operation::Add => {
                vec![grad_output.clone(), grad_output.clone()]
            }
        }
    }

    fn apply(&self, inputs: Vec<Rc<Tensor>>) -> Rc<Tensor> {
        let mut ctx = Context {
            saved_tensors: vec![],
        };
        let data = self.forward(&mut ctx, &inputs);
        Rc::new(Tensor {
            data,
            grad: RefCell::new(None),
            ctx: Some((self.clone(), ctx)),
        })
    }
}

pub struct Tensor {
    data: NDArray,
    pub grad: RefCell<Option<NDArray>>,
    ctx: Option<(Operation, Context)>,
}

impl Tensor {
    pub fn new(data: NDArray) -> Rc<Self> {
        Rc::new(Self {
            data,
            grad: RefCell::new(None),
            ctx: None,
        })
    }

    pub fn backward(&self) {
        let mut visited = HashSet::new();
        let grad_output = NDArray::ones_like(&self.data);
        self._backward(&grad_output, &mut visited);
    }

    fn _backward(&self, grad_output: &NDArray, visited: &mut HashSet<usize>) {
        let tensor_id = self as *const _ as usize;

        if !visited.insert(tensor_id) {
            return;
        }

        {
            let mut grad = self.grad.borrow_mut();
            if let Some(ref existing_grad) = *grad {
                *grad = Some(existing_grad + grad_output);
            } else {
                *grad = Some(grad_output.clone());
            }
        }

        if let Some((ref operation, ref ctx)) = self.ctx {
            let input_grads = operation.backward(ctx, grad_output);
            for (input_tensor, input_grad) in ctx.saved_tensors.iter().zip(input_grads.iter()) {
                input_tensor._backward(input_grad, visited);
            }
        }
    }

    pub fn add(self: &Rc<Self>, other: &Rc<Self>) -> Rc<Tensor> {
        let inputs = vec![Rc::clone(self), Rc::clone(other)];
        Operation::Add.apply(inputs)
    }
}
