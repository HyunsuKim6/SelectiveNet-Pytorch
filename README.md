# SelectiveNet-Pytorch

The author of paper has uploaded code written in Keras, but I thought some people are familiar with Pytorch, so I implemented it in Pytorch.

## Requirements

You will need the following to run the above:
- Pytorch
- Python3, Numpy, Matplotlib, tqdm

Note that I run the code with Windows 10, Pytorch 0.4.1, CUDA 10.1

### Training
Use `train.py` to train the network. Example usage:
```bash
# Example usage
python train.py
```

### Testing
Use `test.py` to test the network. Example usage:
```bash
# Example usage
python test.py
```

## References

- [SelectiveNet: A Deep Neural Network with an Integrated Reject Option][1]
    - I referred to the SelectiveNet paper.

- [geifmany/selectivenet][2]
    - There is author's repository.

[1]: https://arxiv.org/abs/1901.09192
[2]: https://github.com/geifmany/selectivenet
