# Some useful tips for PyTorch and Python

## On how to use Tensorboard

#### Official PyTorch docs and tutorial

1. `torch.utils.tensorboard` class [documentation](https://pytorch.org/docs/stable/tensorboard.html#torch-utils-tensorboard)
2. Pytorch recipes [*How to use TensorBoard with PyTorch*](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html#how-to-use-tensorboard-with-pytorch)
3. Tutorial [*Visualizing Models, Data, and Training with TensorBoard*](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html#visualizing-models-data-and-training-with-tensorboard)
4. Tutorial [*PyTorch Profiler With TensorBoard*](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html#pytorch-profiler-with-tensorboard)

#### Other
1. TensorboardX [github forked page](https://github.com/ywang037/tensorboardX).
2. Chinese tutorial [Pytorch使用tensorboardX可视化。超详细！！！](https://www.jianshu.com/p/46eb3004beca)
3. Chinese tutorial https://zhuanlan.zhihu.com/p/35675109 and https://my.oschina.net/u/4392911/blog/3227975

## How to remove persistant cached files when `.gitignore` not working
1. Install git and open gitbash, 
2. then direct to the folder, and use `git rm --cached XXXX`, 
3. then go to github desktop to commit this change. 

It should be OK after this (no matter the relevant folder has been added to `.gitignore` or doing this after perform the above operation).
https://github.com/Microsoft/vscode/issues/40742