# Papers Reference

This reference contains a list of papers helpful to deeper understand models and 
training techniques implemented in [fastai](https://github.com/fastai/fastai) 
library.

## Model training and parameters tuning

1. [A disciplined approach to neural network hyper-parameters: Part 1 - learning rate, batch size, momentum, and weight decay](https://arxiv.org/pdf/1803.09820.pdf) (Leslie N. Smith). 
Describes an educated process of selecting model's learning rate. Implemented 
as [fit_one_cycle](http://docs.fast.ai/train.html#fit_one_cycle-1) method which 
is an alias for [fit](http://docs.fast.ai/basic_train.html#fit-2) with 
[OneCycleScheduler](http://docs.fast.ai/callbacks.one_cycle.html#OneCycleScheduler) callback. The 
most interesting content starts at **Section 4**.