# pytorch_ext
These packages try to minimize the amount of code you have to write when training a PyTorch model. \
**pytorch_ext** main component is the ModelTrainer class that, given model, data and training parameters takes care of the training loop for you. \
Furthermore the VisdomBoard package builds on top of [visdom](https://github.com/facebookresearch/visdom) to provide visual information about what is going on during training. As the name suggests it is intended to be a simplified version of TensorBoard.

### ModelTrainer
ModelTrainer avoids you to write a custom training loop each time you have to train a model. Given that each model and each training is different, ModelTrainer provides the training backbone and can be easily extended using callbacks to suit any specific needs. \
You are required to write only a small callable object that inherits from `pytorch_ext.model_trainer.TrainingCallback` that takes care of feeding data to the model and uses its output to compute the loss. 

```
import torch

from pytorch_ext.model_trainer import ModelTrainer
from pytorch_ext.model_trainer import TrainingCallback


class SupervisedTraining(TrainingCallback):

    def __call__(self, data_batch):
        data, label = data_batch
        output = self.trainer.model(data)
        return self.trainer.loss_fn(output, label)
        
        
criterion = torch.nn.MSELoss()
epochs = 10
optimizer = torch.optim.SGD()

trainer = ModelTrainer(criterion, epochs, optimizer)

model = create_some_fancy_model()
training_set = get_huge_dataset()

trainer.run(model, training_set, training_callback=SupervisedTraining())
```

That's it. `TrainingCallback` defines a `trainer` attribute that is used to access the trainer instance to which it is attached to. `trainer` cannot be used in the constructor though, so if you need to use `trainer` to initialize something you can do it overriding `TrainingCallback.on_attach()` method.

You can even define some ModelTrainer callback objects (MTCallback) to perform some actions during training. For example if you want your model to be periodically saved during training you could do something along these lines:

```
from pytorch_ext.model_trainer import MTCallback, Event


class Checkpoint(MTCallback):

    def __init__(self, checkpoint_dir: str):
        super(Checkpoint, self).__init__()
        self.event = Event.ON_EPOCH_END
        self.checkpoint_dir = checkpoint_dir

    def __call__(self):
        epoch = self.trainer.current_epoch
        model_path = os.path.join(self.checkpoint_dir, 'checkpoint_{}.pt'.format(epoch))
        torch.save(self.trainer.model, model_path)
        

trainer.attach_callback(Chackpoint('.'))
trainer.run(model, training_set, training_callback=SupervisedTraining())
```

To specify when the MTCallback object has to be called you must assign to `self.event` one of the values of `pytorch_ext.model_trainer.Event` 

To see the documentation and more examples take a look at model_training.model_trainer or model_training.callbacks.



### VisdomBoard 
   