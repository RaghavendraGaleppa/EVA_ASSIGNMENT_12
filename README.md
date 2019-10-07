# Assignment 12



## <u>**mc.ai**</u>

- Goal is to get 94% accuracy on cifar10 in 2 minutes is the goal.

- cifar10 is a dataset of images of size 32x32 belonging to 10 classes.

- Uses DavidNet.

- They use fenwicks library which is custom written by David.

- You can install fenwicks library as :

  ```python
  if tf.gfile.Exists('./fenwicks'):
   tf.gfile.DeleteRecursively('./fenwicks')
  !git clone https://github.com/fenwickslab/fenwicks.git
  ```



### setting up

- First import the fenwicks library and setup its backend GPU.

```pythone
import fenwicks as fenwicks
fw.colab_tpu.setup_gcs()
```

- Set up hyperparameters

  ```python
  BATCH_SIZE = 512 #@param ["512", "256", "128"] {type:"raw"}
  MOMENTUM = 0.9 #@param ["0.9", "0.95", "0.975"] {type:"raw"}
  WEIGHT_DECAY = 0.000125 #@param ["0.000125", "0.00025", "0.0005"] {type:"raw"}
  LEARNING_RATE = 0.4 #@param ["0.4", "0.2", "0.1"] {type:"raw"}
  EPOCHS = 24 #@param {type:"slider", min:0, max:100, step:1}
  WARMUP = 5 #@param {type:"slider", min:0, max:24, step:1}
  
  BUCKET = 'gs://gs_colab'
  PROJECT = 'cifar10'
  ```

- These hyper parameter values are all taken from David net implementation.

- However weight decay used here is only quarter of David Net. Article says that weight decay of 0.005

  will underfit and the accuracy will come down to 90%, and the architectural difference between TPU and GPU is said to be cause of this.



### Preparing Data

-  Since the size of an image in cifar 10 is much larger than that of MNIST, putting entire cifar10 data in memory will lead to a memory error, therefore they are storing the data in GCS.
- Download the dataset. 
- Calculate the mean and std of the dataset. Then proceeds to subract the entire dataset by the mean and std.
- In tensorflow the preferred file format is TFRecord.
- The fenwicks library provides direct access to the TFRecord.



### Data Augmentation and Input Pipeling

- Pad 4 pixels to image and then random crop to 32x32.
- random flip left and right.
- Uses cuttout augmentation
- fenwicks library provides one liners for all these augmentations
- It uses again fenwicks library oneliners to build the input pipeling and other hyperparameters.
- since the original David Net was in PyTorch, the fenwicks help in weight initialization difference between PyTorch and Tensorflow



![Architecture](/home/raghavendragaleppa/Books/NEW.png)

- It seems that the final layer of the Network (i.e, the classifier) is multiplied by a scaling factor of 0.125.



### Model Training

- It uses SGD with nestrov momentum to train the model.
- Uses slanted trainglular LR schedule.
- The model is trained on a TPU instead of a GPU.



## <u>How to train your ResNet</u>

- Written by same guy, David Page
- Train your network in 341s in single GPU and 174s in 8 GPUs
- 8 Pages are written

### Baseline Model

- The goal here is to produce a baseline model for CIFAR10 in 6mins.
- He was able to build  a baseline model on AWSp3.2xlarge  with single V100 in to reach 94% accuracy in 356s.
- Once baseline is created you have to start looking for simple improvements that can be implemented right away![](/home/raghavendragaleppa/Books/Artboard-1-5.svg)
- The above image is the architecture used by a fast.ai student to train CIFAR10 in unber 341s
- You can remove one of the first two batchNorm-ReLU layers which are consecutive.

![](/home/raghavendragaleppa/2019-10-06-150516_1366x768_scrot.png)



- The kink at epoch 15 is also removed.
- With this the time was brought down to 323s.
- Some of the preprocessing(padding, normalisation, transposition) seems to be done at every iteration and being repeated every time. Fresh PyTorch processes are deployed to do these transformations. So it makes sense to do common work once before training, and this helps to keep the number of processes down. Doing this brings down the timing by another 15s. New training time is 308s.
- We can make the calls to random number generators 2 smaller and at the start of the epoch. This saves  another 7s, also make the num_workers as just 1 to keep the number of threads single, which saves another 4s. This all leads to a training time of 297s.
- This is link to the baseline model - https://github.com/davidcpage/cifar10-fast/blob/master/experiments.ipynb.



### Mini-Batches

- From the previous page we have reduced the training time to around 297s.
- Now increase the batch-size from 128 to 512. Larger batches means more efficient computation. And this brings the training time to 256s.



### Regularisation

- We can preload the GPU with random training data to remove data loading and transfer times.
- We can also leave the optimizer step, and the ReLU, and batch Norm layer to leave just the convolutions.

![img](https://296830-909578-raikfcquaxqncofqfm.stackpathdns.com/wp-content/uploads/2019/06/timing_breakdown.svg)

- We get the above rough breakdown of how much each of the operation time takes. A good amount of time is being spent on the Batch Norm part. The reason for this seems to be that the half precision mode for pytorch does not use an optimized CuDNN routine for Batch Norm process and hence slwoing down the computation. Convert the Batch Norm weights back to single precision and you get this :

![img](https://296830-909578-raikfcquaxqncofqfm.stackpathdns.com/wp-content/uploads/2019/06/timing_breakdown_b.svg)

- Now the training time to read 94% is reduced to 186s all on a single V100 GPU.

- The next step is to use CuttOut. Take random 8x8 out of the training images along with the standard data augmentation of padding, clipping and random flipping left-right.
- Accelerating the learning rate to 30 epochs, 4/5 times it reaches 94% acc in 30 Epochs, along with Increasing the batch size to 768.
- The timings for 30epochs run are 161s at batch size 512 and 154s at batch size 768 all on single GPU. Other hyper parameters  (momentum=0.9, weight decay=5e-4) are kept at their original values.
- https://github.com/davidcpage/cifar10-fast/blob/master/experiments.ipynb - The code to reproduce this.



### Architecture

- Here the main goal is to simplify the Architecture and make the network a lot faster.

  ![img](https://296830-909578-raikfcquaxqncofqfm.stackpathdns.com/wp-content/uploads/2019/06/schematic.svg)

- The above is the current network architecture.

![img](https://296830-909578-raikfcquaxqncofqfm.stackpathdns.com/wp-content/uploads/2019/06/residual_block.svg)

- This is the pink Block. It has identity shortcut and preserves the spatial and channel dimensions of the input.

![img](https://296830-909578-raikfcquaxqncofqfm.stackpathdns.com/wp-content/uploads/2019/06/downsampling_block.svg)

- This is the light green Block. It is for downsampling the image spatial size by a factor of 2 and double the output channels.
- Eliminating the branches of the network, we get:

![img](https://296830-909578-raikfcquaxqncofqfm.stackpathdns.com/wp-content/uploads/2019/06/shortest_route.svg)

- Training this network with accelerated LR schedule, yeilds a 55.9% test acc in 36s.
- The problem with this backbone network is  downsampling convolutions have a 1x1 kernel and a stride of 2, so rather than enlarging the receptive field of the network, it's just discarding the information. Replacing 1x1 by 3x3 convolutions, test acc improves to a 85.6% time in 36s.
- We can further improve the training accuracy by applying 3x3 convolutions with a stride of one followed a max pooling of (2x2). This leads to a test acc of 89.7% in 43s.
- The final pooling layer before the classifier is a concatenation of global average pooling and max pooling layers, inherited from the original network.Replace this with a more standard global max pooling layer and double the output dimension of the final convolution to compensate for the reduction in input dimension to the classifier, leading to a final test accuracy of 90.7% in 47s.

- Now use two different types of networks

- Class A:

  ![img](https://296830-909578-raikfcquaxqncofqfm.stackpathdns.com/wp-content/uploads/2019/06/extraL2.svg)

- Class B:

  ![img](https://296830-909578-raikfcquaxqncofqfm.stackpathdns.com/wp-content/uploads/2019/06/extraL2.svg)



- Now each of these two classes are then seperately classified into seven different categories, based on adding of convolutiosn with bn-relu and position of pooling layers.
- Based on brute force search space test, they choose the Residual Network with L1 + L3  which achieves 93.8% test acuracy in 66s for a 20 epoch run. Extending it to 24 epochs gets the training acc to 94% in 7/10 times in just 79s.
- It seems the residual networks (Class B) outperform Extra Networks(Class A).



### Batch Normalisation

- https://colab.research.google.com/github/davidcpage/cifar10-fast/blob/master/batch_norm_post.ipynb - The code for this part
- The learnable parameters of network have some functions that produce a constant output ignoring the input.
- First order optimizers such as SGD do not like these constraints.

- Batch Norm allows the network to train much more rapidly for high learning rates.

  ![img](https://296830-909578-raikfcquaxqncofqfm.stackpathdns.com/wp-content/uploads/2019/06/Artboard-1-7-1-1.svg)

  

- As the LR is increased exponentially, the training acc for the batch norm one seems stable, whereas the one with no batch norm is dead.
- However there are many drawbacks of Batch Norm.
- It is slow.
- It is not much effective for smaller batch sizes.
- ReLU introduces non-zero mean channels, which leads to them approximating constant functions regardless of the input.
- Since ReLU only returs positive values, output channels all have a positive mean. But if we average over the weights, the effect disappears.



### 8 bag of tricks

- Use Test Time Augmentation(TTA).
- Use Mixed-precision numbers
- Move MaxPool Layers so that , Conv->Pooling->BatchNorm->Relu.
- Use Label Smoothing
- Use of CELU Activation function instead of Relu
- Ghost BatchNorm. Instead of applying BatchNorm on entirely of batch size 512 items, use bursts of 32 ietms and apply Batch Norm on them.
- Remove input channel correlation by using PCA.
