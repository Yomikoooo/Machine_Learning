# Machine Learning
This is a repo of my machine learning study.
It includes my learning path, hw and proj of the opencourse, and the conclusive notes of my domain knowledge of ML and DL.


- [Machine Learning](#machine-learning)
  - [My Learning Process:](#my-learning-process)
    - [Programming:](#programming)
    - [Books:](#books)
    - [Math foundation:](#math-foundation)
    - [OpenCourses:](#opencourses)
    - [Projects in kaggle:](#projects-in-kaggle)
    - [Papers:](#papers)
- [Notes](#notes)
  - [Online notes](#online-notes)
  - [Reading materials:](#reading-materials)
  - [0. Basics](#0-basics)
    - [Intro:](#intro)
    - [0.1. Data processing:](#01-data-processing)
    - [0.2. Model:](#02-model)
    - [0.3. Loss function](#03-loss-function)
      - [0.3.1. MLE](#031-mle)
      - [0.3.2. MAE](#032-mae)
      - [0.3.3. Cross Entropy](#033-cross-entropy)
    - [0.4. Optimization algorithm:](#04-optimization-algorithm)
      - [0.4.1. Gradient Descent](#041-gradient-descent)
      - [0.4.2. Newton's Method](#042-newtons-method)
    - [0.5. Data Loader](#05-data-loader)
  - [1. Training and Fitting](#1-training-and-fitting)
    - [1.1. Overfitting and Underfitting](#11-overfitting-and-underfitting)
    - [1.2. Regularization](#12-regularization)
    - [1.3. Data Augmentation](#13-data-augmentation)
    - [1.4. Early Stopping](#14-early-stopping)
    - [1.5. Dropout](#15-dropout)
    - [1.5. Trade-off between bias and variance](#15-trade-off-between-bias-and-variance)
    - [1.6. Cross Validation](#16-cross-validation)
    - [1.7 Model Selection](#17-model-selection)
  - [2. Optimization](#2-optimization)
    - [2.1. Gradient Descent](#21-gradient-descent)
      - [2.1.1. Batch Gradient Descent](#211-batch-gradient-descent)
      - [2.1.2. Stochastic Gradient Descent](#212-stochastic-gradient-descent)
      - [2.1.3. Mini-batch Gradient Descent](#213-mini-batch-gradient-descent)
    - [2.2. Momentum](#22-momentum)
    - [2.3. AdaGrad](#23-adagrad)
    - [2.4. Adam](#24-adam)
    - [2.5. RMSProp](#25-rmsprop)
  - [3. Linear Regression](#3-linear-regression)
  - [4. Classification](#4-classification)
    - [4.1. Logistic Regression](#41-logistic-regression)
    - [4.2. Perceptron learning aigorithm](#42-perceptron-learning-aigorithm)
  - [5. Generalized Linear Model](#5-generalized-linear-model)
    - [5.1 Exponential Family](#51-exponential-family)
    - [5.2. Generalized Linear Model](#52-generalized-linear-model)
    - [5.3 Softmax Regression](#53-softmax-regression)
  - [6. Generative learning algorithms](#6-generative-learning-algorithms)
    - [6.1. Generative vs. Discriminative](#61-generative-vs-discriminative)
    - [6.2 Gaussian Discriminant Analysis](#62-gaussian-discriminant-analysis)
    - [6.3. Naive Bayes](#63-naive-bayes)
  - [7. Support Vector Machine](#7-support-vector-machine)
    - [7.1 Margin](#71-margin)
    - [7.2 Lagrange Duality](#72-lagrange-duality)
    - [7.3 KKT Conditions](#73-kkt-conditions)
    - [7.4 SVM Optimization steps](#74-svm-optimization-steps)
    - [7.5 Kernel Methods](#75-kernel-methods)


## My Learning Process:
**python libraries**(pytorch, numpy, pandas, matplotlib) $\rightarrow$ **opencourse**(lec and hw and proj) $\rightarrow$ **books**(including math) $\rightarrow$ **projects**(find in github or kaggle) $\rightarrow$ **papers**(top conf and research group)

**Detailed**: assuming math preliminary is equipped(at least calculus, linear algebra, probability theory and some mathematical statistics basics, if not sure how many to learn, go to book `PRML` or `Prob ML`), some crash tutorial of `python libraries`(official tutorial is enough), then go to `CS229(or NTU ML2023)` for machine learning basics, after that go to `CS231n` for deep learning basics and computer vision, then go to `CS224n` for natural language processing basics. 
If research, read papers in `top conferences` and `research groups`.
If find jobs, go to `kaggle` to do some projects.

TOP conf: ICML, NIPS, ICLR, CVPR, ICCV, etc. 
Research groups: Google Research, Google Brain, Deepmind, Meta AI, Tencent AI Lab, etc.
Mu Li is has his youtube channel to lead me to read important papers.
If not enough, RSS feed is useful.

### Programming:
| Name | tutorials | docs |
| :--: | :--: | :--: |
| python | 
| pytorch | [Official](https://pytorch.org/tutorials/) | [Official](https://pytorch.org/docs/stable/index.html) |
| numpy |
| pandas |
| matplotlib |

### Books:
| Name | Author | Code | Note |
| :--: | :--: | :--: | :--: |
| Statistical Learning Method |Hang Li|[repo1](https://github.com/SleepyBag/Statistical-Learning-Methods),[repo2](https://github.com/fengdu78/lihang-code) | |
| Pattern Recognition and Machine Learning | Christopher M. Bishop | 
| Deep Learning |Ian Goodfellow|
| Machine Learning: A Probabilistic Perspective | Kevin P. Murphy |

### Math foundation:
| Field | book name | author | opencourse |
| :--: | :--: | :--: | :--: |
| Linear Algebra | Linear Algebra | Gilbert Strang | [MIT 18.06](http://web.mit.edu/18.06/www/) |
| Calculus | Thomas' Calculus | George B. Thomas, Maurice D. Weir, Joel Hass, Frank R. Giordano | [MIT 18.01](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/syllabus/)</br>[MIT 18.02](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/) |
| Differential Equations | Elementary Differential Equations and Boundary Value Problems | William E. Boyce, Richard C. DiPrima | [MIT 18.03](https://ocw.mit.edu/courses/mathematics/18-03sc-differential-equations-fall-2011/) |
| Probability Theory | Introduction to Probability | Dimitri P. Bertsekas, John N. Tsitsiklis | [MIT 6.041](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/)
| Statistics | All of Statistics | Larry Wasserman | [CMU 36-705](http://www.stat.cmu.edu/~larry/=stat705/) |
| Optimization | Convex Optimization | Stephen Boyd, Lieven Vandenberghe | [Stanford EE364a](https://stanford.edu/class/ee364a/)|
| Information Theory | Elements of Information Theory | Thomas M. Cover, Joy A. Thomas | [MIT 6.441](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/pages/syllabus/) |
| Numerical Analysis | Numerical Analysis | Richard L. Burden, J. Douglas Faires | [MIT 18.330](https://ocw.mit.edu/courses/mathematics/18-330-introduction-to-numerical-analysis-spring-2012/) |
| Signal Processing | Digital Signal Processing | Alan V. Oppenheim, Ronald W. Schafer, John R. Buck | [MIT RES 6-007](https://ocw.mit.edu/courses/res-6-007-signals-and-systems-spring-2011/)</br>[MIT 6.341](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-341-discrete-time-signal-processing-fall-2005/) |
| Image Processing | Digital Image Processing | Rafael C. Gonzalez, Richard E. Woods, Steven L. Eddins | 

### OpenCourses:
| Name | repo |Author | Field | Code | Note |
| :--: | :--: | :--: | :--: | :--: | :--: |
|  [Machine Learning](https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php) |  |Hung-Yi Lee| DL
| [CS229](http://cs229.stanford.edu/) | [repo](https://github.com/Yomikoooo/cs229-autumn-2018) |Andrew Ng | ML |
| [CS231n](http://cs231n.stanford.edu/) |  |Fei-Fei Li | CV
| [CS224n](http://web.stanford.edu/class/cs224n/) |  |Christopher Manning| NLP

### Projects in kaggle:
| Name |  Task | Code | Note |
| :--: |  :--: | :--: | :--: |

### Papers:
| Name | Author | Field | Code | Note |
| :--: | :--: | :--: | :--: | :--: |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin | NLP | 



# Notes
## Online notes
|Chapter|Link(Chinese)|Task
|:--|:--|:--|
|0. Basics| [bias and variance](https://zhuanlan.zhihu.com/p/38853908), [Regularization](https://www.zhihu.com/question/20924039/answer/240037674)
|1. Linear Regression
|2. Logistic Regression|[Zhihu](https://zhuanlan.zhihu.com/p/74874291)|classification
|3. Generalized Linear Models
|4. Gaussian Discriminant Analysis|[Zhihu](https://zhuanlan.zhihu.com/p/36982945)|classification
|5. Naive Bayes|[Zhihu](https://www.zhihu.com/question/265995680/answer/303148257)|classification
|6. Support Vector Machines|[Zhihu](https://zhuanlan.zhihu.com/p/77750026)|classification
|7. Kernel Methods|[Zhihu](https://www.zhihu.com/question/24627666/answer/28460490)|SVM trick
|8. Decision Trees|[Zhihu](https://zhuanlan.zhihu.com/p/85731206)|classification
|9. Random Forest|
|10. Boosting Methods|
|11. PCA|


## Reading materials:
[Full stack deep learning](https://zhuanlan.zhihu.com/p/218468169)
[Full stack LLM bootcamp](https://zhuanlan.zhihu.com/p/629589593)

## 0. Basics

### Intro:
Machine Learning is finding a function $f$ that maps input $x$ to output $y$.

some kinds of tasks: speech recognition, image classification, machine translation, etc.
|Types|Description|Task| Model|
|:--|:--|:--|:--|
|Supervised learning|features and labels |classification, regression | SVM, decision tree, random forest, neural network
|Unsupervised learning|features only | clustering, dimensionality reduction, density estimation, anomaly detection, compression, embedding| K-means, PCA, autoencoder, GAN
|**Deep leaning**|learning by deep neural network| | CNN, RNN, LSTM, Transformer
|Reinforcement learning|learning by reward and punishment| | Q-learning, policy gradient

**Frequentist vs Baysian**:
Frequentist: $P(A)$ is the frequency of $A$ in the long run.
Baysian: $P(A)$ is the degree of belief in $A$.

**Baysian inference**: $$P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)}$$
which means "posterier = likelihood * prior / evidence"

MLE(Maximum Likelihood Estimation): $$\hat\theta_{MLE} = argmax_\theta P(D|\theta)$$
MAP(Maximum a Posterior): $$\hat\theta_{MAP} = argmax_\theta P(D|\theta)P(\theta)$$
where $P(D|\theta)$ is the likelihood, $P(\theta)$ is the prior, $P(D)$ is the evidence, $P(\theta|D)$ is the posterier.

where $D$ is the data, $\theta$ is the parameter.

### 0.1. Data processing:

some python libraries to be used:
```python
#Numerical Operations
import numpy as np #matrix
import math

#Data Processing
import pandas as pd #dataframe
import os #path
import csv #csv

#Visualization
import matplotlib.pyplot as plt #plot
import seaborn as sns #plot

#Progress Bar
from tqdm import tqdm

#PyTorch
import torch #framework
import torch.nn as nn # deep learning
import torch.optim as optim # optimization
import torch.nn.functional as F # activation function
from torch.utils.data import DataLoader, Dataset # data loader
from torch.utils.tensorboard import SummaryWriter # tensorboard
```
**CUDA(GPU)** is used to speed up the training process.
why not use CPU, because CPU is not designed for parallel computing.
```python
#CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
```
**batch** is used to speed up the training process because of the parallel computing.

**Download** the dataset from [kaggle](https://www.kaggle.com/c/digit-recognizer/data) or [MNIST](http://yann.lecun.com/exdb/mnist/) or [UCI Archive](https://archive.ics.uci.edu/ml/index.php) to the folder `./data/`.
the dataset is usually in the format of `.csv` or `.zip`.

```bash
#Download dataset from internet, in jupyter notebook or colab and rename to train.csv
!wget -O ./data/train.csv <link>
```



example: $y=b+wx$

$y$ is the to be predicted, $x$ is the feature, $w$ is the weight, $b$ is the bias.

the $w$ and $b$ are the parameters of the model. **training data $(x_i , y_i)$** is used to find the real parameters. and the **testing data $(x_j, y_j)$** is used to evaluate the model.

define the **dataset**:
```python
#dataset
class MyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data #data is a list of tuples
        self.transform = transform #transform is a function to

    def __len__(self):
        return len(self.data) #return the length of the dataset

    def __getitem__(self, idx): #return the idx-th sample
        if torch.is_tensor(idx): #convert idx to list
            idx = idx.tolist()
        sample = self.data[idx] #get the idx-th sample
        if self.transform: #transform the sample
            sample = self.transform(sample) #
        return sample
```

we need to **split** the data into training data and testing data. but we can utilize the data, we don't need to directly split the data. we can use **cross validation** to evaluate the model.
```python
#split the data with sklearn
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=0)

#split not use sklearn
def split_data(data, ratio=0.8):
    train_size = int(len(data) * ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

#Cross Validation with sklearn
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=0)
for train_index, test_index in kf.split(data):
    train_data = data[train_index]
    test_data = data[test_index]

#Cross Validation not use sklearn
def cross_validation(data, k=5):
    data = np.array(data)
    np.random.shuffle(data)
    data = np.array_split(data, k)
    for i in range(k):
        test_data = data[i]
        train_data = np.concatenate(data[:i] + data[i+1:])
        yield train_data, test_data

#print out the data size
print(f"""
train_data: {train_data.shape}
test_data: {test_data.shape}
""")

#

```
if not all features are useful, we need **feature selection**.
```python
#feature selection
def select_feature(train_data,test_data,select_all=True)
    y_train = train_data[:,0]
    x_train = train_data[:,1:]
    y_test = test_data[:,0]
    x_test = test_data[:,1:]

    if select_all:
        return train_data, test_data
    else:
        return train_data[:,[0,1,2,3,4,5,6,7,8,9]], test_data[:,[0,1,2,3,4,5,6,7,8,9]]
    

```

### 0.2. Model: 
If machine learning, we use linear regression, logistic regression, SVM, etc.
If deep learning, we use neural network, CNN, ResNet, etc.
+ linear regression is a special case of neural network(only one layer)
+ in neural network, we use **activation function** to make the model non-linear. every neuron has an activation function.
  + sigmoid: $$\sigma(x) = \frac{1}{1+e^{-x}}$$
  + tanh: $$tanh(x) = \frac{e^x-e^{-x}}{e^x+e^{-x}}$$
  + ReLU: $$ReLU(x) = max(0, x)$$
  + Leaky ReLU: $$LeakyReLU(x) = max(0.01x, x)$$
  + Softmax: $$Softmax(x_i) = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}$$
```python
#example: linear regression
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

#example: neural network
class NeuralNetwork(nn.Module):
    def __init__(self, config['input_dim'], config['hidden_dim'], config['output_dim']):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(config['input_dim'], config['hidden_dim']) #input layer
        self.relu = nn.ReLU() #activation function
        self.fc2 = nn.Linear(config['hidden_dim'], config['output_dim']) #output layer

    def forward(self, x): #forward propagation
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
```


### 0.3. Loss function
is used to measure the difference between the predicted value and the real value.

some common loss functions: MAE, MSE, Cross Entropy, etc.

#### 0.3.1. MLE
**MSE:**
$$J(\theta) = \frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2$$
probablistic interpretation: It can be derived from the **maximum likelihood estimation (MLE)** of the Gaussian distribution. 
linear algebra interpretation: It can be derived from the **least square estimation (LSE)** of the linear system.

#### 0.3.2. MAE
**MAE:**
$$J(\theta) = \frac{1}{m}\sum_{i=1}^m|h_\theta(x^{(i)})-y^{(i)}|$$
probablistic interpretation: It can be derived from the **maximum likelihood estimation (MLE)** of the Laplace distribution.

#### 0.3.3. Cross Entropy
**Cross Entropy:**
$$J(\theta) = -\frac{1}{m}\sum_{i=1}^m\sum_{k=1}^Ky_k^{(i)}log(h_\theta(x^{(i)}))_k$$
where $y_k^{(i)}$ is the one-hot encoding of the real label, $h_\theta(x^{(i)})_k$ is the predicted probability of the $k$-th class.
cross entropy is derived from the **maximum likelihood estimation (MLE)** of the multinomial distribution.

```python
#example: MSE, Cross Entropy, L1, L2
# criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
# criterion = nn.L1Loss()
# criterion = nn.L2Loss()
loss = criterion(y_pred, y)
```

### 0.4. Optimization algorithm:
to find the real parameters, we need to minimize the loss function

some common optimization algorithms: **Gradient Descent(SGD)**, Newton's Method, Adam, Adagrad, etc.

#### 0.4.1. Gradient Descent
**Gradient Descent** is a first-order iterative optimization algorithm for finding the minimum of a function. To find a local minimum of a function using gradient descent, one takes steps proportional to the negative of the gradient (or approximate gradient) of the function at the current point.
$$\theta_j := \theta_j - \alpha\frac{\partial}{\partial\theta_j}J(\theta)$$
where $\alpha$ is the learning rate., $\frac{\partial}{\partial\theta_j}J(\theta)$ is the partial derivative of the loss function with respect to $\theta_j$.

GD equals **minimizing** the **negative** log likelihood function in MLE if we set the likelihood function as loss function.

the problem of GD is that it is very slow when the number of features is large. So we need to use **feature scaling** to speed up the training process.
$$x_j := \frac{x_j-\mu_j}{s_j}$$
where $\mu_j$ is the mean of the feature $x_j$, $s_j$ is the standard deviation of the feature $x_j$.
in general, we need to normalize the features to the range of $[-1, 1]$.

in gradient landscape, local minimum is not guaranteed. So we need to use **multiple start points** to find the **global minimum**. 

**stochastic gradient descent (SGD)** is a variant of GD. it randomly selects a sample from the training set to update the parameters. it is faster than GD but it is not stable. mini-batch gradient descent is a variant of SGD. it randomly selects a mini-batch from the training set to update the parameters. it is faster than GD and more stable than SGD.

**momentum** is a variant of GD. it uses the previous gradients to update the parameters. it is faster than GD and more stable than SGD and mini-batch GD.

#### 0.4.2. Newton's Method
**Newton's Method** is a second-order iterative optimization algorithm for finding the minimum of a function. It is faster than GD but it is not stable. especially when the number of features is large, it is very slow because it needs to calculate the inverse of the Hessian matrix.
$$\theta := \theta - H^{-1}\nabla_\theta J(\theta)$$
where $H$ is the Hessian matrix of the loss function with respect to $\theta$.

**learning rate**: the step size of the gradient descent.
**momentum**: the weight of the previous gradient.

```python
# example: SGD, Adam, Adagrad, etc.
optimizer = optim.SGD(model.parameters(), config['lr'], config['momentum'])
#optimizer = optim.Adam(model.parameters(), lr=0.01)
#optimizer = optim.Adagrad(model.parameters(), lr=0.01)
```

**Hyperparameters**: learning rate, momentum, batch size, etc.

**manual tuning**: try different hyperparameters and choose the best one.
we can use configuration file to store the hyperparameters.
```python
#configuration
config = {
    'input_dim': 28*28,
    'hidden_dim': 100,
    'output_dim': 10,
    'num_epochs': 500,
    'batch_size': 64,
    'lr': 0.01,
    'momentum': 0.9
    'select_all': True

}
```
### 0.5. Data Loader

before training, we need to create a **data loader** to load the data.
```python
#data loader
train_loader = DataLoader(dataset=train_data, batch_size=config['batch_size'], shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=config['batch_size'], shuffle=False)

```

Now we have the model, the loss function and the optimization algorithm. We can start to train the model.

```python
#training and validation
def train(model, train_loader, num_epochs, criterion, optimizer, config):
    if not os.path.exists('models'):
        os.mkdir('models') #create a folder to store the model

    writer = SummaryWriter() #create a tensorboard writer
    n_epochs = config['num_epochs']
    best_loss = math.inf #initialize the best loss
    step = 0 #initialize the step
    #training loop
    for epoch in range(num_epochs):
        model.train() #set the model to training mode
        train_loss_list = [] #store the training loss of each epoch
        train_progress_bar = tqdm(train_loader) #create a progress bar
        for images, labels in train_progress_bar:
            images = images.view(-1, 28*28) 
            labels = labels.to(device)
            outputs = model(images) #forward propagation
            loss = criterion(outputs, labels)
            optimizer.zero_grad() #clear the gradient
            loss.backward() #back propagation
            optimizer.step() #update the parameters
            train_loss_list.append(loss.item()) #store the training loss
            train_progress_bar.set_description(f'Epoch: {epoch+1}/{num_epochs}, Step: {step}, Loss: {loss.item():.4f}') #show the training loss
            step += 1 #update the step
        writer.add_scalar('train_loss', np.mean(train_loss_list), epoch) #write the training loss to tensorboard
        
        #validation
        model.eval() #set the model to evaluation mode
        val_loss_list = [] #store the validation loss of each epoch
        val_progress_bar = tqdm(test_loader) #create a progress bar
        for images, labels in val_progress_bar:
            images = images.view(-1, 28*28)
            labels = labels.to(device)
            outputs = model(images) #forward propagation
            loss = criterion(outputs, labels)
            val_loss_list.append(loss.item()) #store the validation loss
            val_progress_bar.set_description(f'Epoch: {epoch+1}/{num_epochs}, Step: {step}, Loss: {loss.item():.4f}') #show the validation loss
        writer.add_scalar('val_loss', np.mean(val_loss_list), epoch) #write the validation loss to tensorboard
        #save the best model
        if np.mean(val_loss_list) < best_loss:
            best_loss = np.mean(val_loss_list)
            torch.save(model.state_dict(), 'best_model.pth')
    writer.close() #close the tensorboard writer
```
plot the loss curve
```python
#plot the loss curve
def plot_loss_curve(loss_list):
    plt.plot(loss_list)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.show()
```

```python
#save the prediction and download the model
def save_prediction(model, test_loader):
    model.eval()
    prediction = []
    for images, labels in test_loader:
        images = images.view(-1, 28*28)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        prediction.extend(predicted.tolist())
    with open('prediction.csv', 'w') as f:
        f.write('ImageId,Label\n')
        for i, p in enumerate(prediction):
            f.write(f'{i+1},{p}\n')
    files.download('prediction.csv')
    files.download('best_model.pth')
```
## 1. Training and Fitting
### 1.1. Overfitting and Underfitting
overfitting: the model performs well on the training data but performs poorly on the test data.
underfitting: the model performs poorly on both the training data and the test data.
### 1.2. Regularization
Ruegularization is a technique to prevent overfitting.
The simplest explanation of regularization is that we add a penalty term to the loss function. It looks like we add a prior probability to the loss function.
The regularization term is usually the norm of the parameters.

**L1** regularization is also called **Lasso** regularization.
$$L1 = \lambda\sum_{j=1}^n|\theta_j|$$
example: L1 in linear regression
$$L = \frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2 + \lambda\sum_{j=1}^n|\theta_j|$$

**L2** regularization is also called **Ridge** regularization.
$$L2 = \lambda\sum_{j=1}^n\theta_j^2$$
example: L2 in linear regression
$$L = \frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2 + \lambda\sum_{j=1}^n\theta_j^2$$
### 1.3. Data Augmentation

### 1.4. Early Stopping


### 1.5. Dropout
### 1.5. Trade-off between bias and variance
### 1.6. Cross Validation
### 1.7 Model Selection

## 2. Optimization
### 2.1. Gradient Descent
#### 2.1.1. Batch Gradient Descent
#### 2.1.2. Stochastic Gradient Descent
#### 2.1.3. Mini-batch Gradient Descent
### 2.2. Momentum
### 2.3. AdaGrad
### 2.4. Adam
### 2.5. RMSProp

## 3. Linear Regression
**Cases**: Boston Housing Price Prediction, Stock Price Prediction, etc.
**Model: Linear Regression**
$$h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + ... + \theta_nx_n= \vec{\theta}^T\hat x$$
The hypothesis is a linear function of the input features $\hat x$. the parameters $\theta$ are the weights of the features. the model is the parameters $\vec{\theta}^T$.

**Loss Function: MSE** 
**Optimization: Gradient Descent**

**Normal Equation**
$$\theta = (X^TX)^{-1}X^Ty$$
derivation: $\nabla_\theta J(\theta) = X^TX\theta - X^Ty = 0$

**Locally Weighted Linear Regression**
$$J(\theta) = \frac{1}{2m}\sum_{i=1}^mw^{(i)}(h_\theta(x^{(i)})-y^{(i)})^2$$
## 4. Classification
### 4.1. Logistic Regression
**Description**: Logistic Regression is a linear classifier. it is used to classify data into two classes. it is a binary classifier. it can only classify linearly separable data.

**Cases**: Email Spam Detection, Credit Card Fraud Detection, etc.
**Model: Sigmoid Function**
$$h_\theta(x) = \frac{1}{1+e^{-\theta^Tx}}$$
**Loss Function: Cross Entropy**
**Optimization: Gradient Descent**

### 4.2. Perceptron learning aigorithm
this is the simplest neural network. it is a linear classifier. it can only classify linearly separable data.

**Model: Perceptron**
$$h_\theta(x) = \begin{cases}1 & \text{if } \theta^Tx \geq 0\\0 & \text{otherwise}\end{cases}$$

**Loss Function: 0-1 Loss**
$$L(\theta) = \sum_{i=1}^m\mathbb{1}\{y^{(i)}\neq h_\theta(x^{(i)})\}$$
**Optimization: Gradient Descent**

## 5. Generalized Linear Model
special cases: Linear Regression, Logistic Regression, Perceptron, etc. broader family of models called Generalized Linear Model(**GLMs**).
### 5.1 Exponential Family
**Description**: Exponential Family is a family of probability distributions. it includes many common distributions, such as Gaussian distribution, Bernoulli distribution, Poisson distribution, etc. in GLMs, the output distribution is a member of the exponential family.
**Model: Exponential Family**
$$p(y;\eta) = b(y)exp(\eta^TT(y)-a(\eta))$$
where $\eta$ is the natural parameter, $T(y)$ is the sufficient statistic, $a(\eta)$ is the log partition function, $b(y)$ is the base measure.

example: Bernoulli distribution
$$p(y;\phi) = \phi^y(1-\phi)^{1-y}$$
where $\phi$ is the probability of success, $y\in\{0,1\}$.
parameter: $\eta = log\frac{\phi}{1-\phi}$, $T(y) = y$, $a(\eta) = log(1+e^\eta)$, $b(y) = 1$.
example: Gaussian distribution
$$p(y;\mu) = \frac{1}{\sqrt{2\pi\sigma^2}}exp(-\frac{(y-\mu)^2}{2\sigma^2})$$
where $\mu$ is the mean, $\sigma^2$ is the variance.
parameter: $\eta = \frac{\mu}{\sigma^2}$, $T(y) = y$, $a(\eta) = \frac{\eta^2}{2}$, $b(y) = \frac{1}{\sqrt{2\pi}}$.

exponential family has many nice properties. for example, the mean and variance of exponential family can be expressed as a function of the natural parameter $\eta$ and its derivatives while some other distributions can not or integral is required. 
$$E[T(y)] = \frac{\partial a(\eta)}{\partial\eta}$$
$$Var[T(y)] = \frac{\partial^2 a(\eta)}{\partial\eta^2}$$

### 5.2. Generalized Linear Model
**Description**: GLM and exponential family are closely related. We just need to estiamte the natural parameter $\eta$ of the exponential family. the natural parameter $\eta=\theta^Tx$ is a linear function of the input features $x$.
our goal is to predict the expected value of $T(y)$, which is the mean of the output distribution.

### 5.3 Softmax Regression
**Description**: Softmax Regression is a generalization of Logistic Regression. it is a multi-class classifier. it can only classify linearly separable data.

|Distribution|$\eta$| $T(y)$| $a(\eta)$| $b(y)$|
|---|---|---|---|---|
|Bernoulli|$\theta^Tx$|$y$|$log(1+e^\eta)$|1|
|Gaussian|$\theta^Tx$|$y$|$\frac{\eta^2}{2}$|$\frac{1}{\sqrt{2\pi}}$|
|Multinoulli|$\theta^Tx$|$1\{y^{(i)}=k\}$|$log(\sum_{k=1}^Ke^{\eta_k})$|1|
|Poisson|$\theta^Tx$|$y$|$e^\eta$|$e^{-y}$|


**Model: Softmax Function**
$$h_\theta(x) = \frac{e^{\theta_j^Tx}}{\sum_{i=1}^ke^{\theta_i^Tx}}$$
**Loss Function: Cross Entropy**
**Optimization: Gradient Descent**

## 6. Generative learning algorithms
### 6.1. Generative vs. Discriminative
Bayes' rule
$$p(y|x) = \frac{p(x|y)p(y)}{p(x)}$$

|   |Generative model|Discriminative model|
|---|---|---|
|Goal | Directly estimate $p(y\|x)$ | estimate $p(y\|x)$ to then deduce $p(x|y)$ |
|what to learn| $p(x\|y)$ and $p(y)$| $p(y\|x)$ or $h_\theta(x)$|
|example| Naive Bayes, LDA, GDA| Logistic Regression, SVM, Neural Network|

### 6.2 Gaussian Discriminant Analysis
**Description**: Gaussian Discriminant Analysis is a generative learning algorithm. in GDA, the output distribution is Gaussian distribution. it is a binary classifier.

Gaussian and logistic regression is closely related. if the covariance matrix $\Sigma$ is diagonal, then GDA is equivalent to logistic regression.

**Multivariate Gaussian Distribution**
$$p(x;\mu,\Sigma) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))$$
where $\mu$ is the mean vector, $\Sigma$ is the covariance matrix.
$Cov[Z] = E[(Z-E[Z])(Z-E[Z])^T] = E[ZZ^T] - E[Z]E[Z]^T = E[ZZ^T] - \mu\mu^T$
If $Z\sim N(\mu,\Sigma)$, then $E[Z] = \mu$, $Cov[Z] = \Sigma$.

**Model: Gaussian Discriminant Analysis**
$$p(y) = Bernoulli(\phi)$$
$$p(x\|y=0) = N(\mu_0,\Sigma)$$
$$p(x\|y=1) = N(\mu_1,\Sigma)$$
**Loss Function: Maximum Likelihood Estimation**
log likelihood function
$$l(\phi,\mu_0,\mu_1,\Sigma) = log\prod_{i=1}^mp(x^{(i)},y^{(i)};\phi,\mu_0,\mu_1,\Sigma)$$
argmax: $\phi,\mu_0,\mu_1,\Sigma$
$$\phi = \frac{1}{m}\sum_{i=1}^m1\{y^{(i)}=1\}$$
$$\mu_0 = \frac{\sum_{i=1}^m1\{y^{(i)}=0\}x^{(i)}}{\sum_{i=1}^m1\{y^{(i)}=0\}}$$
$$\mu_1 = \frac{\sum_{i=1}^m1\{y^{(i)}=1\}x^{(i)}}{\sum_{i=1}^m1\{y^{(i)}=1\}}$$
$$\Sigma = \frac{1}{m}\sum_{i=1}^m(x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^T$$
**Optimization: Gradient Descent**

**Advantages**: GDA just computes the mean and covariance matrix of the input features. it is easy to implement and works well in practice.

### 6.3. Naive Bayes
**Description**: Our motivating examples is spam detection and text classification. we can use Naive Bayes to solve this problem. 
In NB, the core assumption is that the features are conditionally independent given the class label but it is impossible in reality. however, NB still works well in practice.
The model uses the prior probability $p(y)$ of given class and the conditional probability $p(x|y)$ of given class to predict the output distribution $p(y|x)$ and classify.

**Model: Naive Bayes**
$$P(x\|y) = \prod_{j=1}^np(x_j\|y)$$
Solutions: $$p(y) = \frac{\sum_{i=1}^m1\{y^{(i)}=1\}}{m}$$
and $$p(x_j\|y=0) = \frac{\sum_{i=1}^m1\{x_j^{(i)}=1,y^{(i)}=0\}}{\sum_{i=1}^m1\{y^{(i)}=0\}}$$

**Loss Function: Maximum Likelihood Estimation**

**Laplace Smoothing**: to avoid the problem of zero probability, we can add a small number $\epsilon$ to the numerator and denominator.

**Cases**: text classification, spam detection, etc.
**Advantages**: computation is efficient, and quick to implement.
**disadvantages**: separate features, so it can't learn the interactions between features. word embeddings (from neural network) can solve this problem.

## 7. Support Vector Machine
for *non-linearly* separable data, we can use SVM to solve this problem. SVM is a discriminative learning algorithm. it is a binary classifier.

recommend reading: [Zhihu](https://zhuanlan.zhihu.com/p/77750026)

### 7.1 Margin
**Description**: the margin is the distance between the decision boundary and the closest training example. the goal of SVM is to maximize the margin.

![picture](imgs/7-%20svm1.png)

**Model: SVM**
hyperplane: $w^Tx+b=0$
decision function: $$h_\theta(x) = g(w^Tx+b)$$
$$g(z) = \begin{cases}1 & w^Tx+b\geq0\\-1 & w^Tx+b<0\end{cases}$$

functional margin: the distance between the decision boundary and the training example.
$$\hat{\gamma}^{(i)} = y^{(i)}(w^Tx^{(i)}+b)$$
geometric margin: the distance between the decision boundary and the closest training example.
$$\gamma^{(i)} = y^{(i)}(\frac{w}{||w||}^Tx^{(i)}+\frac{b}{||w||})$$
if $||w||=1$, the functional margin equals the geometric margin.


**Loss Function: Maximum Geometry Margin**
$$\max_{\gamma,w,b}\gamma= \max_{\gamma,w,b}\frac{\hat{\gamma}}{||w||}\sim \min_{\gamma,w,b}\frac{1}{2}||w||^2$$
$$s.t. y^{(i)}(w^Tx^{(i)}+b)-1\geq0, i=1,...,m$$
$$||w||=1$$
**Optimization: Convex Quadratic Programming**

The decision boundary is related to the **support vectors**. the support vectors are the training examples that are closest to the decision boundary. the decision boundary is the hyperplane that is equidistant from the support vectors.

**Soft margin**: the data is not linearly separable, so we can use soft margin to solve this problem. we can introduce the slack variable $\xi$ to the functional margin.

### 7.2 Lagrange Duality
**Description**: Lagrange Duality is a method to solve the optimization problem with constraints. it is a general method to solve the optimization problem.

Consider the following optimization problem:
$$\min_{w}f(w)$$
$$s.t. h_i(w)=0, i=1,...,l$$
$$g_j(w)\leq0, j=1,...,m$$
where $f:\mathbb{R}^n\rightarrow\mathbb{R}$, $h_i:\mathbb{R}^n\rightarrow\mathbb{R}$, $g_j:\mathbb{R}^n\rightarrow\mathbb{R}$.

In this method, we define the generalized Lagrangian function:
$$L(w,\alpha,\beta) = f(w) + \sum_{i=1}^l\alpha_ih_i(w) + \sum_{j=1}^m\beta_jg_j(w)$$
where $\alpha_i$ and $\beta_j$ are Lagrange multipliers.

The dual function is:
$$\theta(\alpha,\beta) = \min_{w}L(w,\alpha,\beta)$$
$$s.t. \alpha_i\geq0, i=1,...,l$$

The dual problem is:
$$\max_{\alpha,\beta}\theta(\alpha,\beta)$$


### 7.3 KKT Conditions
**Description**: KKT conditions are the necessary conditions for the optimization problem. if the optimization problem is convex, the KKT conditions are also the sufficient conditions.

**KKT Conditions**
$$\frac{\partial L}{\partial w_i} = 0, i=1,...,n$$
$$\frac{\partial L}{\partial \alpha_i} = 0, i=1,...,l$$
$$\frac{\partial L}{\partial \beta_i} = 0, i=1,...,m$$
$$h_i(w)=0, i=1,...,l$$
$$g_j(w)\leq0, j=1,...,m$$
$$\beta_j\geq0, j=1,...,m$$
$$\beta_jg_j(w)=0, j=1,...,m$$

### 7.4 SVM Optimization steps
$$\min_{w,b}\frac{1}{2}||w||^2$$
$$s.t. y^{(i)}(w^Tx^{(i)}+b)-1\geq0, i=1,...,m$$
**Step 1**: we can use the Lagrange Duality to solve the optimization problem of SVM.
$$ \min_{\omega,b}\max_\lambda L(w,b,\lambda) = \min_{\omega,b}\max_\lambda\frac{1}{2}||w||^2+\sum_{i=1}^m\lambda_i(1-y^{(i)}(w^Tx^{(i)}+b))$$
$$s.t. \lambda_i\geq0, i=1,...,m$$

**Step 2**: according to the KKT conditions and hard dual, we can get the following optimization problem:
$$\min_{\omega,b} L(w,b,\lambda) ={\lambda}\sum_{i=1}^m\lambda_i-\frac{1}{2}\sum_{i=1}^m\sum_{j=1}^m\lambda_i\lambda_jy^{(i)}y^{(j)}x^{(i)T}x^{(j)}$$

**Step 3**: 
$$\max \sum_{i=1}^m\lambda_i-\frac{1}{2}\sum_{i=1}^m\sum_{j=1}^m\lambda_i\lambda_jy^{(i)}y^{(j)}x^{(i)T}x^{(j)}$$
$$s.t. \sum_{i=1}^m\lambda_iy^{(i)}=0$$
We can use the SMO algorithm to solve the optimization problem of SVM.
**SMO**(Squential Minimal Optimization) alogirithm with its core idea of selecting two variables to optimize at each step, and then fixing the other variables, and then optimizing the two variables, and then fixing the other variables, and so on, until the convergence of the objective function.
The primal constraint is: $\sum_{i=1}^m\lambda_iy^{(i)}=0$.
The new constraint becomes: $\lambda_iy^{(i)}+\lambda_jy^{(j)}=c$. and iterating until the convergence of the objective function.

**Step 4**: after getting the optimal $\lambda$, we can get the optimal $w$ and $b$.
$$w = \sum_{i=1}^m\lambda_iy^{(i)}x^{(i)}$$
$$b = y^{(j)}-\sum_{i=1}^m\lambda_iy^{(i)}x^{(i)T}x^{(j)}$$
the decision boundary is: $w^Tx+b=0$.
the decision function is: $f(x)=sign(w^Tx+b)$.
### 7.5 Kernel Methods
for **non-linearly** separable data, we can use kernel methods to solve this problem. kernel methods are a class of algorithms for pattern analysis. kernel methods are widely used in SVM.

**Kernel Function**
**Description**: kernel function is a function that computes the inner product of two vectors in a higher dimensional space. 
$$K(x,z) = \phi(x)^T\phi(z)$$
where $\phi$ is a feature mapping function that maps the input space to a higher dimensional space.

**Kernel Trick**: The feature mapping function $\phi$ is often difficult to compute because it will map the feature space to a higher dimensional space even infinite dimensional space. The kernel trick is to find a kernel function that can directly compute the inner product of two vectors in a higher dimensional space without explicitly computing the feature mapping function $\phi$.

**Kernel Matrix**: the kernel matrix is a matrix of the inner products of all pairs of a dataset.

commonly used kernel functions:
1. linear kernel: $K(x,z)=x^Tz$
2. polynomial kernel: $K(x,z)=(x^Tz+1)^d$
3. Gaussian kernel: $K(x,z)=exp(-\frac{||x-z||^2}{2\sigma^2})$
4. Laplace kernel: $K(x,z)=exp(-\frac{||x-z||}{\sigma})$

The **advantages** and **disadvantages** of SVM:
Advantages:
1. SVM is suitable for small sample size and high-dimensional data.
2. SVM is suitable for solving non-linear classification problems and regression problems by using kernel methods.
3. The decision function of SVM is only related to a few support vectors, so it is not sensitive to the change of the training set which avoids the problem of dimension disaster.
4. SVM can solve the problem of overfitting by introducing the penalty factor $C$.

Disadvantages:
1. SVM is not suitable for large sample size and high-dimensional data.
2. The training time of SVM is relatively long.
3. The selection of kernel function is not easy and the the space complexity is high. 

## 8. Decision Tree
