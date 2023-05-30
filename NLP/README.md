# Natural Language Processing
before beginning NLP course, we should study the machine learning.

- [Natural Language Processing](#natural-language-processing)
  - [1. Word Vectors](#1-word-vectors)
    - [1.1. Distributional semantics](#11-distributional-semantics)
    - [1.2. Word2Vec](#12-word2vec)
    - [1.3. Skip-gram model](#13-skip-gram-model)
    - [1.4. CBOW model (Continuous Bag-of-Words)](#14-cbow-model-continuous-bag-of-words)
    - [1.5. Negative sampling](#15-negative-sampling)
    - [1.6. Hierarchical softmax](#16-hierarchical-softmax)
    - [1.7. The skip-gram model with nagative sampling](#17-the-skip-gram-model-with-nagative-sampling)
  - [2. GloVe (Global Vectors for Word Representation)](#2-glove-global-vectors-for-word-representation)
    - [2.1. Co-occurrence matrix](#21-co-occurrence-matrix)
    - [2.2. GloVe model](#22-glove-model)
    - [2.3. Evaluate word vectors](#23-evaluate-word-vectors)
    - [2.4. Word Sense](#24-word-sense)

## 1. Word Vectors
| Title | Author | Conference | Date | Model |
| :---: | :---: | :---: | :---: | :---: |
| [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf) | Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean | ICLR |2013 | Word2Vec | 
| [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/pdf/1310.4546.pdf) | Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean | NIPS |2013 | Word2Vec | 
| [word2vec parameter learning explained](https://arxiv.org/pdf/1411.2738.pdf) | Xin Rong | arXiv |2014 | Word2Vec |

materiers:
- [Word2Vec Tutorial - The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
- [Zhihu](https://zhuanlan.zhihu.com/p/53425736)

How do we represent the meaning of a word?
Previously commonest NLP solution is using a thesaurus containing lists of **synonym** sets and **hypernyms** (is-a relationships) called **WordNet**. But it is not enough to represent the meaning of a word.
```python
from nltk.corpus import wordnet as wn
poses = {'n': 'noun', 'v': 'verb', 's': 'adj (s)', 'a': 'adj', 'r': 'adv'}
for synset in wn.synsets("good"):
    print("{}: {}".format(poses[synset.pos()],
                          ", ".join([l.name() for l in synset.lemmas()])))
```
The output is:
```
noun: good
noun: good, goodness
noun: good, goodness
noun: commodity, trade_good, good
adj (s): full, good
adj (s): good
adj (s): estimable, good, honorable, respectable
adj (s): beneficial, good
adj (s): good
adj (s): good, just, upright
...
adverb: well, good
adverb: thoroughly, soundly, good
```
```python
from nltk.corpus import wordnet as wn
panda = wn.synset('panda.n.01')
hyper = lambda s: s.hypernyms()
list(panda.closure(hyper))
```
The output is:
```
[Synset('procyonid.n.01'),
 Synset('carnivore.n.01'),
 Synset('placental.n.01'),
 Synset('mammal.n.01'),
 Synset('vertebrate.n.01'),
 Synset('chordate.n.01'),
 Synset('animal.n.01'),
 Synset('organism.n.01'),
 Synset('living_thing.n.01'),
 Synset('whole.n.02'),
 Synset('object.n.01'),
 Synset('physical_entity.n.01'),
 Synset('entity.n.01')]
```

Problems with resources like WordNet:
- missing nuance
- missing new meanings(impossiable to up-to-date)
- requires human labor to create and adapt

In traditional NLP, we regard words as discrete symbols, such symbols for words are represented by **one-hot** vectors:
- each word in the vocabulary is represented by a vector with 1.0 in the position corresponding to the word's index in the vocabulary and 0s in all other positions.
- the number of dimensions of these vectors is the size of the vocabulary.
for example, $v_{cat} = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]$ and $v_{dog} = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]$.

The problems of one-hot vectors are:
- they are high-dimensional
- they are sparse (most of the elements are zero)
- The two vectors are orthogonal
- There is no natural notion of similarity for one-hot vectors.

**Learn to encode similarity in the vectors themselves**

### 1.1. Distributional semantics
**Distributional semantics:** A word's meaning is given by the words that frequently appear close-by. It is one of the most successful ideas of modern statistical NLP.
When a word *w* appears in a text, its context is the set of words that appear nearby(within a fixed-size window).
We use the many contexts of *w* to build up a representation of *w*.
The **context words** will represent a *word*
```python
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
text = "Hello, this is a test sentence."
tokens = word_tokenize(text.lower())
print(tokens)
```
The output is:
```
['hello', ',', 'this', 'is', 'a', 'test', 'sentence', '.']
```

### 1.2. Word2Vec
We will build a dense vector for each word, chosen so that it is similar to vectors of words that appear in similar contexts, measuring similarity as the vector **dot product**.
**Word vectors** are also called **word embeddings** or **neural word representations**.
for example:
$$\text{banking}= \begin{bmatrix} 0.286 & 0.792 & -0.177 & \cdots & -0.107 & 0.731 & 0.197 & \cdots & -0.532 & 0.419 \end{bmatrix}$$

**Word2Vec** is a framework for learning word vectors.
Idea: 
- We have a large corpus of text
- Every word in a fixed vocabulary is represented by a vector
- Go through each position t in the text, which has a center word *c* and context ("outside") words *o*
- Use the **similarity of the word vectors** for *c* and *o* to **calculate the probability** of *o* given *c* (or vice versa)
- **Keep adjusting the word vectors** to maximize this probability

![Word2Vec](./imgs/w2v.png)

A cool example: $\text{man}-\text{woman}= \text{king}-\text{queen}$

**It's important that the input vector and output vector are just different representations of the same word. Because we predict the word in terms of context words.**

![Word2Vec](./imgs/w2v4.png)
### 1.3. Skip-gram model
For each position $t = 1, \cdots, T$, predict context words within a window of fixed size $m$ given center word $w_t$. The likelihood is given by:
$$L(\theta) = \prod_{t=1}^T \prod_{-m \leq j \leq m, j \neq 0} P(w_{t+j} | w_t; \theta)$$
where $\theta$ is the parameter of the model, which are the word vectors for all words in the vocabulary and the neural network weights, the $m$ is the size of the context window.
The objective function is the average log probability:
$$J(\theta) = -\frac{1}{T} \log L(\theta) = -\frac{1}{T} \sum_{t=1}^T \sum_{-m \leq j \leq m, j \neq 0} \log P(w_{t+j} | w_t; \theta)$$
The conditional probability is defined as:
$$P(o | c) = \frac{\exp(\text{score}(o, c))}{\sum_{w \in V} \exp(\text{score}(w, c))}$$
where $V$ is the vocabulary and the score function is defined as:
$$\text{score}(o, c) = u_o^T v_c$$
where $u_o$ is the "outside" vector for $o$ and $v_c$ is the "center" vector for $c$.
![Word2Vec](./imgs/w2v2.png)
To train the model: optimize value of parameters to minimize loss by SGD.
$$\theta^* = \arg \min_\theta J(\theta)$$
where $\theta$ is the parameter of the model, which are the word vectors for all words in the vocabulary and the neural network weights.
![Word2Vec](./imgs/w2v3.png)
$v$ represents the center word vector, $u$ represents the outside word vector.

### 1.4. CBOW model (Continuous Bag-of-Words)
The CBOW model is very similar to the skip-gram model. The only difference is that CBOW predicts the current word based on the context, whereas the skip-gram predicts surrounding words given the current word.

The model makes the same predictions at each position. We want a model that gives a reasonably high probability estimate to all words that occur in the context.
### 1.5. Negative sampling
The problem with the softmax function is that it is **computationally expensive** to calculate the denominator of the softmax function, which involves summing over all words in the vocabulary.

The idea of negative sampling is to only update the parameters for the target word and a small number of other "negative" words, rather than updating all the parameters for each training sample.

Train binary logistic regressions to differentiate a true pair of $(c, o)$ from a noise pair of $(c, o')$.

### 1.6. Hierarchical softmax
Hierarchical softmax is an alternative to the softmax function that is more efficient to compute. It is based on the idea of a binary tree, where each leaf node represents a word in the vocabulary.

### 1.7. The skip-gram model with nagative sampling
The objective function is:
$$J(\theta) = -\frac{1}{T} \sum_{t=1}^T \left[ \log \sigma(u_{w_t}^T v_{w_t}) + \sum_{i=1}^k \mathbb{E}_{w_i \sim P_n(w)} \log \sigma(-u_{w_i}^T v_{w_t}) \right]$$
where $k$ is the number of negative samples, $P_n(w)$ is the noise distribution, which is usually chosen to be the unigram distribution raised to the 3/4 power.

## 2. GloVe (Global Vectors for Word Representation)
| Title | Author | Conference | Date | Model |
| :---: | :---: | :---: | :---: | :---: |
|[GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) | Jeffrey Pennington, Richard Socher, Christopher D. Manning | EMNLP | 2014 | GloVe |
| [Improving Distributional Similarity with Lessons Learned from Word Embeddings](https://www.aclweb.org/anthology/Q15-1016.pdf) | Omer Levy, Yoav Goldberg, Ido Dagan | TACL | 2015 | GloVe |
| [Evaluation methods for unsupervised word embeddings](https://www.aclweb.org/anthology/D15-1036.pdf) | Tobias Schnabel, Igor Labutov, David Mimno, Thorsten Joachims | EMNLP | 2015 | GloVe |
| [A Latent Variable Model Approach to PMI-based Word Embeddings](https://www.aclweb.org/anthology/P15-1078.pdf) | Omer Levy, Yoav Goldberg | TACL | 2015 | GloVe |
| [Linear Algebraic Structure of Word Senses, with Applications to Polysemy](https://www.aclweb.org/anthology/P15-2071.pdf) | Omer Levy, Yoav Goldberg, Ido Dagan | TACL | 2015 | GloVe |
| [On the Dimensionality of Word Embedding](https://www.aclweb.org/anthology/P15-2078.pdf) | Omer Levy, Yoav Goldberg | TACL | 2015 | GloVe |
### 2.1. Co-occurrence matrix
The co-occurrence matrix $X$ is a $|V| \times |V|$ matrix, where $V$ is the vocabulary. The $(i, j)$-th element of $X$ is the number of times word $i$ occurs in the context of word $j$.
We just accumulate all the statistics of what appear near each other.

**Co-occurrence vectors** are the rows of the co-occurrence matrix. They are the vectors that represent each word. It has some disadvantages:
- The matrix is very large and requires a lot of storage.
- The matrix is very sparse and is less robust.

What we need is **Low-dimensional dense vectors**.
idea: store most of the important information in a fixed, small number of dimensions. We need to reduce the dimensionality.

**SVD(singular value decomposition)** is a good way to reduce the dimensionality of the co-occurrence matrix. The co-occurrence matrix $X$ can be decomposed into three matrices $U$, $\Sigma$, $V^T$:
$$X = U \Sigma V^T$$
where $U$ is a $|V| \times k$ matrix, $\Sigma$ is a $k \times k$ matrix, $V^T$ is a $k \times |V|$ matrix, $k$ is the number of dimensions we want to reduce to.

Running an SVD on raw counts doesn't work well. We need to do some preprocessing:
- Remove very frequent words (the, of, and, etc.)
- Scale the remaining counts using PPMI (positive pointwise mutual information)
We also can use Pearson correlations instead of counts. then set the negative values to zero.

**Count based** vs. **direct prediction**:
- Count based: LSA, HAL, Hellinger PCA. Fast
  - advantages: fast, efficient use of statistics.
  - disadvantages: primarily capture word similarity, disproportionate importance given to large counts.
- Direct prediction: word2vec, GloVe. NNLM, RNN
  - advantages: generate improved performance on other tasks, can capture complex patterns beyond word similarity.
  - disadvantages: scale with corpus size, less efficient use of statistics.

### 2.2. GloVe model
The idea of GloVe is to learn word vectors such that their dot product equals the logarithm of the words' probability of co-occurrence.
$$w_i^T w_j = \log X_{ij}$$
where $w_i$ is the vector for word $i$ and $X_{ij}$ is the number of times word $j$ occurs in the context of word $i$.
with vector differences:
$$(w_i - w_j)^T w_k = \log \frac{X_{ik}}{X_{jk}}$$
The objective function is:
$$J = \sum_{i, j=1}^{|V|} f(X_{ij}) (w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$$
where $f$ is a weighting function, $b_i$ and $\tilde{b}_j$ are bias terms.
The advantage of GloVe is that it is scalable and can learn from large corpora.

![GloVe](./imgs/glove1.png)

### 2.3. Evaluate word vectors
general evaluation in NLP: intrinsic and extrinsic.
- Intrinsic evaluation: evaluate the word vectors on a specific task.
  - fast to compute, but not clear if really helpful unless correlation to real task is established.

Word vector analogies:
- $a$ is to $b$ as $c$ is to $d$.

$$\arg \max_{d \in V} \cos(d, b - a + c)$$
$$ d= \arg \max_{x \in V} \frac{x^T (b - a + c)}{ ||b - a + c||}$$

- Extrinsic evaluation: evaluate the word vectors on a downstream task.
  - take a long time to compute accuracy, and unclear if the subsystem with another improves accuracy.

**named entity recognition:** identifying references to a person, organization or location: Chris Manning lives in Palo Alto.

### 2.4. Word Sense
Most words have lots of meanings. We need to disambiguate the meaning of a word in a particular context. Especially common words.
What we want to do is to choose the correct meaning of a word in a particular context.

One idea is that clusting word windows around words, retrain with each word assigned to multiple different clusters $\text{bank}_1$, $\text{bank}_2$, $\text{bank}_3$, $\text{bank}_4$, etc.

**Linear algebraic structure of word senses:**
Different senses of a word reside in a linear superposition (weighted sum) in standard word embeddings like word2vec and GloVe.
$$\text{bank} = \alpha_1\text{bank}_1 + \alpha_2\text{bank}_2 + \alpha_3 \text{bank}_3 + \alpha_4\text{bank}_4$$

Because of ideas from sparse coding you can actually separate out the senses.
![word sense](./imgs/glove2.png)
