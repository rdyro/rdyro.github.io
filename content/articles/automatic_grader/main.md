# Introduction

The purpose of this article is to present our approach to the problem of
automatically grading short answers by students given a handful of examples of
graded answers for a particular question. The additional particular requirements
for our problem setting are
- the model must be very computationally efficient
    - the number of supported students can be high and the revenue per student very low
    - the model should be able to perform live regrades in a matter of
    milliseconds for each student
- the model must be resistant to adversarial attacks
    - inputting gibberish or a completely irrelevant answer should never result
    in a grade other than 0
- the model must be able to support questions from various of science and
liberal arts fields and in multiple languages
    - it should work for any subject taught in school
    - it should be usable by at least the major languages
We propose to solve the problem by using existing multi-lingual language
embedding models, which convert questions and answers into a fixed-length
real-numbered context vector, to which we apply online regression methods.
In order to make the model additionally resistant to adversarial inputs, we
build and additional *relevancy* model whose purpose is to determine if the
answer is relevant to the question (even if it is incomplete or wrong, does it
at least pertain to the subject).

In this work we propose to use an existing pre-trained, transformer-based,
multilingual model to generate embedding vectors -- real-valued, fixed-length
vectors -- representing every answer to train 
1. a grade classification model outputting the probability of a grade for each
numerical grade (from 0 to N)
2. a relevancy classification model outputting probability that the answer is
relevant (on topic, even if wrong or incomplete) or irrelevant (if in a wrong
language or completely inapplicable)

----

# Short Overview of Practical NLP

The field of natural language processing (NLP) has a long history and includes
many completely different approaches from explicit text structure discovery to
deep learning methods, most recently (around 2017) involving the transformer
model (TODO add link).

We will focus on the recent deep-learning approaches and will use a
transformer-based embedding model. In this paradigm, the text to be embedded
(converted to a real-valued vector) needs to (1) first pass through a tokenizer,
getting converted to a list indices representing words or word fragments in a
set of words & word fragments known to the model, (2) then the token index list
is passed once forward through the transformer-based model and averaged into a
fixed length real-valued vector.

Thus, we have two components, (1) a tokenizer which converts text to integers
indices and (2) a model which converts integers into a *hopefully meaningful*
real-valued vector corresponding to the vector. Tokenizers and models are
usually bundled together since (i) a model trained on one tokenizer will only
work well with a tokenizer that is the same or differs by only a very small
number of words and (ii) that tokenizer was shown to successfully result in a
good model -- if the model is a pre-trained or verified to work well.

### The Tokenizer

The purpose of a tokenizer is to convert a text into a set of numbers that a
language model can ingest. A tokenizer typically has a fixed dictionary where
each word (or a word fragment) corresponds to a unique integer index. The
tokenizer then converts each word in the text into the corresponding integer
index. To avoid an exploding size of the vocabulary, tokenizers sometimes split
longer words into smaller segments that are shared between words. These short
words and common fragments are collectively known as *tokens*. For English, a
text of length N will usually be converted into 1.33 N tokens.

Apart from words, the vocabulary of a tokenizer can also contain special tokens like end of text indicators, end of sentence indicators, unknown word indicators, etc. An unknown word is either split into existing tokens (if it can be decomposed as such) or replaced with the *unkown word* token. 

A tokenizer converts words into numbers, but sometimes splits longer (or
unknown) words into shorter segments to both avoid an exploding size of the
vocabulary and to allow the model to generalize to new words.

A trained language model will typically have a specific tokenizer associated
with it as changing which integer corresponds to which token will result in
model receiving unrelated inputs.

<p align="center">
<img src="../../../assets/images/tokenization_illustration.svg" style="width:50%;max-width:500px" />
<br>
<p align="left">
Fig: A tokenizer converts words into numbers, but sometimes splits longer (or unknown) words into shorter segments to both avoid an exploding size of the vocabulary and to allow the model to generalize to new words.
</p>
</p>

### The Embedding Model

An embedding model is simply a language model that converts the tokenized text
into a fixed length real-valued vector for further downstream tasks. Many
language applications start with a pre-trained embedding model and append
another, task-specific model to work using the embeddings produced by the
embedding model. Typically, embedding models are trained in such a way that
their embedding output produces good results when used with a variety of
downstream models. Some applications fine-tune the embedding model, not only
train the downstream model to achieve better results, but that sometimes risks
reducing the generalization properties of the embedding model (since the
embedding model is often trained on a very diverse dataset for the express
purpose of achieving good generalization properties).

<p align="center">
<img src="../../../assets/images/downstream_language_model.svg" style="width:80%;max-width:800px" />
<br>
<p align="left">
Fig: An pre-trained embedding model followed by a downstream model is a common
approach to solving natural language processing tasks.
</p>
</p>

Modern embedding model often use the transformer architecture. Because most
transformers are designed to work on real-valued vector representation of words,
the integer output of the tokenizer is often first passed through an *embedding
layer* which simply projects an integer index into a random (but potentially
learnable) real-valued vector.

Training and inference is often more efficiently done in batches, but most
transformer architecture require that all inputs in a batch have the same
length. Most tokenizers are designed to pad all, but the shortest input in a
batch with special pad tokens which are supposed to not affect that text
fragment embedding value. A text fragment with and without pad tokens should
produce the same embedding value (which is mostly the case, except when the
embedding model using very low precision arithmetic, like `int8`, as some errors
tend to accumulate). Additionally, most embedding models are designed with a max
input text length and the associate tokenizer will often truncated longer texts
by simply cutting off the end of the text after the maximum numbered token.

The embedding model is simply a fixed-length map from a variable length text to a fixed-length real-valued vector. For transformer models, the embedding is often simply a summation of the final embedding of each token in the text.

$$ f_\text{embedding}: \text{"brown dog..."} \rightarrow \mathbb{R}^{512} $$

where 768 is an example number dependent on the model architecture.

----

# The Problem: Automatically Grading 100s of Assignments

The problem described in this article is to develop a system that can
automatically grade short answer problems -- problems for which the student is
required to write an answer in the form of a short text. 

The system is designed to generalize from a small handful of teacher-graded
answers (circa 20-30) to automatically grade remaining future answers, as well
as grading student answers in real-time (below 1 second).

----

# Estimating Relevance

Because the model is available to the students in real-time, we want to avoid
the possibility that the student will insert irrelevant text into the answer box
until the answer gets a high grade. Some examples could include: 
- an empty answer
- a copy of the question
- an expertly written answer from the same domain, but not relevant to the question

For that reason we attempt to estimate relevance as the probability that an
answer is relevant to the question. Importantly:
- an incorrect answer on topic is still *relevant*
- an incomplete answer is still *relevant*

## The Relevance Model

The relevance model is a simple downstream binary classifier model built on top
of the embedding model. The model is trained to predict whether an answer is
relevant to the question or not. The architecture is a simple multi-layered
perceptron (MLP) trained using gradient descent with validation set
hyperparameter tuning (batch size, learning rate, dropout, optimizer) using
[ray.tune](https://www.ray.io/ray-tune).

The dataset is a combination of several instruction datasets (e.g.,
[RedPajama](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-Instruct))
(popularized by large language model research). Negative example are produced by randomly assigning questions (instructions) to responses.

----

# Using a Pre-trained Model

We use the multilingual BERT model for its excellent ratio of generalization
capability and computational efficiency. The model weights (expressed in
`float32` arithmetic) are only around 600 MB. TODO expand

----

# Meta-learning the Model

TODO

----

# Adapting the Relevance

TODO

----

# Alternative Approaches

One obvious alternative approach is using a large language model (LLM),
specifically the instruction interface (a chat mode, where the model responds
to instruction) to grade answers using graded examples. The benefit of this
approach is that it is much more flexible: grading can be based on example
grades or a grading rubric directly, but the downside is the high computational
resources required to run the model. 

The computational cost in the large language models scales with the context size
linearly or super-linearly. For our problem setting, where we wish to provide
the model with example graded answers, including each additional example answer
increases the computational cost and places a soft limit on the maximum number
of examples considered. In contrast, our approach uses a much smaller embedding
model, where the embedding cost is strictly linearly dependent on the number of
embedded answers and both the online adaptation and classification costs are
virtually negligible in comparison to the embedding cost.

## Computational Cost Comparison

TODO

----

# Data Generation

Data for training both the relevance model and the grading model can be obtained
from real responses, since labeling is relatively inexpensive:
- grades are assigned to questions in the existing, manual grading scheme
- relevance scores are binary and they are often easy to assign

However, training both models could benefit from large amounts of synthetic
data. Recent advanced in large language models yield instruction-agents, models
which can generate natural sounding text in a variety of language. 

We can easily use a model like this (in this case ChatGPT-3.5 from OpenAI TODO
add link) to generate examples of relevant and irrelevant synthetic responses for
training. Consider the example

Instruction:
> I want you to generate examples of relevant and irrelevant answers to the
> question: How many cases are there for finding roots of a quadratic equation? I
> want you to generate 50 relevant and 50 irrelevant answers. The relevant answers
> should include both correct, incorrect, and also incomplete answers. The
> irrelevant answers should look like they are designed to fool and automatic
> grading system for that question. The irrelevant answers should be completely
> off-topic and some of them should be nonsensical and/or in a different language.
> Make both categories of answers include long and short answers.

> ChatGPT
> Relevant Answers:
>
> There are generally three cases for finding roots of a quadratic equation.
>
> Two cases are considered when finding roots of a quadratic equation.
>
> The number of cases for finding roots of a quadratic equation depends on the discriminant.
>
> The cases for finding roots of a quadratic equation can vary depending on the coefficients.
>
> One case involves finding two distinct real roots for a quadratic equation.
>
> Incomplete answer: The cases for finding roots of a quadratic equation are determined by the...
>
> Incorrect answer: Quadratic equations always have four cases for finding roots.
>
> There can be two imaginary roots when solving a quadratic equation.
>
> An incomplete answer: The cases for finding roots of a quadratic equation depend on the number...
>
> One case involves finding one real root for a quadratic equation.
>
> Long answer: The number of cases for finding roots of a quadratic equation can be determined by analyzing the discriminant, which is the expression b^2 - 4ac. If the discriminant is greater than zero, there are two distinct real roots. If the discriminant is equal to zero, there is one real root, which is repeated. And if the discriminant is less than zero, there are two complex roots, which are conjugate pairs of the form (a + bi) and (a - bi), where a and b are real numbers and i is the imaginary unit.
>
> Incomplete answer: Quadratic equations have multiple cases for finding roots, such as...
>
> Incorrect answer: The number of cases for finding roots of a quadratic equation is always two.
>
> There are multiple cases for finding roots of a quadratic equation, including complex solutions.
>
> Long answer: When solving a quadratic equation, there are three possible cases: having two distinct real roots, having one real root (with multiplicity), or having two complex roots.
>
> One case involves finding no real roots for a quadratic equation.
>
> Incorrect answer: There is only one case for finding roots of a quadratic equation.
>
> The number of cases for finding roots of a quadratic equation depends on the coefficients and the degree of the equation.
>
> Incomplete answer: The cases for finding roots of a quadratic equation can be classified into...
>
> One case involves finding two equal real roots for a quadratic equation.
>
> Long answer: The number of cases for finding roots of a quadratic equation can vary from zero to two, depending on the values of the coefficients a, b, and c in the equation.
>
> Incorrect answer: Quadratic equations have different cases for finding roots based on their leading coefficient.
>
> There are cases where a quadratic equation may have infinite solutions.
>
> Incomplete answer: The cases for finding roots of a quadratic equation can be determined by analyzing the quadratic formula.
>
> Two cases are considered when finding roots of a quadratic equation: real and complex.
>
> Long answer: The number of cases for finding roots of a quadratic equation is determined by the nature of the discriminant. If the discriminant is positive, there are two distinct real roots. If it is zero, there is one real root. And if it is negative, there are two complex roots.
>
> There are various scenarios for finding roots of a quadratic equation, such as...
>
> Incomplete answer: The number of cases for finding roots of a quadratic equation depends on the values of the coefficients and the discriminant, which is calculated using the formula b^2 - 4ac.
>
> Incorrect answer: Quadratic equations can have an infinite number of cases for finding roots.
>
> The number of cases for finding roots of a quadratic equation can be determined by examining the sign of the discriminant.
>
> Long answer: The cases for finding roots of a quadratic equation can be categorized as follows: two real and distinct roots when the discriminant is positive, one real repeated root when the discriminant is zero, and two complex conjugate roots when the discriminant is negative.
>
> Incorrect answer: The number of cases for finding roots of a quadratic equation is determined by the order of the equation.
>
> There are specific criteria to determine the cases for finding roots of a quadratic equation.
>
> Incomplete answer: The cases for finding roots of a quadratic equation depend on various factors, including...
>
> The number of cases for finding roots of a quadratic equation can be identified by analyzing the coefficients and the discriminant.
>
> Long answer: Depending on the values of the coefficients a, b, and c in a quadratic equation, there can be three cases: two distinct real roots, one repeated real root, or two complex roots.
>
> Incorrect answer: Quadratic equations have a fixed number of cases for finding roots, regardless of their coefficients.
>
> The cases for finding roots of a quadratic equation can be determined by examining the leading coefficient.
>
> Incomplete answer: The number of cases for finding roots of a quadratic equation is determined by...
>
> One case involves finding two imaginary roots for a quadratic equation.
>
> Long answer: The number of cases for finding roots of a quadratic equation depends on the value of the discriminant. If the discriminant is positive, there are two distinct real roots. If it is zero, there is one real root. And if it is negative, there are two complex roots.
>
> Incorrect answer: Quadratic equations can have a variable number of cases for finding roots, depending on the degree of the equation.
>
> The cases for finding roots of a quadratic equation can be classified based on the values of the coefficients.
>
> Incomplete answer: The number of cases for finding roots of a quadratic equation can be determined by examining the coefficients and the discriminant, which is the part of the quadratic formula under the square root symbol.
>
> There are specific patterns to identify the cases for finding roots of a quadratic equation.
>
> Long answer: Depending on the coefficients and the value of the discriminant, a quadratic equation can have either two distinct real roots, one repeated real root, or two complex roots.
>
> Incorrect answer: The number of cases for finding roots of a quadratic equation is always the same, regardless of the equation's form.
>
> There are multiple possibilities for the cases when finding roots of a quadratic equation.
>
> Incomplete answer: The cases for finding roots of a quadratic equation can be determined by analyzing the equation's factors.
>
> The number of cases for finding roots of a quadratic equation can be determined by the nature of the discriminant.
>
> Irrelevant Answers:
>
> The quadratic equation is a type of dance move that originated in the 1920s.
>
> ¡Hola! No sé qué es una ecuación cuadrática.
>
> The number of cases for finding roots of a quadratic equation is 42, the meaning of life, the universe, and everything.
>
> Quadratic equations are like a box of chocolates; you never know how many cases you'll get.
>
> Quadratic equations have zero cases for finding roots because they are too shy to show themselves.
>
> Quadratic equations have exactly five cases for finding roots, just like the fingers on your hand.
>
> The number of cases for finding roots of a quadratic equation depends on the price of tea in China.
>
> Quadratic equations have a secret code, and only those with the right decoder ring can determine the number of cases for finding roots.
>
> Sorry, I can't answer that. I'm busy counting sheep.
>
> Quadratic equations are aliens from another dimension, and their roots cannot be found in our universe.
>
> The cases for finding roots of a quadratic equation are locked away in the lost city of Atlantis.
>
> Quadratic equations have a unique case for finding roots that involves time travel.
>
> The number of cases for finding roots of a quadratic equation depends on the alignment of the stars.
>
> Quadratic equations have a secret society that decides the number of cases for finding roots.
>
> The cases for finding roots of a quadratic equation are written in a language only mathematicians understand.
>
> The number of cases for finding roots of a quadratic equation is classified information.
>
> Quadratic equations have an unlimited number of cases for finding roots because they are magical.
>
> The cases for finding roots of a quadratic equation are stored on a floppy disk.
>
> Quadratic equations have a secret handshake that reveals the number of cases for finding roots.
>
> The number of cases for finding roots of a quadratic equation depends on the color of your socks.
>
> Quadratic equations have a hidden treasure map that leads to the number of cases for finding roots.
>
> The cases for finding roots of a quadratic equation are like a crossword puzzle waiting to be solved.
>
> Quadratic equations have a special case for finding roots that involves talking to plants.
>
> The number of cases for finding roots of a quadratic equation depends on the length of your hair.
>
> Quadratic equations have a secret language that only cats can understand.
>
> The cases for finding roots of a quadratic equation are stored in a parallel universe.
>
> Quadratic equations have a secret dance routine that reveals the number of cases for finding roots.
>
> The number of cases for finding roots of a quadratic equation depends on the phase of the moon.
>
> Quadratic equations have a secret recipe that determines the number of cases for finding roots.
>
> The cases for finding roots of a quadratic equation are hidden in a riddle that only the Sphinx can solve.
>
> Quadratic equations have a secret door that opens to reveal the number of cases for finding roots.
>
> The number of cases for finding roots of a quadratic equation depends on the position of the planets.
>
> Quadratic equations have a secret codebook that reveals the number of cases for finding roots.
>
> The cases for finding roots of a quadratic equation are written on the wings of butterflies.
>
> Quadratic equations have a secret handshake that only mathematicians know.
>
> The number of cases for finding roots of a quadratic equation depends on the color of your eyes.
>
> Quadratic equations have a secret language that only dolphins can understand.
>
> The cases for finding roots of a quadratic equation are hidden in a hieroglyphic puzzle.
>
> Quadratic equations have a secret map that leads to the number of cases for finding roots.
>
> The number of cases for finding roots of a quadratic equation depends on the alignment of the planets.
>
> Quadratic equations have a secret society that determines the number of cases for finding roots.
>
> The cases for finding roots of a quadratic equation are written in a lost ancient script.
>
> Quadratic equations have a secret formula that reveals the number of cases for finding roots.
>
> The number of cases for finding roots of a quadratic equation depends on the phase of the sun.
>
> Quadratic equations have a secret chant that reveals the number of cases for finding roots.
>
> The cases for finding roots of a quadratic equation are hidden in a crossword puzzle.
>
> Quadratic equations have a secret key that unlocks the number of cases for finding roots.
>
> The number of cases for finding roots of a quadratic equation depends on the song playing on the radio.
>
> Quadratic equations have a secret message that reveals the number of cases for finding roots.
>
> The cases for finding roots of a quadratic equation are stored in a secret vault.