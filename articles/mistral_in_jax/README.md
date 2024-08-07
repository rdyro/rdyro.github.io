<p align="center" style="font-size: 40px">
Optimizing LLM inference speed in float16 in JAX with Pallas
</p>

### Table of Contents

- [Introduction](#introduction)
    - [Decode Layer - The Core Computations](#decode-layer---the-core-computations)
    - [Setup: Initial Profile](#setup-initial-profile)
- [Optimizing Matrix Multiplication](#optimizing-matrix-multiplication)
    - [Choosing the right parameters](#choosing-the-right-parameters)
    - [Mixed-integer optimization: Choosing the best set of hyperparameters](#mixed-integer-optimization-choosing-the-best-set-of-hyperparameters)
    - [Result: Optimal selection from a small set](#result-optimal-selection-from-a-small-set)
    - [Result: Optimized Profile](#result-optimized-profile)
- [Conclusions](#conclusions)
  - [References](#references)

# Introduction

We aim to optimize the Local LLM model in JAX with `float16` precision to match or surpass `PyTorch` performance. We're using 
Mistral-7B instruction-tuned as the only test-case for this project.
The starting point is an existing implementation of the model in Flax from [Hugging Face](https://huggingface.co/rdyro/Mistral-7B-Instruct-v0.1).

> Why `float16`? 

This is a common format for fine-tuning and further development. While inference tends to be done in lower precision, `float16` is likely the highest precision needed, `float32` would rarely be used. In that sense, `float16` is an important format to optimize for.

<p align="center">
    <img src="notebooks/figs/tokens_per_sec.svg" style="width: 100%;max-width: 500px">
    <p align="center">Fig: An abstract figure. The existing JAX implementation is slower than PyTorch (on RTX 3090), but the optimized version outperforms it. The test cases consists of ~1k tokens of context and ~1.5k new tokens generated autoregressively.</p>
</p>

### Decode Layer - The Core Computations

<p align="center">
  <img src="notebooks/figs/decoder_layer_graph/decoder_layer_graph.svg" style="width: 100%">
  <p align="center">Fig: The computation graph of the decode layer. The LLM consists of a stack of these layers, often with identical parameter shapes (but different weight values).</p>
</p>

Importantly, in causal decoding, there are two stages:
- **context ingestion** - a full sequence attention computation
- **single-query decoding** - the computation of the next token, repeatedly in a loop

Because the **single-query decoding** dominates the runtime in the context of a chat-like application, we focus on optimizing this part of the model and won't analyze the context ingestion part. We, of course, do not want to make it slower, even if it experiences runtime regression, it likely won't be noticeable in the overall performance.

| Stage             | Runtime          |
|-------------------|------------------|
| Context ingestion | 574 ms           |
| Generation        | 62,000 ms (62 s) |

Both context ingestion and generation involve a pass through the model, for different computation shapes. Thus, regardless, the optimization should focus on accelerating the single-pass runtime of the model. However, clearly, the computation shapes to optimized for are of the generation kind – the single-query decoding.

The core computation – the single-pass through the model – consists of two phases:

**Phase 1:** The attention operation, which takes the form 

$$
\text{softmax}\left(Q  K^T \right) V
$$

where $Q \in \mathbb{R}^{n \times d}$, $K \in \mathbb{R}^{m \times d}$, and $V
\in \mathbb{R}^{m \times d}$ are the query, key, and value matrices,
respectively. The output is a matrix of size $n \times d$.

Because the softmax operation is applied to the rows of the matrix, the number
of queries need not match the number of keys and values. This is particularly
useful in the case of single-query decoding, where the query size is 1.

**Phase 2:** Output Multi-layer Perceptron (MLP). Several possibilities are
*common, but for Mistral, the MLP is a SiLU activated feed-forward network
*without bias.

$$
W_\text{down} \text{SiLU}\left(W_\text{up} x \odot W_\text{gate} x \right)
$$

where $W_\text{up} \in \mathbb{R}^{d \times 4d}$, $W_\text{gate} \in
\mathbb{R}^{d \times 4d}$, and $W_\text{down} \in \mathbb{R}^{4d \times d}$ are
the weight matrices of the MLP.

We can benchmark the performance of the model to determine the first Phase to tackle for optimization.

### Setup: Initial Profile

<p align="center">
    <img src="notebooks/figs/single_decode_profile_default_annotated.svg" style="width: 100%">
    <p align="center">Fig: Profile of the decode layer with default settings for a single-query decoding. The MLP phase is the bottleneck.</p>
</p>

Unoptimized single-pass (the main computation scope) runtime: **42 ms**.

<p align="center">
  <img src="notebooks/figs/decoder_layer_graph/decoder_layer_graph_with_highlight.svg" style="width: 100%">
  <p align="center">Fig: The highlighted blocks dominate the runtime of the decode layer. The MLP phase is the bottleneck.</p>
</p>

# Optimizing Matrix Multiplication

A simple matrix multiplication has three input dimensions:
- $m$ - the number of rows in the left matrix
- $k$ - the contracting dimension
- $n$ - the number of columns in the right matrix

Because we established the MLP phase is the bottleneck and because the sole computationally expensive operation in the MLP phase is the matrix multiplication, we focus on optimizing this matrix multiplication (matmul) operation.

JAX exposes Pallas - a Triton-like higher level kernel language.

On a high level, the strategy is to (1) implement a simple matrix multiplication kernel and (2) tune the parameters of the kernel for particular hardware and input shapes.

<p align="center">
    <img src="notebooks/figs/matrix_multiplication_sketch_with_main_dimensions.svg" style="width: 100%">
    <p align="center">Fig: A visual representation of a simple matrix multiplication kernal with an inner-loop accumulating the matrix multiplication of a slice of the matrices by multiplying smaller blocks of the matrices.</p>
</p>

 The figure above shows probably the simplest possible matrix multiplication kernel. The hope here is that the Pallas language (or in fact, the underlying Triton compiler) can optimize the warp-level parallelism and memory access patterns better to make this simple kernel efficient.


The kernel is implemented in Pallas as follows:

```python
def matmul_kernel(x_ref, A_ref, o_ref, block_x: int, block_a: int, block_d: int):
    row_id, col_id = pl.program_id(0), pl.program_id(1)
    col_slice = pl.dslice(col_id * block_d, block_d)
    A_mask_j = (col_id * block_d + jnp.arange(block_d) < A_ref.shape[1])[None, :]
    a_i = jnp.arange(block_a)
    x_mask_i = (row_id * block_x + jnp.arange(block_x) < x_ref.shape[0])[:, None]
    x_j = jnp.arange(block_a)

    def body_i(start_i, carry_i):
        o_prev = carry_i
        x_mask = x_mask_i & (start_i * block_a + x_j < x_ref.shape[1])[None, :]
        x = pl.load(
            x_ref,
            (pl.dslice(row_id * block_x, block_x), pl.dslice(start_i * block_a, block_a)),
            mask=x_mask,
        )
        a_mask = A_mask_j & (start_i * block_a + a_i < A_ref.shape[0])[:, None]
        a = pl.load(A_ref, (pl.dslice(start_i * block_a, block_a), col_slice), mask=a_mask)
        return pl.dot(x, a) + o_prev

    o_init = jnp.zeros((block_x, block_d), dtype=jnp.float32)
    o = lax.fori_loop(0, pl.cdiv(A_ref.shape[0], block_a), body_i, o_init)
    o_slice = (pl.dslice(row_id * block_x, block_x), pl.dslice(col_id * block_d, block_d))
    o_mask = (row_id * block_x + jnp.arange(block_x) < o_ref.shape[0])[:, None] & (
        col_id * block_d + jnp.arange(block_d) < o_ref.shape[1]
    )
    pl.store(o_ref, o_slice, o.astype(o_ref.dtype), mask=o_mask)
```

### Choosing the right parameters

A Pallas kernel optimization, especially for such a simple kernel, might be
sensitive to the input dimensions. Picking a single set of kernel
hyperparameters is unlikely to be optimal for all input shapes. 

JAX recompiles a program if the dimensions change, so we can choose kernel
hyperparameters based on the input dimensions. On one extreme, we could pick a
large set of combinations of input dimensions and find the optimal kernel
hyperparameters for each combination. On the other extreme, we could pick a
single set of kernel hyperparameters, and that works well for all input
dimensions. The first option will squeeze out the last bit of performance, but
it might poorly generalize to unseen input shapes. The second option is likely to
be suboptimal, but is much more likely to generalize.

We choose a middle ground, a small set of input dimensions, and a small set of
kernel hyperparameters. We then test every input dimensions combination on every
hyperparameter combination. Finally, to improve generalize, we want to select
only 4 hyperparameter sets and create a map of the best hyperparameters for each
input shape. This requires us to find a small set of hyperparameters that, when
selected for every input dimension combination, will give us the best
performance. This is a mixed-integer optimization problem.

### Mixed-integer optimization: Choosing the best set of hyperparameters

Problem: pick a set of hyperparameters that is less than 4, such that the total speedup is maximized when for each input dimension combination, we choose one of the hyperparameter sets from that small set (of size 4).

In principle, this requires testing all possible combinations of size 4 in a set
45 possible hyperparameter combinations -- $\binom{45}{4} = 1.5 \times 10^5$.
This is large but not necessarily infeasible to iterate over. However, we need
to consider another hyperparameter possibility, the fallback to the native
`matmul` implementation. This increases the number of hyperparameter
combinations to $\binom{46}{4} = 1.4 \times 10^6$. This is still feasible to
iterate over, but we turn to a mixed-integer optimization solver to find the
optimal set of hyperparameters instead.


### Result: Optimal selection from a small set

<p align="center">
    <img src="notebooks/figs/optimal_config_visualized.svg" style="width: 100%">
    <p align="center">Fig: Optimal kernel hyperparameter configuration for each of the input shape test point. Configuration 0 is the in-built matmul. Percentage numbers denote how much slower the in-built implementation is from the optimized kernel. Tested on Nvidia RTX 3090.</p>
</p>

As seen from the results, the optimized kernel can be significantly faster than the in-built matmul implementation. The speedup is particularly pronounced for larger input shapes.

### Result: Optimized Profile

<p align="center">
    <img src="notebooks/figs/single_decode_profile_fast_annotated.svg" style="width: 100%">
    <p align="center">Fig: Profile of the decode layer with optimized matrix multiplication for a single-query decoding. The MLP phase takes roughly the same time as the attention phase.</p>
</p>

Optimized single-pass (the main computation scope) runtime: **25 ms**.

# Conclusions

- **In the context of local LLMs, the speed of individual operations is crucial.** The inference pass is not particularly complex, so even uncompiled PyTorch model is able to achieve a very high throughput. JAX, despite compilation, suffers from a slow default implementation of the `matmul` operation for some input sizes – at least on RTX 3090 as of version 0.4.29.

- What optimizations did not work:
    - **Similar optimization done on RTX 4090 did not yield any significant speedup.** Likely, the native `matmul` implementation is already optimized for that hardware.
    - **Optimized attention did not yield a speed-up.** I attempted similar tuning for the multi-head attention operation. However, despite microbenchmarks showing a significant speedup, the overall performance of the model did not improve or sometimes dropped. This might have to do with how JAX compiles the program -- (I am speculating that) the Pallas kernel is a unit and cannot be partitioned and rolled into other operations in the computation graph. Similarly, the custom single-query attention bespoke Pallas kernel yielded significant improvements (up to 2x faster) in microbenchmarks, but the overall performance of the model did not improve.
    - **Replacing smaller linear layers with an optimized Pallas kernel resulted in a slight slowdown.** Again, this might have to do with full computation graph optimization that becomes impossible once a linear layer is partitioned into a Pallas kernel as a black-box computation unit.

- Faster model compilation & loading is important:
    - **The default way to load the Mistral model is slow,** it (as of 2024-07-02) performs two passes through the model on the default device. This is very suboptimal if the default device is a GPU, because the data type is `float32` and memory usage explodes to over 30GB (which few single GPUs have). If the default device is the CPU, the pass is incredibly slow. It's possible to solve this by using a `_do_init=False` initialization, but it is user-hostile and documented scantily.


## References

- [Accelerating matrix multiplication in Triton](https://pytorch.org/blog/accelerating-triton/)
- [Flash-decoding, a useful reference for implementing optimized single-query attention](https://pytorch.org/blog/flash-decoding/)
- [The Mistral AI, authors of the Mistral LLM model](https://mistral.ai/)