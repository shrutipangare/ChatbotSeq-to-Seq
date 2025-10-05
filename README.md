# Chatbot with Seq2Seq and Attention

A PyTorch implementation of a conversational chatbot using sequence-to-sequence architecture with Luong attention mechanisms, trained on the Cornell Movie Dialogs Corpus.


## Overview

This project implements an end-to-end chatbot that can engage in conversations by learning from movie dialogue exchanges. The model uses an encoder-decoder architecture with attention mechanisms to generate contextually relevant responses.

The project follows the [PyTorch Chatbot Tutorial](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html) and extends it with:
- Systematic hyperparameter optimization using Weights & Biases
- Performance profiling and optimization
- TorchScript conversion for production deployment


## Requirements

```txt
python>=3.7
torch>=1.9.0
numpy>=1.19.0
matplotlib>=3.3.0
wandb>=0.12.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/chatbot-seq2seq.git
cd chatbot-seq2seq
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the Cornell Movie Dialogs Corpus (automatically handled by the script):
```bash
python download_data.py
```

## Dataset

This project uses the [Cornell Movie Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html), which contains:
- 220,579 conversational exchanges
- 304,713 utterances
- 10,292 movie character pairs
- 617 movies

The dataset is preprocessed to create question-answer pairs suitable for training a conversational model.

## Model Architecture

### Encoder
- **Type**: GRU-based bidirectional encoder
- **Input**: Tokenized word sequences
- **Output**: Hidden states for each time step

### Decoder
- **Type**: GRU-based decoder with attention
- **Attention**: Luong (multiplicative) attention mechanism
- **Output**: Generated response sequences

### Key Components
```
Input Sentence → Embedding → Encoder (GRU) → Context Vector
                                                    ↓
                                            Attention Mechanism
                                                    ↓
Target Sentence ← Embedding ← Decoder (GRU) ← Attended Context
```

## Training

### Hyperparameters

The model supports various hyperparameters that can be tuned:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `clip` | Gradient clipping threshold | 50.0 |
| `decoder_learning_ratio` | Learning rate ratio for decoder | 5.0 |
| `learning_rate` | Base learning rate | 0.0001 |
| `optimizer` | Optimization algorithm | SGD/Adam |
| `teacher_forcing_ratio` | Probability of using teacher forcing | 1.0 |
| `hidden_size` | Hidden layer size | 500 |
| `encoder_n_layers` | Number of encoder layers | 2 |
| `decoder_n_layers` | Number of decoder layers | 2 |
| `dropout` | Dropout probability | 0.1 |

### Training Command

```bash
python train.py \
    --epochs 50 \
    --batch-size 64 \
    --learning-rate 0.0001 \
    --hidden-size 500 \
    --clip 50.0 \
    --teacher-forcing-ratio 1.0
```

### With Weights & Biases Logging

```bash
python train.py \
    --use-wandb \
    --wandb-project "chatbot-seq2seq" \
    --wandb-entity "your-username"
```

## Hyperparameter Tuning Results

### Methodology

Hyperparameter optimization was performed using Weights & Biases Sweeps with the following strategy:
- **Search Method**: Bayes optimization
- **Metric**: Minimize validation loss
- **Number of Runs**: 20 different configurations

### Key Findings

![Parallel Coordinates Plot](https://github.com/shrutipangare/ChatbotSeq-to-Seq/blob/main/W%26B%20Chart%203_17_2025%2C%2010_04_52%20PM.png)
*Parallel coordinates visualization showing the relationship between hyperparameters and final loss*

![Training Loss Curves](images/training_loss.png)
*Loss curves across different hyperparameter configurations showing convergence patterns*

Based on W&B sweeps, the most impactful hyperparameters were identified through systematic experimentation:

1. **Gradient Clipping (`clip`)**
   - Range tested: 10-100
   - Impact: Highest importance - critical for convergence stability
   - Optimal: 25-50

2. **Learning Rate (`learning_rate`)**
   - Range tested: 0.0001-0.001
   - Impact: High importance - key factor in optimization speed and final loss
   - Optimal: 0.00025

3. **Teacher Forcing Ratio (`teacher_forcing_ratio`)**
   - Range tested: 0.5-1.0
   - Impact: High importance - balances training stability and model generalization
   - Optimal: 0.5

4. **Decoder Learning Ratio (`decoder_learning_ratio`)**
   - Range tested: 1.0-10.0
   - Impact: Moderate importance
   - Optimal: 3.0-5.0

5. **Optimizer**
   - Tested: SGD vs Adam
   - Finding: Adam optimizer showed better performance for this task

### Best Configuration

```python
best_config = {
    'clip': 25-50,
    'decoder_learning_ratio': 3.0-5.0,
    'learning_rate': 0.00025,
    'optimizer': 'adam',
    'teacher_forcing_ratio': 0.5,
    'hidden_size': 500,
    'encoder_n_layers': 2,
    'decoder_n_layers': 2,
    'dropout': 0.1
}
```

### Parameter Importance Analysis

According to the W&B parameter importance chart:
- **`_items.clip`**: Highest correlation with final loss
- **`clip`**: Second highest importance
- **`_items.learning_rate`** and **`learning_rate`**: Significant impact on convergence

## Performance Optimization

### Profiling

The model was profiled using PyTorch Profiler to identify computational bottlenecks:

```bash
python profile.py --model-path checkpoints/best_model.pth
```

**Profiling Results:**
- Memory consumption analysis across operators
- Operator-level time breakdown
- GPU utilization metrics
- Identification of performance bottlenecks

**Key Insights:**
- GRU operations consume ~60% of training time
- Attention mechanism adds ~15% computational overhead
- Embedding lookups are memory-intensive but fast

### TorchScript Conversion

The trained model was converted to TorchScript for production deployment:

#### Tracing vs Scripting

**Tracing:**
```python
traced_model = torch.jit.trace(model, example_inputs)
```
- Works by running your model once with example inputs, recording the operations that occur, and creating a static graph of these operations
- Pros: Faster, simpler, effective for models with static control flow
- Cons: Has limitations - doesn't capture dynamic control flow
- **Use case**: Works well for encoder steps with fixed operations

**Scripting:**
```python
scripted_model = torch.jit.script(model)
```
- Directly analyzes and compiles your Python code to TorchScript, preserving dynamic control flow
- Pros: Offers more comprehensive compilation, preserves control flow, more flexible
- Cons: Slightly slower compilation, requires TorchScript-compatible code
- **Use case**: Necessary for the decoder where generation behavior depends on previous outputs

#### Key Modifications for TorchScript Conversion

1. **Type Annotations**: Add explicit type annotations to methods and function parameters to help the TorchScript compiler understand tensor shapes and types

2. **Control Flow Compatibility**: Replace Python-specific control flows with TorchScript-compatible alternatives:
   - Use `torch.jit.script_if` instead of Python if-statements on non-tensor values
   - Replace Python lists/dictionaries with TorchScript-compatible `torch.List` and `torch.Dict`

3. **Module Structure Changes**:
   - Move helper functions inside the module class or declare them with `@torch.jit.script`
   - Ensure any external function calls are to TorchScript-compatible functions

4. **Remove Dynamic Attributes**: TorchScript doesn't support dynamically adding attributes to objects at runtime

5. **Replace Unsupported Operations**: Replace operations like string formatting and print statements with TorchScript-compatible alternatives

#### Performance Comparison

| Framework | CPU Latency (ms) | GPU Latency (ms) | CPU Speedup | GPU Speedup |
|-----------|------------------|------------------|-------------|-------------|
| PyTorch (Eager) | 292.45 | 18.73 | 1.0x | 1.0x |
| TorchScript | 297.12 | 12.89 | 0.98x | **1.45x** |

**Key Findings:**
- **GPU Performance**: TorchScript provides ~1.45x speedup on GPU (18.73ms → 12.89ms)
- **CPU Performance**: Minimal difference on CPU, slightly slower due to compilation overhead
- **Recommendation**: Use TorchScript for GPU deployment for optimal inference performance

**Convert Your Model:**
```bash
python convert_to_torchscript.py \
    --model-path checkpoints/best_model.pth \
    --output-path checkpoints/model_traced.pt \
    --method trace
```

## Usage

### 1. Training the Model

Basic training:
```bash
python train.py --epochs 50 --batch-size 64
```

Advanced training with all options:
```bash
python train.py \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.0001 \
    --hidden-size 500 \
    --encoder-layers 2 \
    --decoder-layers 2 \
    --dropout 0.1 \
    --clip 50.0 \
    --teacher-forcing-ratio 1.0 \
    --save-dir checkpoints \
    --use-wandb
```

### 2. Interactive Chat

Start chatting with the trained model:
```bash
python chat.py --model-path checkpoints/best_model.pth
```

Example conversation:
```
> Hello!
Bot: Hi there! How are you doing?

> What's your name?
Bot: I don't have a name, but you can call me Bot!

> Tell me a joke
Bot: I'm not very good at jokes, but I'll try my best!
```

### 3. Running Hyperparameter Sweeps

Configure sweep:
```yaml
# sweep_config.yaml
program: train.py
method: bayes
metric:
  name: loss
  goal: minimize
parameters:
  learning_rate:
    min: 0.0001
    max: 0.001
  clip:
    min: 10
    max: 100
  teacher_forcing_ratio:
    min: 0.5
    max: 1.0
```

Run sweep:
```bash
wandb sweep sweep_config.yaml
wandb agent your-sweep-id
```

### 4. Evaluating the Model

Evaluate on test set:
```bash
python evaluate.py \
    --model-path checkpoints/best_model.pth \
    --test-data data/test.txt
```

### 5. Using TorchScript Model

Load and use the optimized model:
```python
import torch

# Load TorchScript model
model = torch.jit.load('checkpoints/model_traced.pt')

# Use for inference
output = model(input_tensor)
```

## Results

### Training Metrics

- **Final Training Loss**: ~3.6
- **Iterations**: 4000
- **Best Performance**: Achieved with Adam optimizer, learning rate 0.00025, clip 25-50
- **Training Environment**: GPU-accelerated training

### Model Performance

- Successfully trained chatbot capable of generating contextually appropriate responses
- Achieved convergence with optimized hyperparameters (final loss: 3.60738)
- Performance improvement of ~1.45x with TorchScript conversion on GPU
- Identified key hyperparameters affecting model convergence: clip, learning_rate, and teacher_forcing_ratio
- Stable training with gradient clipping and teacher forcing

### Qualitative Results

The model demonstrates:
- Coherent short responses to greetings and simple questions
- Contextual understanding of basic conversational patterns
- Appropriate sentiment matching (positive responses to positive inputs)
- Some limitations with long-term context and complex queries


## References

### Papers
- Luong, M. T., Pham, H., & Manning, C. D. (2015). [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025). arXiv:1508.04025.
- Sutskever, I., Vinyals, O., & Le, Q. V. (2014). [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215). NIPS 2014.

### Resources
- [PyTorch Chatbot Tutorial](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html)
- [Cornell Movie Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [TorchScript Documentation](https://pytorch.org/docs/stable/jit.html)


⭐ If you find this project helpful, please consider giving it a star!
