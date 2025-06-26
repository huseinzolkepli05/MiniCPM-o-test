# MiniCPM-o-test

MiniCPM-o-test

# Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Setup Modal:

```bash
modal setup
```

3. Run inference:

```bash
export HF_TOKEN=your_hf_token
modal run inference.py
```

## Result

```
Wrote output.wav to mle-take-home/output.wav with length 2.3893333333333335
Total time taken: 2.7777853730000004
Time to first byte: 1.5479438859999988
Realtime Factor: 1.1625775835658483
```

What I did only add warmup,

```python
@app.local_entrypoint()
def main():
    engine = MinicpmInferenceEngine()
    
    # warmup
    engine.run.remote("Hi, how are you? testing testing")
    
    # actual running
    result = engine.run.remote("Hi, how are you?")

    PARENT_DIR = Path(__file__).parent

    sf.write(PARENT_DIR / "output.wav", result["audio_array"], result["sample_rate"])
    audio_duration_seconds = len(result["audio_array"]) / result["sample_rate"]
    print(f"Wrote output.wav to {PARENT_DIR / 'output.wav'}")
    print(f"Time to first byte: {result['time_to_first_byte']}")
    print(f"Realtime Factor: {result['total_time'] / audio_duration_seconds}")
```

While if not warmup,

```
Wrote output.wav to /home/husein/ssd4/mle-take-home/output.wav with length 1.9626666666666666
Total time taken: 6.996136690999997
Time to first byte: 6.143056928999997
Realtime Factor: 3.564607689028531
```

## What I learnt

Assume we follow the same `torch==2.5.1` and `transformers==4.44.2` versions from [inference.py](inference.py).

### Static cache torch compile LLM does not work

The snippet,

```python
model.llm.forward = torch.compile(model.llm.forward, mode="reduce-overhead", fullgraph=True)
```

#### prepare_inputs_for_generation bugs

```python
if past_key_values is not None:
    if isinstance(past_key_values, Cache):
        cache_length = past_key_values.get_seq_length()
        past_length = past_key_values.seen_tokens
    else:
        cache_length = past_length = past_key_values[0][0].shape[2]
```

`past_key_values.seen_tokens` is None for static cache, there is no `_seen_tokens` attribute been set for static cache.

#### Dynamo break for logging.warning_once

For new UUID, it will set `is_first` as True and this will set `self.audio_past_key_values = None` at https://huggingface.co/openbmb/MiniCPM-o-2_6/blob/9a8db9d033b8e61fa1f1a9f387895237c3de98a2/modeling_minicpmo.py#L1148

When do forward at https://huggingface.co/openbmb/MiniCPM-o-2_6/blob/9a8db9d033b8e61fa1f1a9f387895237c3de98a2/modeling_minicpmo.py#L1159,

```python
outputs = self.llm(
    past_key_values=self.llm_past_key_values,
    inputs_embeds=inputs_embeds,
    attention_mask=attention_mask,
    position_ids=None,  # position_ids,
    use_cache=True,
    return_dict=True,
)
```

Because `self.llm_past_key_values` is None, it will hit this line https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/qwen2/modeling_qwen2.py#L872,

```
if use_cache and not isinstance(past_key_values, Cache) and not self.training:
    use_legacy_cache = True
    past_key_values = DynamicCache.from_legacy_cache(past_key_values)
    logger.warning_once(
        "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
        "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
    )
```

This will break torch compile. PyTorch 2.5.1 not able to ignore logger yet while PyTorch 2.6 already support to ignore logging by simply,

```python
from transformers.models.qwen2.modeling_qwen2 import logger
torch._dynamo.config.ignore_logger_methods.add(logger.warning_once)
```

#### Fullgraph break

Even if we fix `prepare_inputs_for_generation` by monkey patching,

```python
import types
from transformers.models.llama.modeling_llama import _prepare_4d_causal_attention_mask_with_cache_position

def prepare_inputs_for_generation(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    cache_position=None,
    position_ids=None,
    use_cache=True,
    **kwargs,
):
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = 0
        else:
            cache_length = past_length = past_key_values[0][0].shape[2]

        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

            position_ids = position_ids.clone(memory_format=torch.contiguous_format)

    if inputs_embeds is not None and cache_position[0] == 0:
        model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
    else:
        model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

    if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
        if model_inputs["inputs_embeds"] is not None:
            batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
            device = model_inputs["inputs_embeds"].device
        else:
            batch_size, sequence_length = model_inputs["input_ids"].shape
            device = model_inputs["input_ids"].device

        dtype = self.lm_head.weight.dtype
        min_dtype = torch.finfo(dtype).min

        attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=past_key_values.get_max_length(),
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=batch_size,
        )

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
        }
    )
    return model_inputs

model.llm.prepare_inputs_for_generation = types.MethodType(prepare_inputs_for_generation, model.llm)
```

Because the `prepare_inputs_for_generation` excluded `cache_position`, this is trigger data dependent issue,

```
Unsupported: data dependent operator: aten._local_scalar_dense.default; to enable, set torch._dynamo.config.capture_scalar_outputs = True

from user code:
   File "/home/husein/.local/lib/python3.10/site-packages/transformers/models/qwen2/modeling_qwen2.py", line 1104, in forward
    outputs = self.model(
  File "/home/husein/.local/lib/python3.10/site-packages/transformers/models/qwen2/modeling_qwen2.py", line 882, in forward
    cache_position = torch.arange(
```

This is because `torch.arange` depends on another aggregation function from static cache to calculate,

```
if cache_position is None:
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
    )
```

By default HuggingFace Transformers `prepare_inputs_for_generation` will generate a torch arange tensor by using input shape, not from `get_seq_length()`.

#### Dynamic control flow is not supported

```
UserError: Dynamic control flow is not supported at the moment. Please use functorch.experimental.control_flow.cond to explicitly capture the control flow. For more information about this error, see: https://pytorch.org/docs/main/generated/exportdb/index.html#cond-operands

from user code:
   File "/home/husein/.local/lib/python3.10/site-packages/transformers/models/qwen2/modeling_qwen2.py", line 1104, in forward
    outputs = self.model(
  File "/home/husein/.local/lib/python3.10/site-packages/transformers/models/qwen2/modeling_qwen2.py", line 915, in forward
    layer_outputs = decoder_layer(
  File "/home/husein/.local/lib/python3.10/site-packages/transformers/models/qwen2/modeling_qwen2.py", line 655, in forward
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
  File "/home/husein/.local/lib/python3.10/site-packages/transformers/models/qwen2/modeling_qwen2.py", line 552, in forward
    kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
  File "/home/husein/.local/lib/python3.10/site-packages/transformers/cache_utils.py", line 76, in get_usable_length
    if max_length is not None and previous_seq_length + new_seq_length > max_length:

Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
```

Because there is a condition at https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/qwen2/modeling_qwen2.py#L552, so this will break torch compile.

### Static cache make it slower

I assumed preallocate the memory should be much faster compared to dynamic caching that dynamically grow the size by concating while static cache just need to assign by using index.

**The generation is not correct because the modeling source code does not maintain proper position ids for static cache position, but nice to try**.

```python
import types
from transformers.models.llama.modeling_llama import _prepare_4d_causal_attention_mask_with_cache_position
from transformers import StaticCache
import time
import numpy as np

def prepare_inputs_for_generation(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    cache_position=None,
    position_ids=None,
    use_cache=True,
    **kwargs,
):
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = 0
        else:
            cache_length = past_length = past_key_values[0][0].shape[2]

        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

            position_ids = position_ids.clone(memory_format=torch.contiguous_format)

    if inputs_embeds is not None and cache_position[0] == 0:
        model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
    else:
        model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

    if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
        if model_inputs["inputs_embeds"] is not None:
            batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
            device = model_inputs["inputs_embeds"].device
        else:
            batch_size, sequence_length = model_inputs["input_ids"].shape
            device = model_inputs["input_ids"].device

        dtype = self.lm_head.weight.dtype
        min_dtype = torch.finfo(dtype).min

        attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=past_key_values.get_max_length(),
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=batch_size,
        )

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
        }
    )
    return model_inputs

model.llm.prepare_inputs_for_generation = types.MethodType(prepare_inputs_for_generation, model.llm)

past_key_values = StaticCache(
    config=model.llm.config,
    max_batch_size=1,
    max_cache_len=2048,
    device=model.llm.device,
    dtype=model.llm.dtype
)

text = 'Hi, how are you?'
input_ids = _tokenizer(text, return_tensors="pt").to("cuda")
# warmup
model.llm(**input_ids)

times = []
for i in range(3):
    past_key_values.reset()
    before = time.time()
    model.llm.generate(**input_ids, past_key_values = past_key_values)
    times.append(time.time() - before)
print(np.mean(times))
```

Output,

```
0.4221157232920329
```

While dynamic cache,

```
text = 'Hi, how are you?'
input_ids = _tokenizer(text, return_tensors="pt").to("cuda")
# warmup
model.llm(**input_ids)

times = []
for i in range(3):
    before = time.time()
    model.llm.generate(**input_ids)
    times.append(time.time() - before)
print(np.mean(times))
```

Output,

```
0.2912732760111491
```
