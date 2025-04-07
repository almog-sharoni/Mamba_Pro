import argparse
import json
import os
import struct
import torch
import numpy as np  # Added import for numpy
import math

BFLOAT8_BIAS = 15

# -----------------------------------------------------------------------------
# common utilities

def serialize_fp32(file, tensor, scales_file=None):
    """ writes one fp32 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)

def serialize_int8(file, tensor, scales_file=None):
    """ writes one int8 tensor and its scale to file """
    tensor_cpu = tensor.detach().cpu()
    max_val = tensor_cpu.abs().max()
    if max_val == 0:
        scale = 1.0
    else:
        scale = max_val / 127  # int8 range is -128 to 127
    tensor_quant = torch.clamp((tensor_cpu / scale).round(), -128, 127).to(torch.int8)
    
    # Write int8 data
    b = tensor_quant.view(-1).numpy().astype(np.int8).tobytes()
    file.write(b)
    # print 10 elements for debugging
    print(f"Serializing tensor with scale: {scale}")
    print(f"Serialized {len(tensor_quant.view(-1))} elements to int8.")
    print(f"First 10 elements: {tensor_quant.view(-1)[:10]}")
    # Write scale as float32 to scales file
    if scales_file:
        scales_file.write(struct.pack('f', scale))

def floatToBFloat8(value: float) -> int:
    sign = 0
    if value < 0:
        sign = 1
        value = -value
    if value == 0.0:
        return sign << 7
    exp = int(math.floor(math.log2(value)))
    exp += BFLOAT8_BIAS
    if exp >= 31:
        # Infinity
        return (sign << 7) | (0x1F << 2)
    if exp <= 0:
        # Subnormal
        mantissa = int(value / (2 ** (1 - BFLOAT8_BIAS)) * 4) & 0x03
        return (sign << 7) | mantissa
    mantissa = int((value / (2 ** (exp - BFLOAT8_BIAS)) - 1.0) * 4) & 0x03
    return (sign << 7) | ((exp & 0x1F) << 2) | mantissa

def serialize_bfloat8(file, tensor, scales_file=None):
    tensor_cpu = tensor.detach().cpu().to(torch.float32)
    max_val = tensor_cpu.abs().max()
    scale = 1.0 if max_val == 0 else float(max_val / 2.0)
    # write scale
    if scales_file:
        scales_file.write(struct.pack('f', scale))
    print(f"Serializing tensor with scale: {scale}")
    arr = tensor_cpu.view(-1).numpy()
    bfloat8_bytes = bytearray()
    for val in arr:
        val_scaled = val / scale
        bfloat8_bytes.append(floatToBFloat8(val_scaled))
    file.write(bfloat8_bytes)
    print(f"Serialized {len(arr)} elements to bfloat8.")

# -----------------------------------------------------------------------------
# model export functions

def print_first_10_weights(tensor, layer_name, log_file):
    """Print the first 10 weights of a tensor for debugging."""
    weights = tensor.detach().cpu().view(-1).numpy()
    log_file.write(f"{layer_name} first 10 weights: {weights[:10]}\n")

def write_weights(file, model, key, log_file, serialize_func=serialize_fp32):
    """ writes the layer weights to file using the specified serialization function """
    print(f"writing {key} {list(model[key].shape)[::-1]}")
    # print_first_10_weights(model[key], key, log_file)
    serialize_func(file, model[key])

def write_layer_weights(file, scales_file, model, layer_fmt, n_layers, log_file, serialize_func=serialize_fp32):
    """ writes the layer weights to file for all layers using the specified serialization function """
    for n in range(n_layers):
        layer_key = layer_fmt % n
        print(f"writing {layer_key} {list(model[layer_key].shape)[::-1]} with {serialize_func.__name__}")
        # print_first_10_weights(model[layer_key], layer_key, log_file)
        serialize_func(file, model[layer_key], scales_file)

def model_export(model, config, filepath, scales_filepath, quantize=False, bfloat8=False):
    """
    Export the model weights in float32, int8, or bfloat8 .bin file to be read from C.
    If quantize is True, convolutional and fully connected layers are exported as INT8.
    If bfloat8 is True, convolutional and fully connected layers are exported as BFLOAT8.
    Scales are exported to a separate file when quantize is True.
    """
    version = 1

    out_file = open(filepath, 'wb')
    scales_file = open(scales_filepath, 'wb') if quantize or bfloat8 else None
    log_file = open("debug_weights.log", "w")

    # first write the header (256 bytes)
    # write magic, uint32 of "Mamb"
    out_file.write(struct.pack('I', 0x4d616d62))
    # write version
    out_file.write(struct.pack('i', version))

    # Get model dimensions from the state dict
    n_layers = len([k for k in model.keys() if k.startswith('mamba_layers') and k.endswith('.D')])
    dim = model['proj.weight'].shape[0]  # 136
    n_classes = model['fc.weight'].shape[0]  # 10
    input_dim = model['proj.weight'].shape[1]  # 23
    d_inner = model['mamba_layers.0.D'].shape[0]  # 272
    dt_rank = model['mamba_layers.0.dt_proj.weight'].shape[1]  # 9
    d_state = model['mamba_layers.0.A_log'].shape[1]  # 51
    d_conv = model['mamba_layers.0.conv1d.weight'].shape[2]  # 10

    # Write header parameters in correct order matching C code:
    # n_layers, n_classes, dim, input_dim, d_inner, dt_rank, d_state, d_conv
    header = struct.pack('iiiiiiii', 
                        n_layers,   # number of layers
                        n_classes,  # number of output classes
                        dim,        # embedding dimension
                        input_dim,  # input feature dimension
                        d_inner,    # inner dimension
                        dt_rank,    # delta rank
                        d_state,    # state dimension
                        d_conv)     # conv dimension
    out_file.write(header)

    # pad the rest with zeros
    pad = 256 - out_file.tell()
    assert pad >= 0
    out_file.write(b'\0' * pad)

    # Write initial projection layers
    write_weights(out_file, model, 'cls_token', log_file)
    write_weights(out_file, model, 'proj.weight', log_file)
    write_weights(out_file, model, 'proj.bias', log_file)

    # Convert A_log to A for all layers
    for n in range(n_layers):
        key = f'mamba_layers.{n}.A_log'
        model[f'mamba_layers.{n}.A'] = -torch.exp(model.pop(key))

    # Define layer patterns that should be quantized
    quantize_patterns = ['conv', 'proj', 'fc']

    # Function to determine if a key should be quantized
    def should_quantize(key):
        return quantize and any(pattern in key for pattern in quantize_patterns)

    def should_quantize_bfloat8(key):
        return bfloat8 and any(pattern in key for pattern in quantize_patterns)

    # Write mamba layer weights
    layer_weight_keys = [
        'mamba_layers.%d.in_proj.weight',
        'mamba_layers.%d.conv1d.weight',
        'mamba_layers.%d.conv1d.bias',
        'mamba_layers.%d.x_proj.weight',
        'mamba_layers.%d.dt_proj.weight',
        'mamba_layers.%d.dt_proj.bias',
        'mamba_layers.%d.A',
        'mamba_layers.%d.D',
        'mamba_layers.%d.out_proj.weight',
    ]

    for key_fmt in layer_weight_keys:
        example_key = key_fmt % 0
        if should_quantize_bfloat8(example_key):
            serialize_func = lambda f, t, s=None: serialize_bfloat8(f, t, s)
        else:
            serialize_func = serialize_int8 if should_quantize(example_key) else serialize_fp32
        write_layer_weights(out_file, scales_file, model, key_fmt, n_layers, log_file, serialize_func=serialize_func)
        print_first_10_weights(model[example_key], key_fmt, log_file)

    # Write layer norms
    for n in range(n_layers):
        write_weights(out_file, model, f'layer_norms.{n}.weight', log_file)

    # Write final classification layer
    serialize_func = serialize_int8 if should_quantize('fc.weight') else serialize_fp32
    write_weights(out_file, model, 'fc.weight', log_file, serialize_func=serialize_func)
    write_weights(out_file, model, 'fc.bias', log_file)

    out_file.close()
    if scales_file:
        scales_file.close()
    log_file.close()
    print(f"done. saved weights to {filepath}")
    if quantize or bfloat8:
        print(f"done. saved scales to {scales_filepath}")

# -----------------------------------------------------------------------------
# Load / import functions

def load_model(path):
    print(f"loading model from {path}")

    # load the model
    if os.path.isdir(path):
        filepath = os.path.join(path, 'pytorch_model.bin')
    else:
        filepath = path
    model = torch.load(filepath, map_location='cpu')

    # remove the 'backbone.' prefix from the keys
    unwanted_prefix = 'backbone.'
    for k, v in list(model.items()):
        if k.startswith(unwanted_prefix):
            model[k[len(unwanted_prefix):]] = model.pop(k)

    # get the path to the config file
    if os.path.isdir(path):
        config_path = os.path.join(path, 'config.json')
    else:
        config_path = os.path.join(os.path.dirname(path), 'config.json')  # Fix the syntax error here
    # load the config
    with open(config_path) as f:
        config = json.load(f)
    # rename config.n_layers to config.n_layers
    config['n_layers'] = config.pop('n_layer')
    config = argparse.Namespace(**config)    
    print(f"loaded model with {config.n_layers} layers, d_model={config.d_model}, vocab_size={config.vocab_size}")

    return model, config

def get_model_from_huggingface(model_name: str):
    """Download model from HuggingFace and get the path to the model file.
    The model name can be one of the following:
        'state-spaces/mamba-130m'
        'state-spaces/mamba-370m'
        'state-spaces/mamba-790m'
        'state-spaces/mamba-1.4b'
        'state-spaces/mamba-2.8b'
        'state-spaces/mamba-2.8b-slimpj'
    """
    from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
    from transformers.utils.hub import cached_file

    config_path = cached_file(model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
    model_path = cached_file(model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)

    return model_path

# -----------------------------------------------------------------------------
# CLI entrypoint

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=str, help="model name or folder where the model files are located", default="state-spaces/mamba-130m")
    parser.add_argument("destination", type=str, help="full path to the output file", default="model.bin")
    parser.add_argument("--int8", action='store_true', help="Export conv and fully connected layers as INT8")
    parser.add_argument("--bfloat8", action='store_true', help="Export conv and fully connected layers as BFLOAT8")
    args = parser.parse_args()

    # if the source starts with 'state-spaces/mamba-' then load the model from HuggingFace
    if args.source.startswith('state-spaces/mamba-'):
        model_path = get_model_from_huggingface(args.source)
    else:
        model_path = args.source

    model, config = load_model(model_path)

    if model is None:
        parser.error("Can't load input model!")

    # export
    scales_path = args.destination + ".scales" if args.int8 or args.bfloat8 else None
    model_export(model, config, args.destination, scales_path, quantize=args.int8, bfloat8=args.bfloat8)
    print(f"done. saved weights to {args.destination}")
    if args.int8 or args.bfloat8:
        print(f"done. saved scales to {scales_path}")