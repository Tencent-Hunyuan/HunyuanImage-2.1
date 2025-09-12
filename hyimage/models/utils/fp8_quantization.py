import os
import torch
import gc
import copy
import tqdm


def per_tensor_quantize(tensor: torch.Tensor):
    """Quantize a tensor using per-tensor static scaling factor.

    Args:
        tensor: The input tensor to quantize.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Quantized tensor and scale factor.
    """
    finfo = torch.finfo(torch.float8_e4m3fn)
    device = tensor.device

    if tensor.numel() == 0:
        min_val, max_val = (
            torch.tensor(-16.0, dtype=tensor.dtype, device=device),
            torch.tensor(16.0, dtype=tensor.dtype, device=device),
        )
    else:
        min_val, max_val = tensor.aminmax()

    amax = torch.maximum(min_val.abs(), max_val.abs())

    if torch.isnan(amax) or amax <= 0:
        raise RuntimeError(f"Illegal amax: {amax}")

    scale = finfo.max / amax.clamp(min=1e-12)
    qweight = (tensor * scale).clamp(min=finfo.min, max=finfo.max)

    deqweight = qweight.to(torch.float8_e4m3fn).clone()
    scale = torch.tensor(scale.float().reciprocal(), device=device)

    return deqweight, scale


def torch_fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor, out_dtype=torch.bfloat16, bias=None) -> torch.Tensor:
    """Perform FP8 GEMM operation using torch._scaled_mm.
    
    Args:
        a: Input tensor A.
        a_s: Scale factor for tensor A.
        b: Input tensor B.
        b_s: Scale factor for tensor B.
        out_dtype: Output data type.
        bias: Optional bias tensor.
        
    Returns:
        torch.Tensor: Result of the GEMM operation.
    """
    need_reshape = a.dim() == 3
    if need_reshape:
        batch_size = a.shape[0]
        A_input = a.reshape(-1, a.shape[-1])
    else:
        batch_size = None
        A_input = a

    output = torch._scaled_mm(
        A_input,
        b.t(),
        out_dtype=out_dtype,
        scale_a=a_s,
        scale_b=b_s,
        bias=bias,
    )
    # Handle tuple output from _scaled_mm
    if isinstance(output, tuple):
        output = output[0]

    if need_reshape:
        output = output.reshape(batch_size, output.shape[0] // batch_size, output.shape[1])

    return output


def fp8_gemm(A, A_scale, B, B_scale, bias, out_dtype, native_fp8_support=False, quant_type="fp8-per-tensor", origin_shape=None):
    """Perform FP8 GEMM operation with fallback to standard linear.
    
    Args:
        A: Input tensor A.
        A_scale: Scale factor for tensor A.
        B: Input tensor B.
        B_scale: Scale factor for tensor B.
        bias: Optional bias tensor.
        out_dtype: Output data type.
        native_fp8_support: Whether to use native FP8 support.
        quant_type: Quantization type.
        origin_shape: Original shape for reshaping.
        
    Returns:
        torch.Tensor: Result of the GEMM operation.
    """
    if A.numel() == 0:
        return torch.empty(size=(0, B.shape[0]), dtype=out_dtype, device=A.device)

    if native_fp8_support and quant_type == "fp8-per-tensor":
        output = torch_fp8_gemm(A, A_scale, B, B_scale, out_dtype, bias)
    else:
        output = torch.nn.functional.linear(
            A.to(out_dtype) * A_scale,
            B.to(out_dtype) * B_scale.to(out_dtype),
            bias=bias,
        )

    return output


class FP8DynamicLinear(torch.nn.Module):
    """FP8 Dynamic Linear layer with quantization support."""
    
    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.nn.Parameter,
        native_fp8_support: bool = False,
        quant_type: str = "fp8-per-tensor",
    ):
        """Initialize FP8 Dynamic Linear layer.
        
        Args:
            weight: Weight tensor.
            weight_scale: Scale factor for weights.
            bias: Bias parameter.
            native_fp8_support: Whether to use native FP8 support.
            quant_type: Quantization type.
        """
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        self.bias = bias
        self.native_fp8_support = native_fp8_support
        self.quant_type = quant_type

    def forward(self, x):
        """Forward pass with FP8 quantization.
        
        Args:
            x: Input tensor.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        if x.dtype == torch.float32:
            x = x.to(torch.bfloat16)

        if self.quant_type == "fp8-per-tensor":
            origin_shape = None
            qinput, x_scale = per_tensor_quantize(x)

        output = fp8_gemm(
            A=qinput,
            A_scale=x_scale,
            B=self.weight,
            B_scale=self.weight_scale,
            bias=self.bias,
            out_dtype=x.dtype,
            native_fp8_support=self.native_fp8_support,
            quant_type=self.quant_type,
            origin_shape=origin_shape,
        )

        return output


def replace_module(model: torch.nn.Module, name: str, new_module: torch.nn.Module):
    """Replace a module in the model with a new module.
    
    Args:
        model: The model containing the module.
        name: Name of the module to replace.
        new_module: The new module to replace with.
    """
    if "." in name:
        parent_name = name.rsplit(".", 1)[0]
        child_name = name[len(parent_name) + 1 :]
        parent = model.get_submodule(parent_name)
    else:
        parent_name = ""
        parent = model
        child_name = name

    setattr(parent, child_name, new_module)


def convert_fp8_linear(module, fp8_map_path=None):
    """Convert linear layers to FP8 quantized versions.
    
    Args:
        module: The module to convert.
        fp8_map_path: Path to FP8 scale map file.
    """
    setattr(module, "fp8_matmul_enabled", True)
    if fp8_map_path is None:
        fp8_map = {}
    else:
        if os.path.exists(fp8_map_path):
            if fp8_map_path.endswith(".safetensors"):
                import safetensors
                fp8_map = safetensors.torch.load_file(fp8_map_path)
            else:
                fp8_map = torch.load(fp8_map_path, map_location='cpu')
        else:
            raise ValueError(f"Invalid fp8_map path: {fp8_map_path}.")

    named_modules = list(module.named_modules())

    for name, linear in tqdm.tqdm(named_modules, desc="Quantizing weights"):
        if isinstance(linear, torch.nn.Linear):
            if "double_block" in name or "single_blocks" in name:
                if "embed" not in name:
                    if fp8_map_path is None:
                        if linear.weight.dtype != torch.float8_e4m3fn:
                            quant_weight, weight_scale = per_tensor_quantize(linear.weight)
                            fp8_map[name] = weight_scale
                        else:
                            raise ValueError(f"Invalid weight dtype: {linear.weight.dtype}")

                    else:
                        quant_weight = linear.weight.to(torch.float8_e4m3fn).clone()
                        weight_scale = fp8_map[name]

                    bias = copy.deepcopy(linear.bias) if linear.bias is not None else None

                    quant_linear = FP8DynamicLinear(
                        weight=quant_weight, 
                        weight_scale=weight_scale, 
                        bias=bias, 
                        native_fp8_support=False, 
                        quant_type="fp8-per-tensor"
                    )

                    replace_module(module, name, quant_linear)
                    del linear.weight
                    del linear.bias
                    del linear
    gc.collect()
    torch.cuda.empty_cache()
    return fp8_map
