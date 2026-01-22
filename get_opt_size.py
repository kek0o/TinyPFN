import torch
from model import TinyPFNModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def measure_cost(model, example_input):
    """Devuelve parámetros MB, activaciones MB y FLOPs"""
    from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis

    model.eval()
    
    # Params
    total_params = sum(p.numel() for p in model.parameters())
    param_mb = total_params * 4 / (1024**2)

    # FLOPs
    flops = FlopCountAnalysis(model, example_input)
    try:
        total_flops = flops.total()
    except:
        total_flops = None

    # Activations
    acts = ActivationCountAnalysis(model, example_input)
    try:
        total_acts = acts.total()
        act_mb = total_acts * 4 / (1024**2)
    except:
        act_mb = None

    return param_mb, act_mb, total_flops

def build_dummy_input(rows=32, features=10, batch=1):
    """Crea input de ejemplo para medir activaciones y FLOPs"""
    x = torch.randn(batch, rows, features).float().to(DEVICE)
    y = torch.zeros(batch, 1).float().to(DEVICE)
    return (x, y), rows

def test_tinypfn_config(embedding_size, num_heads, mlp_hidden_size, num_layers, num_outputs=2, features=10):
    model = TinyPFNModel(
        embedding_size=embedding_size,
        num_attention_heads=num_heads,
        mlp_hidden_size=mlp_hidden_size,
        num_layers=num_layers,
        num_outputs=num_outputs
    ).to(DEVICE)
    
    example_input, split = build_dummy_input(rows=32, features=features)
    param_mb, act_mb, flops = measure_cost(model, (example_input, split))
    
    print(f"TinyPFN config: E={embedding_size}, H={num_heads}, MLP={mlp_hidden_size}, L={num_layers}")
    print(f"  Param memory: {param_mb:.3f} MB")
    print(f"  Activation memory: {act_mb:.3f} MB")
    print(f"  FLOPs per inference: {flops/1e6:.2f} MFLOPs\n")
    return param_mb, act_mb, flops

# Ejemplo: probar varias configuraciones
for emb in [16, 32, 64]:
    for mlp in [16, 32, 64]:
        test_tinypfn_config(embedding_size=emb, num_heads=1, mlp_hidden_size=mlp, num_layers=1, features=30)
