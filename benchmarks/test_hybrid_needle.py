
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
from optiq.core.turbo_kv_cache import TurboQuantKVCache
from optiq.core.turbo_quant import TurboQuantProd, TurboQuantMSE
import math

MODEL_ID = "mlx-community/Qwen2.5-3B-Instruct-4bit"

def generate_orthogonal_matrix(d, seed=42):
    key = mx.random.key(seed)
    G = mx.random.normal(shape=(d, d), key=key)
    Q, R = mx.linalg.qr(G, stream=mx.cpu)
    mx.eval(Q, R)
    diag_signs = mx.sign(mx.diag(R))
    Q = Q * diag_signs[None, :]
    mx.eval(Q)
    return Q

class OrthogonalTurboQuantProd(TurboQuantProd):
    def __init__(self, d, bits, seed=42, damping=0.7):
        super().__init__(d, bits, seed)
        self.qjl = generate_orthogonal_matrix(d, seed + 1000)
        self.damping = damping
    def dequantize(self, mse_indices, qjl_signs, residual_norms, norms):
        x_mse = self.mse.dequantize(mse_indices, mx.ones_like(norms))
        scale = math.sqrt(math.pi / 2.0) / math.sqrt(self.d)
        qjl_correction = (scale * residual_norms * (qjl_signs.astype(mx.float16) @ self.qjl))
        return (x_mse + qjl_correction * self.damping) * norms

class HybridTurboCache(TurboQuantKVCache):
    def __init__(self, head_dim, bits, seed=42, k_damping=0.7):
        super().__init__(head_dim, bits, use_qjl=True, seed=seed)
        self.k_quantizer = OrthogonalTurboQuantProd(head_dim, 4, seed, damping=k_damping)
        self.v_quantizer = TurboQuantMSE(head_dim, 4, seed + 500)

def build_needle_prompt(n_tokens, tokenizer):
    needle = "IMPORTANT FACT: The experimental blueberry accession designated UF-B4291 carries a novel disease-resistance allele called FROSTBLOCK-7 in the VcMYB4 locus."
    question = "What is the name of the disease-resistance allele found in blueberry accession UF-B4291, and in which locus was it identified?"
    filler = "Blueberry breeding is a complex process involving genomic selection and phenotyping. " * 1000
    filler_tokens = tokenizer.encode(filler)
    target_filler = n_tokens - len(tokenizer.encode(needle)) - len(tokenizer.encode(question)) - 50
    filler_tokens = filler_tokens[:target_filler]
    mid = len(filler_tokens) // 2
    prompt_tokens = filler_tokens[:mid] + tokenizer.encode(needle) + filler_tokens[mid:] + tokenizer.encode(question)
    return mx.array(prompt_tokens)

def run_needle_test(model, tokenizer, n_ctx=2000):
    print(f"\nHybrid Needle Test: n_ctx={n_ctx}")
    n_layers = len(model.layers)
    head_dim = 128
    caches = [HybridTurboCache(head_dim, 4, seed=42+i) for i in range(n_layers)]
    prompt_tokens = build_needle_prompt(n_ctx, tokenizer)
    sampler = make_sampler(temp=0.0)
    generated_text = ""
    for token, _ in generate_step(prompt_tokens, model, max_tokens=100, sampler=sampler, prompt_cache=caches):
        tok_int = token.item() if hasattr(token, "item") else int(token)
        generated_text += tokenizer.decode([tok_int])
        if "\n" in generated_text: break
    print(f"Response: {generated_text.strip()}")
    passed = "FROSTBLOCK-7" in generated_text and "VcMYB4" in generated_text
    print(f"Result: {'✅ PASSED' if passed else '❌ FAILED'}")
    return passed

if __name__ == "__main__":
    model, tokenizer = load(MODEL_ID)
    run_needle_test(model, tokenizer, n_ctx=800)
