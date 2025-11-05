from modelscope import AutoModelForCausalLM
import torch

model_name = "iflytek/Spark-Chemistry-X1-13B"

# Load FP32 weights
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32, # explicitly FP32
    device_map="auto",
    trust_remote_code=True
)

# Convert to BF16
model = model.to(torch.bfloat16)

#  Save BF16 weights for later fast loading
save_path = "./Spark-Chemistry-X1-13B-bf16"
model.save_pretrained(save_path, safe_serialization=True)
