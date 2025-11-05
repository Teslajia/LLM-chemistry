from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
# Load model and tokenizer
model_name = "iflytek/Spark-Chemistry-X1-13B"
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="auto",
    trust_remote_code=True
)
# Deliberative
chat_history = [
  {
    "role" : "system",
    "content" : "请你先深入剖析给出问题的关键要点与内在逻辑,生成思考过程,再根据思考过程回答给出问题。思考过程以<unused6>开头,在结尾处用<unused7>标注结束,<unused7>后为基于思考过程的回答内容"
  }
  ,
  {
    "role" : "user",
    "content" : "请回答下列问题:高分子材料是否具有柔顺性主要决定于()的运动能力。\nA、主链链节\nB、侧基\nC、侧基内的官能团或原子?"
  }]


inputs = tokenizer.apply_chat_template(
    chat_history,
    tokenize=True,
    return_tensors="pt",
    add_generation_prompt=True
).to(model.device)

outputs = model.generate(
    inputs,
    max_new_tokens=8192,
    top_k=1,
    do_sample=True,
    repetition_penalty=1.02,
    temperature=0.7,
    eos_token_id=5,
    pad_token_id=0,
)

response = tokenizer.decode(
    outputs[0][inputs.shape[1] :],
    skip_special_tokens=True
)
print(response)
