import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Define model names
finetuned_model_name = "rkumar1999/Llama-3.1-8B-Instruct-Open-R1-Distill"
original_model_name = "meta-llama/Llama-3.1-8B-Instruct"

# Setup quantization configuration for 4-bit quantization
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # Use FP16 for computation
    bnb_4bit_use_double_quant=True,        # Enable double quantization for accuracy
    bnb_4bit_quant_type="nf4"              # 'nf4' or 'fp4'
)

# Load finetuned model and its tokenizer
finetuned_tokenizer = AutoTokenizer.from_pretrained(finetuned_model_name)
finetuned_model = AutoModelForCausalLM.from_pretrained(
    finetuned_model_name,
    device_map="auto",
    quantization_config=quant_config,
    low_cpu_mem_usage=True
)

# Load original model and its tokenizer
original_tokenizer = AutoTokenizer.from_pretrained(original_model_name)
original_model = AutoModelForCausalLM.from_pretrained(
    original_model_name,
    device_map="auto",
    quantization_config=quant_config,
    low_cpu_mem_usage=True
)

# Choose device: GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
finetuned_model.to(device)
original_model.to(device)

# Define a list of 10 challenging math and logic problems (prompts)
prompts = [
    "Given a geometric sequence of positive terms {a_n} with the sum of the first n terms denoted by S_n, and the following equation: 2^10 S_30 + S_10 = (2^10 + 1) S_20. Determine the common ratio of the sequence {a_n}: (A) 1/8 (B) 1/4 (C) 1/2 (D) 1.",
    "What is the value of 21*75?",
    "Solve for x: 3x^2 - 12x + 9 = 0.",
    "What is the derivative of f(x) = x^3 - 5x^2 + 2x - 7?",
    "Evaluate the integral: ∫ (2x + 3) dx.",
    "If the sum of two numbers is 15 and their product is 56, what are the two numbers?",
    "Solve the logic puzzle: If all bloops are razzies and all razzies are lazzies, are all bloops definitely lazzies?",
    "What is the next number in the Fibonacci sequence: 1, 1, 2, 3, 5, 8, ...?",
    "Solve the equation: log_2(x) + log_2(x - 3) = 3.",
    "In a standard deck of 52 cards, what is the probability of drawing 2 aces in a row without replacement?"
]

# Open a file to write the evaluation results
with open("evaluation_results.txt", "w") as outfile:
    for prompt in prompts:
        outfile.write(f"Prompt: {prompt}\n")
        
        # Generate output from the finetuned model
        inputs_ft = finetuned_tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs_ft = finetuned_model.generate(
                **inputs_ft,
                max_length=512,
                do_sample=True,      # Optional: enables sampling for diverse outputs
                top_p=0.95,          # Optional: nucleus sampling parameter
                top_k=50             # Optional: top-k filtering parameter
            )
        generated_ft = finetuned_tokenizer.decode(outputs_ft[0], skip_special_tokens=True)
        outfile.write("Finetuned Model Output: " + generated_ft + "\n")
        
        # Generate output from the original model
        inputs_orig = original_tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs_orig = original_model.generate(
                **inputs_orig,
                max_length=512,
                do_sample=True,
                top_p=0.95,
                top_k=50
            )
        generated_orig = original_tokenizer.decode(outputs_orig[0], skip_special_tokens=True)
        outfile.write("Original Model Output: " + generated_orig + "\n")
        
        outfile.write("-" * 80 + "\n")

print("Evaluation complete. Results saved to evaluation_results.txt")
