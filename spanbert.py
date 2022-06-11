from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("bertin-project/bertin-roberta-base-spanish")

model = AutoModelForMaskedLM.from_pretrained("bertin-project/bertin-roberta-base-spanish")