import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import sys

def predict(input_csv, output_csv):
    # Model ve tokenizer yükleniyor
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")

    binary_model = AutoModelForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-cased", num_labels=2)
    binary_model.load_state_dict(torch.load("binary_model.pt"))
    binary_model.to("cuda")
    binary_model.eval()

    multi_model = AutoModelForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-cased", num_labels=4)
    multi_model.load_state_dict(torch.load("multi_model.pt"))
    multi_model.to("cuda")
    multi_model.eval()

    class_mapping = {0: "RACIST", 1: "SEXIST", 2: "PROFANITY", 3: "INSULT"}

    # CSV dosyası okunuyor ve tahminler yapılıyor
    input_df = pd.read_csv(input_csv, sep="|")
    output_df = input_df.copy()

    for index, row in input_df.iterrows():
        text = row["text"]

        # İkili sınıflandırma modeli kullanılarak is_offensive tahmini yapılıyor
        binary_inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        binary_inputs.to("cuda")
        with torch.no_grad():
            binary_outputs = binary_model(**binary_inputs)
        binary_probs = torch.softmax(binary_outputs.logits, dim=-1)
        is_offensive = int(binary_probs[0, 1] > 0.5)
        output_df.at[index, "offensive"] = is_offensive

        # Eğer is_offensive 1 ise, multi sınıflandırma modeli kullanılarak hedef etiketi tahmin ediliyor
        if is_offensive:
            multi_inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            multi_inputs.to("cuda")
            with torch.no_grad():
                multi_outputs = multi_model(**multi_inputs)
            multi_probs = torch.softmax(multi_outputs.logits, dim=-1)
            target = class_mapping[torch.argmax(multi_probs).item()]
        else:
            target = "OTHER"

        output_df.at[index, "target"] = target

    # Tahminler yapılan DataFrame'i CSV dosyasına kaydediliyor
    output_df.to_csv(output_csv, sep="|", index=False)

if __name__ == "__main__":
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    #predict(input_csv, output_csv)