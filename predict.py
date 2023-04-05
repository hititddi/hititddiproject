import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import sys
import string

def preprocess_text(text):
    text = text.lower()  # Harfleri küçük hale getirin
    text = text.translate(str.maketrans("", "", string.punctuation))  # Noktalama işaretlerini kaldırın
    return text

def predict(df):
    # TODO:

    #************************

    print("Tahmin süreci başladi!!!")
    
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

    for index, row in df.iterrows():
        
        text = preprocess_text(row["text"])

        # İkili sınıflandırma modeli kullanılarak is_offensive tahmini yapılıyor
        binary_inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        binary_inputs.to("cuda")
        with torch.no_grad():
            binary_outputs = binary_model(**binary_inputs)
        binary_probs = torch.softmax(binary_outputs.logits, dim=-1)
        is_offensive = int(binary_probs[0, 1] > 0.5)

        # Eğer is_offensive 1 ise, multi sınıflandırma modeli kullanılarak hedef etiketi tahmin ediliyor
        if is_offensive:
            multi_inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            multi_inputs.to("cuda")
            with torch.no_grad():
                multi_outputs = multi_model(**multi_inputs)
            multi_probs = torch.softmax(multi_outputs.logits, dim=-1)
            target = class_mapping[torch.argmax(multi_probs).item()]
            df.loc[index,"offansive"]="1"
            df.loc[index,"target"]=target
        else:
            target = "OTHER"
            df.loc[index,"offansive"]="0"
            df.loc[index,"target"]=target

    # Tahminler yapılan DataFrame'i CSV dosyasına kaydediliyor
    #************************
    
    return df

if __name__ == "__main__":

    #girdi csv dosyası ismi
    input_file = sys.argv[1]

    #kayit csv dosyasi ismi
    output_file = sys.argv[2]

    #binary model yolu
    binary_model = str(sys.argv[3])

    #multi mode yolu
    multi_model = str(sys.argv[4])

    print(binary_model)
    print(multi_model)

    # For windows users, replace path seperator
    file_name = input_file.replace("\\", "/")

    df = pd.read_csv(input_file, sep="|")

    predict(df, binary_model, multi_model)
    df.to_csv(output_file, index=False, sep="|")