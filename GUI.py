import tkinter as tk
from tkinter import scrolledtext
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_path = "saved_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)


def translate_text():
    input_text = input_box.get("1.0", tk.END).strip()
    if not input_text:
        output_label.config(text="Please enter some text.")
        return
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_label.config(text=translated_text)

window = tk.Tk()
window.title("Machine Translation App")
window.geometry("600x400")
window.configure(bg="white")

title = tk.Label(window, text="Machine Translation App", font=("Helvetica", 16, "bold"), bg="white", fg="blue")
title.pack(pady=10)

subtitle = tk.Label(window, text="Enter text to translate:", font=("Helvetica", 12), bg="white")
subtitle.pack()

input_box = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=60, height=5, font=("Helvetica", 11))
input_box.pack(pady=10)

translate_button = tk.Button(window, text="Translate", command=translate_text, font=("Helvetica", 12), bg="#4a90e2", fg="white")
translate_button.pack(pady=5)

output_title = tk.Label(window, text="Translation:", font=("Helvetica", 12, "bold"), bg="white")
output_title.pack(pady=(15, 0))

output_label = tk.Label(window, text="", font=("Helvetica", 12), wraplength=550, justify="left", bg="white")
output_label.pack(pady=5)

window.mainloop()
