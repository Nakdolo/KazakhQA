import os
from flask import Flask, request, render_template, redirect, url_for, session, flash
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'key123'

model_path = "C:/Users/Lenovo/Desktop/Main/Lessons/NLP/assig/project/kazakh_qa_model(5e)"
tokenizer = AutoTokenizer.from_pretrained(model_path)
qa_model = AutoModelForQuestionAnswering.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
qa_model.to(device)

def answer_question(context, question):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = qa_model(**inputs)

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    if end_index < start_index:
        end_index = start_index

    all_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    answer_tokens = all_tokens[start_index:end_index+1]
    answer_tokens = [token for token in answer_tokens if token not in tokenizer.all_special_tokens]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    
    if answer in context:
        highlighted_context = context.replace(answer, f"<span class='highlight'>{answer}</span>")
    else:
        highlighted_context = context

    return answer, highlighted_context

def save_history(context, question, answer):
    with open("history.txt", "a", encoding="utf-8") as file:
        file.write(f"{datetime.now()}\n")
        file.write(f"Context: {context}\n")
        file.write(f"Question: {question}\n")
        file.write(f"Answer: {answer}\n")
        file.write("------\n")

def load_history():
    if os.path.exists("history.txt"):
        with open("history.txt", "r", encoding="utf-8") as file:
            history = file.read().strip().split("------\n")
            history = [entry.strip() for entry in history if entry.strip()]
            return history
    return []

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        context = request.form["context"]
        question = request.form["question"]
        answer, highlighted_context = answer_question(context, question)
        save_history(context, question, answer)
        return render_template("index.html", context=context, question=question, answer=answer, highlighted_context=highlighted_context)
    return render_template("index.html")

@app.route("/history")
def history():
    history = load_history()
    return render_template("history.html", history=history)

if __name__ == "__main__":
    app.run(debug=True)
