import tkinter as tk
from tkinter import scrolledtext
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize model and tokenizer
model_path = r"C:\Users\spite\Documents\FT-Llama2-HR_Chatbot\results\final_merged_checkpoint" 
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

def ask_model(question, passage, max_length=200):
    # Encode the question and generate the output using the model
    inputs = tokenizer(question, return_tensors='pt')
    outputs = model.generate(inputs['input_ids'], max_length=max_length, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_start = response.find("### Response:") + len("### Response:")
    response_end = response.find("### End")

    # Extract the final response and clean it up
    final_response = response[response_start:response_end].strip()
    return final_response

def handle_query(passage, question, display_area):
    display_area.config(state=tk.NORMAL)
    display_area.insert(tk.END, "You: " + question + "\n")
    display_area.insert(tk.END, "Analyzing your question...\n")
    response = ask_model(question, passage)
    display_area.insert(tk.END, "Bot: " + response + "\n\n")
    display_area.config(state=tk.DISABLED)

def submit_action(entry_question, entry_passage, display_area):
    question = entry_question.get()
    passage = entry_passage.get("1.0", tk.END).strip()
    threading.Thread(target=handle_query, args=(passage, question, display_area)).start()

def main():
    window = tk.Tk()
    window.title("Chatbot UI")

    # Text area for passage
    label_passage = tk.Label(window, text="Enter your passage:")
    label_passage.pack(pady=(10, 0))
    entry_passage = scrolledtext.ScrolledText(window, height=8, width=50)
    entry_passage.pack()

    # Text area for question
    label_question = tk.Label(window, text="Enter your question:")
    label_question.pack(pady=(10, 0))
    entry_question = tk.Entry(window, width=50)
    entry_question.pack()

    # Display area
    display_area = scrolledtext.ScrolledText(window, height=15, width=60, state=tk.DISABLED)
    display_area.pack(pady=(10, 0))

    # Submit button
    submit_button = tk.Button(window, text="Ask", command=lambda: submit_action(entry_question, entry_passage, display_area))
    submit_button.pack(pady=(5, 10))

    window.mainloop()

if __name__ == "__main__":
    main()
