from transformers import AutoModelForCausalLM, AutoTokenizer

# Replace 'your_model_directory' with the path to your model's directory
model_path = r"C:\Users\spite\Documents\FT-Llama2-HR_Chatbot\results\final_merged_checkpoint" 
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)


def ask_model(question, passage, max_length=200):
    # Encoding the question
    inputs = tokenizer.encode(question, return_tensors='pt')
    # Generating the response
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    # Decoding the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extracting the response section
    response_start = response.find("### Response:") + len("### Response:")
    response_end = response.find("### End")
    
    # Clean up any leading or trailing whitespace or newline characters in the response
    final_response = response[response_start:response_end].strip()
    return final_response

def main():
    print("Welcome to the Chatbot Terminal! Please provide a question and a passage for analysis.\n")
    
    # User input for the passage
    passage = input("Enter your passage:\n")
    
    # User input for the question
    question = input("\nEnter your question:\n")
    
    # Processing the question and passage
    print("\nAnalyzing your question and calculating the response...\n")
    response = ask_model(question, passage)
    
    # Displaying the response
    print("Response:\n" + response)

    print("\nThank you for using the Chatbot Terminal! If you have another question, please start again.")

# Run the main function
if __name__ == "__main__":
    main()
