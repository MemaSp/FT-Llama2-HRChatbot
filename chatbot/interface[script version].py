# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

# Replace 'your_model_directory' with the path to your model's directory
model_name_or_path = r"C:\Users\spite\Documents\FT-Llama2-HR_Chatbot\results\final_merged_checkpoint" 
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)


# %%
def ask_model(question, passage, max_length=200):
    inputs = tokenizer.encode(question, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_start = response.find("### Response:") + len("### Response:")
    response_end = response.find("### End")
    
    # Clean up any leading or trailing whitespace or newline characters in the response
    final_response = response[response_start:response_end].strip()
    return final_response

# %%
def ask_model(question, passage, max_length=200):
    """
    Generate a model response based on a question and a passage provided.
    Format the output to extract the part between "### Response:" and "### End".
    
    :param question: The question to ask the model.
    :param passage: The passage context for the model.
    :param max_length: Maximum length of the generated response.
    """

    RESPONSE_KEY = "### Response:"
    END_KEY = "### End"
    
    # Encode the question and passage to the model's required input format
    inputs = tokenizer.encode(question, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    
    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the response part between the designated markers
    response_start = response.find(RESPONSE_KEY) + len(RESPONSE_KEY)
    response_end = response.find(END_KEY)
    
    # Clean up any leading or trailing whitespace or newline characters in the response
    final_response = response[response_start:response_end].strip()
    
    return final_response


# %% [markdown]
# "context": "User Jayden has been involved in the following projects: Vertex where they worked week1: 18hrs, week2: 15hrs, week3: 19hrs.",
#         "instruction": "Can you provide the total hours Jayden worked on Vertex?",
#         "response": "Jayden worked on Vertex for a total of 52 hours over 3 weeks.",
#         "category": "summarization"

# %% [markdown]
# "context": "User Abigail has been involved in the following projects: Quest where they worked week1: 12hrs, week2: 13hrs.",
#         "instruction": "Can you provide the total hours Abigail worked on Quest?",
#         "response": "Abigail worked on Quest for a total of 25 hours over 2 weeks.",
#         "category": "summarization"

# %%
# Define the passage and question
passage = "Paris is the capital city of France"
question = "what is the capital city of France, please?"

# Call the updated ask_model function with both the question and passage
response = ask_model(question,passage)
print(response)


# %%
def main():
    print("Type 'exit' to quit.")
    while True:
        question = input("Ask a question: ")
        if question.lower() == 'exit':
            break
        response = ask_model(question)
        print("Answer:", response)

if __name__ == "__main__":
    main()



