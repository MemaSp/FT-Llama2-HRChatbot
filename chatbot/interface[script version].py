# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

# Replace 'your_model_directory' with the path to your model's directory
model_path = r"C:\Users\spite\Documents\FT-Llama2-HR_Chatbot\results\final_merged_checkpoint" 
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)


# %% [markdown]
# untrained model

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

# Replace 'your_model_directory' with the path to your model's directory
model_path = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

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
# Test 1
# passage = “Over the past three weeks, several users have been involved in multiple projects.User Noah has been involved in the following projects: Nature project where they worked week 1: 11 hours, week 2: 21 hours, and week 3: 20 hours. For the Harmony project, Noah worked 21 hours in week 1 and 12 hours in week 2. Noah did not work on the Harmony project in week 3. User Leah has contributed to the following projects: Fusion where she worked week 1: 15 hours, week 2: 18 hours, and week 3: 22 hours. In the Pulse project, Leah worked 10 hours in week 1 and 16 hours in week 2, but did not contribute any hours in week 3
# 
# 
# Question =  Can you calculate the total hours Noah worked on Nature project?

# %% [markdown]
# Test 2
# passage = “Over the past three weeks, several users have been involved in multiple projects.User Noah has been involved in the following projects: Nature project where they worked week 1: 11 hours, week 2: 21 hours, and week 3: 20 hours. For the Harmony project, Noah worked 21 hours in week 1 and 12 hours in week 2. Noah did not work on the Harmony project in week 3. User Leah has contributed to the following projects: Fusion where she worked week 1: 15 hours, week 2: 18 hours, and week 3: 22 hours. In the Pulse project, Leah worked 10 hours in week 1 and 16 hours in week 2, but did not contribute any hours in week 3
# 
# 
# Question =  List the projects Leah contributed too please?

# %% [markdown]
# Test 3
# passage = "User Fiona has been involved in the following projects: Zenith where they worked week1: 12hrs, week2: 14hrs. Apex where they worked week1: 16hrs, week2: 22hrs.",
# 
# 
# Question = Can you list all the projects Fiona was involved with please?
# 

# %% [markdown]
# Test 4
# passage = "User Leah has contributed to the following projects: Fusion where she worked week 1: 15 hours, week 2: 18 hours, and week 3: 22 hours. In the Pulse project, Leah worked 10 hours in week 1 and 16 hours in week 2, but did not contribute any hours in week 3 ",
# 
# 
# Question =  "Calculate the total hours Leah contributed to the Fusion project,please?"

# %% [markdown]
# Test 5
# passage = "User Abigail has been involved in the following projects: Quest where they worked week1: 12hrs, week2: 13hrs.User Jayden has been involved in the following projects: Vertex where they worked week1: 18hrs, week2: 15hrs, week3: 19hrs.",
# 
# question =  "in which projcts did Jayden contributed to please?"

# %%
# Define the passage and question
passage = "Over the past three weeks, several users have been involved in multiple projects.User Noah has been involved in the following projects: Nature project where they worked week 1: 11 hours, week 2: 21 hours, and week 3: 20 hours. For the Harmony project, Noah worked 21 hours in week 1 and 12 hours in week 2. Noah did not work on the Harmony project in week 3. User Leah has contributed to the following projects: Fusion where she worked week 1: 15 hours, week 2: 18 hours, and week 3: 22 hours. In the Pulse project, Leah worked 10 hours in week 1 and 16 hours in week 2, but did not contribute any hours in week 3",

question =  "Can you calculate the total hours Noah worked on Nature project?"

# Call the updated ask_model function with both the question and passage
response = ask_model(question,passage)
print(response)


# %%
# Define the passage and question
passage = "Over the past three weeks, several users have been involved in multiple projects.User Noah has been involved in the following projects: Nature project where they worked week 1: 11 hours, week 2: 21 hours, and week 3: 20 hours. For the Harmony project, Noah worked 21 hours in week 1 and 12 hours in week 2. Noah did not work on the Harmony project in week 3. User Leah has contributed to the following projects: Fusion where she worked week 1: 15 hours, week 2: 18 hours, and week 3: 22 hours. In the Pulse project, Leah worked 10 hours in week 1 and 16 hours in week 2, but did not contribute any hours in week 3",

question =  "List the projects Leah contributed too please?"

# Call the updated ask_model function with both the question and passage
response = ask_model(question,passage)
print(response)

# %%
# Define the passage and question
passage =  "User Fiona has been involved in the following projects: Zenith where they worked week1: 12hrs, week2: 14hrs. Apex where they worked week1: 16hrs, week2: 22hrs.",

question =  "Can you list all the projects Fiona was involved with please?"

# Call the updated ask_model function with both the question and passage
response = ask_model(question,passage)
print(response)

# %%
# Define the passage and question
passage =  "User Leah has contributed to the following projects: Fusion where she worked week 1: 15 hours, week 2: 18 hours, and week 3: 22 hours. In the Pulse project, Leah worked 10 hours in week 1 and 16 hours in week 2, but did not contribute any hours in week 3 ",


question =  "Calculate the total hours Leah contributed to the Fusion project,please?"

# Call the updated ask_model function with both the question and passage
response = ask_model(question,passage)
print(response)

# %%
# Define the passage and question
passage = "User Abigail has been involved in the following projects: Quest where they worked week1: 12hrs, week2: 13hrs.User Jayden has been involved in the following projects: Vertex where they worked week1: 18hrs, week2: 15hrs, week3: 19hrs.",

question =  "in which projcts did Jayden contributed to please?"

# Call the updated ask_model function with both the question and passage
response = ask_model(question,passage)
print(response)

# %% [markdown]
# import torch
# torch.cuda.empty_cache()

# %%
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


