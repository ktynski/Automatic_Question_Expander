import openai
import pandas as pd
import concurrent.futures
import streamlit as st


def generate_text(prompt, max_tokens, n, temperature):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an all-knowing AI expert at inferring content, topics, subtopics, subsubtopics, etc. as well as likely questions, with a focus on high intent. Please think step by step to complete the following, and provide only the answer:"},
            {"role": "user", "content": f"{prompt}"}],
        max_tokens=max_tokens,
        n=n,
        stop=None,
        temperature=temperature,
        stream=True,  # Enable streaming
    )
    for event in response:
        if event['type'] == 'message':
            message_content = event['choices'][0]['message']['content'].strip()
            print(message_content)  # Print the generated text
    return message_content

def generate_subtopics(topic, max_tokens, n, temperature):
    prompt = f"Generate 40 common subtopics related to {topic}"
    subtopics = generate_text(prompt, max_tokens, n, temperature).strip().split("\n")
    return subtopics

def generate_subsubtopics(subtopic, max_tokens, n, temperature):
    prompt = f"Generate 20 common subsubtopics related to {subtopic}"
    subsubtopics = generate_text(prompt, max_tokens, n, temperature).strip().split("\n")
    return subsubtopics

def generate_questions(topic, subtopic, subsubtopic, max_tokens, n, temperature):
    prompt = f"You are an all knowing AI that specializes in search behavior, SEO, and search intent. You are especially good at predicting questions related to topics. Please Generate 30 of the most common questions related to {subsubtopic} within the context of {subtopic} in the larger topic of {topic}"
    questions = generate_text(prompt, max_tokens, n, temperature).strip().split("\n")
    return [question.split(". ")[-1] for question in questions]

def process_subsubtopic(topic, subtopic, subsubtopic, max_tokens, n, temperature):
    questions = generate_questions(topic, subtopic, subsubtopic, max_tokens, n, temperature)
    data = {
        "topic": topic,
        "subtopic": subtopic,
        "subsubtopic": subsubtopic,
    }
    for i, question in enumerate(questions):
        data[f"Question {i + 1}"] = question
    return data

# Streamlit app starts here
def app():
    st.title("Topic, Subtopic, and Question Generator")
    st.markdown("Generate subtopics, subsubtopics, and questions using OpenAI's GPT-3 model.")

    api_key = st.text_input("Enter your OpenAI API key:")
    if api_key:
        openai.api_key = api_key
    else:
        st.warning("Please enter a valid OpenAI API key.")
        return

    topic = st.text_input("Enter the main topic:", "Home Security")
    max_tokens = st.slider("Max tokens:", 10, 2000, 1500, step=10)
    n = st.slider("Number of responses (n):", 1, 5, 1)
    temperature = st.slider("Temperature:", 0.1, 1.0, 0.6, step=0.1)

    if st.button("Generate"):
        st.write("Generating data...")
        subtopics = generate_subtopics(topic, max_tokens, n, temperature)
        data = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for subtopic in subtopics:
                subsubtopics = generate_subsubtopics(subtopic, max_tokens, n, temperature)
                results = [executor.submit(process_subsubtopic, topic, subtopic, subsubtopic, max_tokens, n, temperature) for subsubtopic in subsubtopics]
                for f in concurrent.futures.as_completed(results):
                    data.append(f.result())
        df = pd.DataFrame(data)
        st.write(df)

if __name__ == "__main__":
    app()

       
