import os
from dotenv import load_dotenv, find_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.sequential import SequentialChain
from langchain.chains import LLMChain
import streamlit as st

load_dotenv(find_dotenv())


def generate_joke_and_caricature(topic, character):
    model = ChatAnthropic(model="claude-sonnet-4-20250514",
                      temperature= 0.5,
                      max_tokens=2000,
                      max_retries=2)

    template = """
    As a comedian, please come up with a simple funny quote. Max words = 90
    Please use this topic {topic} and this character {character} to generate the quote


    Quote
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = LLMChain(llm=model, prompt=prompt, output_key="funny_quote")


    template_for_caricature = """
    Based on this funny quote: "{funny_quote}"
    
    Create a detailed caricature description that visually represents this joke. Include:
    1. The character's exaggerated features and expression
    2. Visual elements that illustrate the humor
    3. Props, setting, or background details
    4. Specific artistic style suggestions
    
    Make it vivid and detailed enough for an artist to draw.
    
    Caricature Description:
    """

    prompt_for_caricature = ChatPromptTemplate.from_template(template_for_caricature)
    chain_for_caricature = LLMChain(llm=model, prompt=prompt_for_caricature, output_key="caricature_description")

    seq_chain = SequentialChain(
        chains=[chain, chain_for_caricature],
        input_variables=["topic", "character"],
        output_variables=["funny_quote", "caricature_description"]
    )
    output_seq_chain = seq_chain.invoke({"topic":topic, "character":character})
    print(output_seq_chain['funny_quote'])
    print(output_seq_chain['caricature_description'])
    return(output_seq_chain)

def main():
    st.set_page_config(page_title="Joke & Caricature Generator", layout="centered")
    st.title("🎭 Joke & Caricature Generator 🎨")
    st.header("Generate jokes with visual caricature descriptions!")
    
    character_input = st.text_input(label="Enter the character")
    topic_input = st.text_input(label="Enter the topic")
    submit_button = st.button("Generate Joke & Caricature")
    if topic_input and character_input:
        if submit_button:
            with st.spinner("Generating joke and caricature description...."):
                result = generate_joke_and_caricature(topic = topic_input, 
                                                     character= character_input)
            st.success("Joke and caricature successfully generated!")
            
            st.markdown("### 💬 The Joke:")
            st.markdown(f"*\"{result['funny_quote']}\"*")
            
            st.markdown("### 🎨 Caricature Description:")
            st.markdown(result['caricature_description'])
            
            with st.expander("📝 View Details"):
                st.write("**Joke:**", result['funny_quote'])
                st.write("**Caricature:**", result['caricature_description'])



# Call the main function to run the app
if __name__ == "__main__":
    main()