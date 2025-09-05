import os
from dotenv import load_dotenv, find_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.sequential import SequentialChain
from langchain.chains import LLMChain
import streamlit as st

load_dotenv(find_dotenv())


def generate_funny_quote(topic, character, language):
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


    template_for_transalation = """
    Translate the {funny_quote} in {language}. Make sure the language is simple
    """

    prompt_for_translation = ChatPromptTemplate.from_template(template_for_transalation)
    chain_for_translation = LLMChain(llm=model, prompt=prompt_for_translation, output_key="translated_output")

    seq_chain = SequentialChain(
        chains=[chain, chain_for_translation],
        input_variables=["topic", "character", "language"],
        output_variables=["funny_quote", "translated_output"]
    )
    output_seq_chain = seq_chain.invoke({"topic":topic, "character":character, "language":language})
    print(output_seq_chain['funny_quote'])
    print(output_seq_chain['translated_output'])
    return(output_seq_chain)

def main():
    st.set_page_config(page_title="FunnAI Quote Generator", layout="centered")
    st.title("Let MeAI put a smile on your face")
    st.header("Lets get started .....")
    
    character_input = st.text_input(label="Enter the character")
    topic_input = st.text_input(label="Enter the topic")
    language_input = st.text_input(label="Enter the language")  
    submit_button = st.button("Submit")
    if topic_input and character_input and language_input:
        if submit_button:
            with st.spinner("Generating a funny quote...."):
                quote = generate_funny_quote(topic = topic_input, 
                                            character= character_input, 
                                            language=language_input)
            st.success("Quote successfully generated...")
            st.markdown(f"**Original Quote:**\n\n*\"{quote['funny_quote']}\"*")
            st.markdown(f"**Translated Quote ({language_input}):**\n\n*\"{quote['translated_output']}\"*")
            with st.expander("English version:"):
                st.write(quote['funny_quote'])
            with st.expander(f"{language_input} Version"):
                st.write(quote['translated_output'])



# Call the main function to run the app
if __name__ == "__main__":
    main()