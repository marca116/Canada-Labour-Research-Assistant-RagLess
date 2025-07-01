import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Add the parent directory to sys.path
import time

import streamlit as st

from config import ChatbotInterfaceConfig, PromptTemplateType, OllamaRAGConfig, ConsoleConfig
from local import get_ollama_answer_local, get_ollama_answer_local_stream
from local_vllm import get_vllm_answer, get_vllm_answer_stream
from remote import get_llm_answer_remote, get_llm_answer_remote_stream

# Get the prompt template based on whether the model is remote or not
def get_prompt_template(custom_system_prompt, is_remote, language):
    if is_remote:
        prompt_template = PromptTemplateType.structured_with_context_remote
    else:
        prompt_template = PromptTemplateType.structured_with_context_local

    if language == "fr":
        prompt_template += PromptTemplateType.answer_in_french

    if custom_system_prompt:
        prompt_template += "\n\n" + custom_system_prompt

    return prompt_template

# @st.cache_data(show_spinner=False, ttl=600)
def retrieve_database(database_question,
                      language,
                      is_remote,
                      chat_model,
                      hyperparams,
                      custom_system_prompt,
                      previous_messages,
                      previous_question_chunks,
                      nb_previous_questions=1):

    if chat_model is None:
        chat_model = ChatbotInterfaceConfig.default_model_local if not is_remote else ChatbotInterfaceConfig.default_model_remote

    # Clone the hyperparams and find the max context length based on the model
    hyperparams = hyperparams.copy()

    prompt_template_type = get_prompt_template(custom_system_prompt, is_remote, language)

    # Text prompts
    prompt = prompt_template_type
    question_intro = PromptTemplateType.question_intro_en if language == "en" else PromptTemplateType.question_intro_fr
    prompt_question = "\n\n" + question_intro + ": " + database_question

    prompt_message = {
        'role': 'user',
        'content': prompt + prompt_question
    }

    messages = []
    previous_questions_and_answers = []

    if nb_previous_questions > 0 and previous_messages and previous_question_chunks and len(previous_messages) >= 2:
        previous_question_text = PromptTemplateType.previous_question_en if language == "en" else PromptTemplateType.previous_question_fr
        previous_answer_text = PromptTemplateType.previous_answer_en if language == "en" else PromptTemplateType.previous_answer_fr

        # Get nb_previous_questions Q&A pairs, in reverse order
        nb_questions = 0
        nb_answers = 0
        for message in previous_messages[::-1]:
            if nb_questions >= nb_previous_questions and nb_answers >= nb_previous_questions:
                break

            if message["role"] == "user":
                previous_question = {
                    "role": "user",
                    "content": f"{previous_question_text.capitalize()}:\n\n" + message["content"]
                }
                previous_questions_and_answers.insert(0, previous_question)
                nb_questions += 1
            else:
                previous_answer = {
                    "role": "assistant",
                    "content": f"{previous_answer_text.capitalize()}:\n\n" + message.get("content")
                }
                previous_questions_and_answers.insert(0, previous_answer)
                nb_answers += 1

    num_ctx = hyperparams.get("num_ctx")

    # Only update the num_ctx if it's not None (always the case for vLLM)
    if num_ctx is not None:
        hyperparams["num_ctx"] = num_ctx

    # List of all messages to be sent to the LLM
    messages = previous_questions_and_answers + [prompt_message]
    
    return messages, chat_model, hyperparams

def get_previous_question_chunks(previous_messages, nb_previous_questions=1):
    if nb_previous_questions <= 0 or not previous_messages:
        return []

    previous_question_chunks = []
    nb_prev_assistant_messages = 0
    # Get chunks from the last nb_previous_questions assistant messagesm starting from the last
    for message in previous_messages[::-1]:
        if nb_prev_assistant_messages >= nb_previous_questions:
            break

        if message["role"] == "user":
            continue

        chunks = message.get("chunks", [])
        # Add the chunks to the previous question chunks, at the beginning of the list
        for chunk in chunks:
            previous_question_chunks.insert(0, chunk)

        nb_prev_assistant_messages += 1

    return previous_question_chunks

def print_context_info(total_used_tokens, hyperparams):
    if ConsoleConfig.verbose:
        if total_used_tokens is not None:
            print(f"Total used tokens: {total_used_tokens}")
        num_ctx = hyperparams.get("num_ctx")
        if num_ctx is not None:
            print(f"Context window size: {num_ctx}")

@st.cache_data(show_spinner=False, ttl=600)
def retrieve_database_local(database_question,
                      language,
                      is_remote=False,
                      chat_model=None,
                      hyperparams=OllamaRAGConfig.HyperparametersAccuracyConfig,
                      custom_system_prompt=None,
                      previous_messages=None,
                      nb_previous_questions=1,
                      engine="ollama"):
    
    previous_question_chunks = get_previous_question_chunks(previous_messages, nb_previous_questions)
    
    messages, chat_model, hyperparams = retrieve_database(
        database_question, language, is_remote, chat_model, hyperparams, 
        custom_system_prompt, previous_messages, previous_question_chunks, nb_previous_questions
    )

    if is_remote:
        answer = get_llm_answer_remote(chat_model, messages, hyperparams)
    elif engine=="ollama":
        answer = get_ollama_answer_local(chat_model, messages, hyperparams)
    elif engine=="vllm":
        answer = get_vllm_answer(chat_model, messages, hyperparams)

    return answer

def retrieve_database_stream(database_question,
                      language,
                      is_remote=False,
                      chat_model=None,
                      hyperparams=OllamaRAGConfig.HyperparametersAccuracyConfig,
                      custom_system_prompt=None,
                      previous_messages=None,
                      nb_previous_questions=1,
                      engine="ollama"):
    
    messages, chat_model, hyperparams = retrieve_database(
        database_question, language, is_remote, chat_model, hyperparams, 
        custom_system_prompt, previous_messages, nb_previous_questions
    )

    if is_remote:
        stream_generator = get_llm_answer_remote_stream(chat_model, messages, hyperparams)
    elif engine=="ollama":
        stream_generator = get_ollama_answer_local_stream(chat_model, messages, hyperparams)
    elif engine=="vllm":
        stream_generator = get_vllm_answer_stream(chat_model, messages, hyperparams)

    return stream_generator
    

if __name__ == "__main__":
    from config import vLLMRAGConfig, vLLMChatbotInterfaceConfig

    start_time = time.time()
    answer, _, _, _, original_answer = retrieve_database_local(
        "How do you change your password in WEIMS?", 
        "en", 
        is_remote=False, 
        chat_model=vLLMChatbotInterfaceConfig.default_model_local,
        hyperparams=vLLMRAGConfig.HyperparametersAccuracyConfig,
        engine="vllm"
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    #print(answer)
    print(original_answer)
