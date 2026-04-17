from google import genai
from pipeline.LLM_prompt import RAG_TEMPLATE, TOKEN_TEMPLATE

def check_tokens(chunks):
    client = genai.Client() 
    text_only_list = [chunk['text'] for chunk in chunks]

    response = client.models.count_tokens(
        model="gemini-2.5-flash-lite", 
        contents=text_only_list # Temizlenmiş metin listesini gönderiyoruz
    )
    
    return response.total_tokens

def get_answer(question, context):
    client = genai.Client()
    
    prompt = RAG_TEMPLATE.format(context=context, question=question)

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite", # Mevcut en güncel stabil model
        contents=prompt
    )

    return response.text