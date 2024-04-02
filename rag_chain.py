# get Nvidia API key
import getpass
import os 
# Google Search
import requests
from fastapi import HTTPException
from loguru import logger
from langchain.tools import Tool
# Debug
from langchain.callbacks.base import BaseCallbackHandler
# Search Query Chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
# Answer Chain
from langchain_core.output_parsers import StrOutputParser
# Related Question Chain
from langchain.output_parsers import NumberedListOutputParser
import json

# ===========================================================
# set service keys
NVAPI_KEY = 'nvapi-xxx'
SEARCH_API_KEY_GOOGLE = "xxx"                                           # [Caution] Private Key here !
SEARCH_ID_GOOGLE = "xxx" # cx parameter                                                    


# ===========================================================
# get Nvidia API key

if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    assert NVAPI_KEY.startswith("nvapi-"), f"{NVAPI_KEY[:5]}... is not a valid key"
    os.environ["NVIDIA_API_KEY"] = NVAPI_KEY

# Core LC Chat Interface
from langchain_nvidia_ai_endpoints import ChatNVIDIA

llm = ChatNVIDIA(model="mixtral_8x7b")


# ===========================================================
# Google Search Tool
# -----------------------------------------------------------
# Constant values for the RAG model.

# Search engine related. You don't really need to change this.
GOOGLE_SEARCH_ENDPOINT = "https://customsearch.googleapis.com/customsearch/v1"


# Specify the number of references from the search engine you want to use.
# 8 is usually a good number.
REFERENCE_COUNT = 8

# Specify the default timeout for the search engine. If the search engine
# does not respond within this time, we will return an error.
DEFAULT_SEARCH_ENGINE_TIMEOUT = 5
# -----------------------------------------------------------
def search_with_google(query: str, subscription_key: str, cx: str):
    """
    Search with google and return the contexts.
    """
    params = {
        "key": subscription_key,
        "cx": cx,
        "q": query,
        "num": REFERENCE_COUNT,
    }
    response = requests.get(
        GOOGLE_SEARCH_ENDPOINT, params=params, timeout=DEFAULT_SEARCH_ENGINE_TIMEOUT
    )
    if not response.ok:
        logger.error(f"{response.status_code} {response.text}")
        raise HTTPException(response.status_code, "Search engine error.")
    json_content = response.json()
    try:
        contexts = json_content["items"][:REFERENCE_COUNT]
        # Renaming 'link' to 'url' and 'displayLink' to 'displayUrl' for each context
        for context in contexts:
            if 'link' in context:
                context['url'] = context.pop('link')
            if 'displayLink' in context:
                context['displayUrl'] = context.pop('displayLink')
            if 'title' in context:
                context['name'] = context.pop('title')
            if 'pagemap' in context:
                if 'metatags' in context['pagemap']:
                    # Extracting the og:image URL, width, and height
                    og_image_url = context['pagemap']['metatags'][0].get('og:image', '')
                    og_image_width = context['pagemap']['metatags'][0].get('og:image:width', '')
                    og_image_height = context['pagemap']['metatags'][0].get('og:image:height', '')

                    # Modifying the data structure to add primaryImageOfPage at the top level
                    context['primaryImageOfPage'] = {
                        'thumbnailUrl': og_image_url,
                        'width': og_image_width,
                        'height': og_image_height,
                        # Assuming imageId is not available in the original data, and og:image URL is used as a placeholder
                        'imageId': og_image_url.split('/')[-1] # Extracting the filename as an imageId
                    }
    except KeyError:
        logger.error(f"Error encountered: {json_content}")
        return []
    return contexts

def prepare_search_results(contexts, get_raw = False):
    """
    Prepare search results for the agent and frontend display.
    
    Args:
        contexts (list): List of search result contexts from the Bing API.
        
    Returns:
        tuple: A tuple containing two items:
            - agent_context (str): Search result contexts formatted for the agent.
            - frontend_contexts (list): Original search result contexts for frontend display.
    """
    # agent_context = "\n\n".join([f"[[citation: {c['title']}]] {c['snippet']}" for c in contexts])
    agent_context = "\n\n".join([f"[[citation:{i+1}]] {c['snippet']}" for i, c in enumerate(contexts)])
    frontend_contexts = contexts
    if get_raw:
        return agent_context, frontend_contexts
    return agent_context

search_tool = Tool(
    name="Google Search",
    func=lambda query: prepare_search_results(search_with_google(query, SEARCH_API_KEY_GOOGLE, SEARCH_ID_GOOGLE)),
    description="A search tool that uses the Google search engine to find relevant information on the web.",
    # coroutine=SerpAPIWrapper()
)

tools = [search_tool]


# ===========================================================
# Debug

class AgentVerbose(BaseCallbackHandler):
    async def on_llm_start(self, serialized, prompts, **kwargs):
        """Run when LLM starts running."""
        print(f"> LLM")
        # print(f"Prompt: {prompts}")
        for prompt in prompts:
            print(prompt)

    async def on_llm_end(self, response, **kwargs):
        """Run when LLM ends running."""
        print(f"< LLM")
        # print(f"response: {response}")
        if response.generations:
            for generation in response.generations:
                for chunk in generation:
                    if chunk.text:
                        print(chunk.text)

    async def on_chain_error(self, error, **kwargs):
        """Run when chain errors."""
        print(f"> Chain **Error**")
        print(f"error: {str(error)}")

    async def on_agent_action(self, action, **kwargs):
        """Run on agent action."""
        print(f"> Agent")
        print(f"action: {action}")

    async def on_agent_finish(self, finish, **kwargs):
        """Run on agent end."""
        print(f"< Agent")
        print(f"finish: {finish}")

    async def on_tool_start(self, serialized, input_str, **kwargs):
        """Run when tool starts running."""
        print(f"> Tool")
        print(f"input: {input_str}")

    async def on_tool_end(self, output, **kwargs):
        """Run when tool ends running."""
        print(f"< Tool")
        print(f"output: {output}")

    async def on_retriever_start(self, serialized, query, **kwargs):
        """Run on retriever start."""
        print(f"> Retriever")
        print(f"query: {query}")

    async def on_retriever_end(self, documents,  **kwargs):
        """Run on retriever end."""
        print(f"< Retriever")
        print(f"documents: {documents}")


# ===========================================================
# Search with LLM
def search_with_llm(search_query, generate_related_questions=True):
    """Search with llm"""
    # -------------------------------------------------------
    # Search Query Chain
    # Define your desired data structure.
    class SearchQuery(BaseModel):
        tool: str = Field(description="selected search tool name")
        query: str = Field(description="generate a search query")
    query_parser = JsonOutputParser(pydantic_object=SearchQuery)

    # Define prompt template
    search_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the following user instructions, select a search tool and generate a search query to look up in order to get information relevant to the conversation. Use the same language as `Human`. You have access to the following tools: \n{tools}\n\n{format_instructions}"),
        ("user", "{input}")
    ])
    # Partially fill the prompt with search tool info
    search_tools = "Google Search: A search tool that uses the Google search engine to find relevant information on the web."
    search_prompt = search_prompt.partial(
        tools=search_tools
    )
    # Partially fill the prompt with format instructions.
    search_prompt = search_prompt.partial(format_instructions=query_parser.get_format_instructions())
    # print(prompt.messages)

    query_gen_chain = search_prompt | llm | query_parser
    result_query = query_gen_chain.invoke({"input": search_query}, config={'callbacks': [AgentVerbose()]})

    # -------------------------------------------------------
    # Search the web
    agent_context, frontend_contexts = prepare_search_results(search_with_google(result_query['query'], SEARCH_API_KEY_GOOGLE, SEARCH_ID_GOOGLE), get_raw=True)
    yield "contexts", frontend_contexts
    # -------------------------------------------------------
    # Answer Chain

    # Define prompt template
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a large language AI assistant built by Neutrino AI™. You are given a user question, and please write clean, concise and accurate answer to the question. You will be given a set of related contexts to the question, each starting with a reference number like [[citation:x]], where x is a number. Please use the context and cite the context at the end of each sentence if applicable.

Your answer must be correct, accurate and written by an expert using an unbiased and professional tone. Do not give any information that is not related to the question, and do not repeat. Say "information is missing on" followed by the related topic, if the given context do not provide sufficient information.

Please cite the contexts with the reference numbers, in the format [citation:x]. If a sentence comes from multiple contexts, please list all applicable citations, like [citation:3][citation:5]. Other than code and specific names and citations, your answer must be written in the same language as the question.

Here are the set of contexts:
```
{context}
```
    """),
        ("user", "{input}")
    ])

    # Partially fill the prompt with context
    answer_prompt = answer_prompt.partial(
        context=agent_context
    )

    str_parser = StrOutputParser()

    answer_chain = answer_prompt | llm | str_parser
    result = answer_chain.invoke({"input": search_query}, config={'callbacks': [AgentVerbose()]})
    yield "llm_response", result
    # -------------------------------------------------------
    # Generate Related Questions
    if generate_related_questions:
        # Define prompt template
        related_q_prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are a helpful assistant that helps the user to ask related questions, based on user's original question and the related contexts. Please identify worthwhile topics that can be follow-ups, and write questions no longer than 20 words each. Please make sure that specifics, like events, names, locations, are included in follow up questions so they can be asked standalone. For example, if the original question asks about "the Manhattan project", in the follow up question, do not just say "the project", but use the full name "the Manhattan project". Your related questions must be in the same language as `Human`.

Here are the contexts of the question:
```
{context}
```
Remember, based on the original question and related contexts, suggest three such further questions. Do NOT repeat the original question. Each related question should be no longer than 20 words. Do NOT include comma in each question. \n\n{format_instructions}
        """),
            ("user", "{input}")
        ])

        # Partially fill the prompt with context
        related_q_prompt = related_q_prompt.partial(
            context=agent_context
        )

        # "hi, bye" → ['hi', 'bye']
        num_parser = NumberedListOutputParser()
        # Partially fill the prompt with format instructions.
        related_q_prompt = related_q_prompt.partial(format_instructions=num_parser.get_format_instructions())

        associate_chain = related_q_prompt | llm | num_parser
        
        try:
            questions_list = associate_chain.invoke({"input": search_query}, config={'callbacks': [AgentVerbose()]})
            # questions_list = associate_chain.invoke({"input": search_query})
            # 将每个问题转换为一个字典，键为"question"
            related_questions = [{"question": question} for question in questions_list]
            # 转换为JSON格式的字符串
            # related_questions = json.dumps(questions_json)
        except Exception as e:
            related_questions = '[]'
        yield "related_questions", related_questions
        # return frontend_contexts, result, related_questions

    # return frontend_contexts, result

if __name__ == "__main__":
    def print_test_sub(test_sub):
        """Print test infomations."""
        print("\n===========================================================")
        print(f"> Test {test_sub}...")
        print("-----------------------------------------------------------")

    # -------------------------------------------------
    # Test llm
    print_test_sub("NVIDIA AI Foundation Endpoints Connection")
    llm_query = "tell me a joke."
    print("> Query: ", llm_query)
    for chunks in llm.stream(llm_query):
        print(chunks.content, end="")

    # -------------------------------------------------
    # Test Search
    print_test_sub("Google Search Tool")
    search_query = "what year was breath of the wild released?"
    contexts = prepare_search_results(search_with_google(search_query, SEARCH_API_KEY_GOOGLE, SEARCH_ID_GOOGLE))
    # contexts = "\n\n".join([f"[[citation: {c['title']}]] {c['snippet']}" for i, c in enumerate(results)])
    print("> Query: ", search_query)
    print()
    print(contexts)

    # --------------------------------------------------
    # Test Answer Chain
    print_test_sub("Answer Chain")
    search_query = "when was breath of the wild first released?"
    frontend_contexts, result, related_questions = search_with_llm(search_query)
    print("> frontend_contexts:\n", frontend_contexts)
    print("> frontend_contexts:\n", result)
    print("> frontend_contexts:\n", related_questions)

