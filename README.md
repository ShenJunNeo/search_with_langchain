<div align="center">
<h1 align="center">Search with Langchain</h1>
Build your own conversational search engine using less than 500 lines of code.
<br/>
<a href="http://twctoolbox.com/" target="_blank"> Live Demo </a>
<br/>
<img width="70%" src="https://github.com/leptonai/search_with_lepton/assets/1506722/845d7057-02cd-404e-bbc7-60f4bae89680">
</div>


## Features
- Built-in support for LLM
- Built-in support for search engine
- Customizable pretty UI interface
- Shareable, cached search results

## Setup Search Engine API
There are two default supported search engines: Bing and Google.
 
### Bing Search
To use the Bing Web Search API, please visit [this link](https://www.microsoft.com/en-us/bing/apis/bing-web-search-api) to obtain your Bing subscription key. You also need to replace the `search_with_google` function in `rag_chain.py` with `search_with_bing` in `search_with_lepton.py`. Change the service keys accordingly.

### Google Search
You have three options for Google Search: you can use the [SearchApi Google Search API](https://www.searchapi.io/) from SearchApi, [Serper Google Search API](https://www.serper.dev) from Serper, or opt for the [Programmable Search Engine](https://developers.google.com/custom-search) provided by Google. I have implemented **Programmable Search Engine** in `rag_chain.py`. But you could change it to other ones (see section Bing Search).

## Setup LLM and Search Service

> [!NOTE]
> I don't get access to powerful GPUs :( so I use the [NVIDIA AI Foundation Endpoints](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-foundation/models/mixtral-8x7b/api). But I recommend you replace it with [ollama](https://python.langchain.com/docs/integrations/llms/ollama) if you insist to have everthing set up locally.
> Set up your service keys starting line 23 in `rag_chain.py` like this:
```python
# ===========================================================
# set service keys
NVAPI_KEY = 'nvapi-xxx'
SEARCH_API_KEY_GOOGLE = "xxx"                                          
SEARCH_ID_GOOGLE = "xxx" # cx parameter   
```

> Running the following commands to set up the environment.

```shell
pip install langchain
pip install loguru
pip install --upgrade --quiet langchain-nvidia-ai-endpoints
pip install fastapi
pip install "uvicorn[standard]"
```


## Build


1. Build web
```shell
cd web && npm install && npm run build
```
2. Run server
```shell
python search_with_langchain.py
```

3. Visit your local conversational search engine at http://localhost:8080/ !

## Error Handling

1. prettier/prettier

If you have encounter something like
```shell
 Error: Delete `␍`  prettier/prettier
```
during build, visit `web/.eslintrc.json` and add a line to turn prettier/prettier off like this. (That's how I get around this anyway.)
```json
"rules": {
    "unused-imports/no-unused-imports": "error",
    "prettier/prettier": "off"
  }
```