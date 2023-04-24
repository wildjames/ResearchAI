# ResearchAI
Leverage LLMs to accelerate literature reviews. The AI should take a reasearch question, survey the literature, and provide a summary of the literature as a whole. The AI should also be able to provide a summary of a given paper, and provide a list of papers that are most relevant to the research question.

Note that while this is largely my own work, I am taking heavy inspiration from the AutoGPT project for some of the common functions.

# Reading List

This is a brief list of the basic functionality I will need:
- [ ] Access the GPT-3 API - properly!
- [ ] Use some kind of local storage to store a vectorised memory of the papers I have ingested
- [ ] Be able to search Google for keywords and relevant general knowledge
- [ ] Be able to search for papers on open-source academic databases (e.g. arXiv, Cryptology ePrint Archive, etc.)
- [ ] Be able to ingest papers into local storage

## Data ingestion
Follow this cookbook to ingest your data into the GPT-3 API: [here](https://github.com/openai/openai-cookbook/blob/main/apps/chatbot-kickstarter/powering_your_products_with_chatgpt_and_your_data.ipynb)
