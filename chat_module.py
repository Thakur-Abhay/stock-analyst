from .data_rover import ContextEnricher
import logging
from .document_qa import answer_question


class StockChatBot:
    def __init__(self):
        self.context_enricher = ContextEnricher()
        self.model_name = "meta-llama/Meta-Llama-3-8B"

    async def generate(self, user_query):
        logging.info(f"Chat module called for the user query: {user_query}")
        

