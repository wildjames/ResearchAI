import json
import logging
from dataclasses import dataclass
from typing import List, Tuple

import openai
from config import Config
from modelsinfo import COSTS

logger = logging.getLogger(__name__)

cfg = Config()
openai.api_key = cfg.OPENAI_API_TOKEN
print_total_cost = cfg.DEBUG


def create_chat_message(role, content):
    """
    Create a chat message with the given role and content.

    Args:
    role (str): The role of the message sender, e.g., "system", "user", or "assistant".
    content (str): The content of the message.

    Returns:
    dict: A dictionary containing the role and content of the message.
    """
    return {"role": role, "content": content}


@dataclass
class ResearchAI:
    """The ResearchAI class, which contains all the logic for the research assistant.
    The researcher takes a research question, and any context, and uses the OpenAI API to answer it. It will do so by
    looping a defined "research loop", defining questions and queries, searching the internet for clues, reading papers,
    and summarizing the results until it has an answer to the research question.

    This class will keep track of the AI's cost, and the user's budget, and will stop when the budget is exceeded.
    """

    question: str = None
    sub_questions: List[str] = None
    context: List[str] = None

    sub_questions_answered = False
    main_question_answered = False

    turn = 0

    model = "gpt-3.5-turbo"
    temperature = 0.0
    max_tokens = 500

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0
    total_budget = 0
    debug = cfg.DEBUG

    def reset(self):
        self.main_question_answered = False
        self.sub_questions_answered = False
        self.turn = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0
        self.total_budget = 0.0

    def define_question(self, question: str = None, context: List[str] = None):
        """Set the research question and any context for it.
        If none is given, get the research question from the user. Also prompts for any context they would like to give."""

        if question is None:
            question = input(
                "What is your research question? e.g. 'What is the difference between low mass star formation, and high mass star formation?'\n "
            )
            if question == "":
                raise ValueError("You must provide a research question.")

        if context is None:
            context = []
            print(
                "Is there any additional information, or sub-questions you would like to provide? e.g. 'What is the mass cutoff for a low mass star?' or 'Begin by looking up star formation regions'"
            )
            while True:
                info = input("> ")
                if info:
                    context.append(info)
                else:
                    break

        if self.debug:
            logger.debug(f"Question: {question}")
            logger.debug(f"Context: {context}")

        self.question = question
        self.context = context

        return question, context

    def generate_first_prompt(self):
        """Create a prompt template for the AI to use to answer the research question."""

        content = f"You are a researcher, and you are trying to answer the following research question: {self.question}. "
        content += "You should NOT use your own knowledge to answer the question, but instead use the information contained in these messages ONLY. "
        content += f"Your current sub-questions related to this are: {self.sub_questions}. You have found the following information: {self.context}. "
        content += "Your research takes place in turns, in the following format: "
        content += "1. Receive this message outlining the question and context. "
        content += "2. Define sub-questions relevant to the research question. "
        content += "3. Decide if you need to search the internet for context to aid you in crafting the academic database query in the next step, "
        content += "and if so create a search query. You will consider all information you find on the internet to be unusable to your research."
        content += "4. Create a search query for a database of academic papers. "
        content += "5. Read the results of your internet search and academic paper query, and summarize your findings. "
        content += "6. Attempt to answer your sub-questions, if you have gathered enough information. "
        content += "7. Decide if the answer to your research question is clear, and if so, answer it. "
        content += "8. If you are not sure if you have answered the research question, go back to step 2. "
        content += "9. If you have answered the research question, summarize your findings. 10. End the research session. You are currently in turn {turn}. "
        content += "You will define your actions this turn in two phases, in the form of two valid JSON strings. You are currently in the first phase"
        content += "The first phase is to define your sub-questions, decide if you need to search the internet, create your candidate search queries, "
        content += "and read the results of your search queries. You will give your answer to this in this format: "
        content += '{"sub_questions": [list of strings], "internet_search": boolean, "internet_query": string, "academic_query": string}. '
        content += "You will then receive the results of your internet search and academic paper query, and proceed to the second phase. "

        return content

    def get_proposed_research_json(self):
        """Get the proposed research from the AI in JSON format."""

        content = self.generate_first_prompt()

        messages = [
            create_chat_message("system", content),
        ]

        response = self.create_chat_completion(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # TODO: Check that this is valid JSON, and if it is not then attempt a fix.
        # Probably steal the one from AutoGPT
        json_response = response.choices[0].message.content
        actions = json.loads(json_response)

        return actions

    def create_chat_completion(
        self,
        messages: list,  # type: ignore
        model: str | None = None,
        temperature: float = cfg.temperature,
        max_tokens: int | None = None,
    ) -> str:
        """
        Create a chat completion and update the cost.
        Args:
        messages (list): The list of messages to send to the API.
        model (str): The model to use for the API call.
        temperature (float): The temperature to use for the API call.
        max_tokens (int): The maximum number of tokens for the API call.
        Returns:
        str: The AI's response.
        """
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if self.debug:
            logger.debug(f"Response: {response}")
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        self.update_cost(prompt_tokens, completion_tokens, model)
        return response

    def embedding_create(
        self,
        text_list: List[str],
        model: str = "text-embedding-ada-002",
    ) -> List[float]:
        """
        Create an embedding for the given input text using the specified model.

        Args:
        text_list (List[str]): Input text for which the embedding is to be created.
        model (str, optional): The model to use for generating the embedding.

        Returns:
        List[float]: The generated embedding as a list of float values.
        """
        response = openai.Embedding.create(input=text_list, model=model)

        self.update_cost(response.usage.prompt_tokens, 0, model)
        return response["data"][0]["embedding"]

    def update_cost(self, prompt_tokens, completion_tokens, model):
        """
        Update the total cost, prompt tokens, and completion tokens.

        Args:
        prompt_tokens (int): The number of tokens used in the prompt.
        completion_tokens (int): The number of tokens used in the completion.
        model (str): The model used for the API call.
        """
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost += (
            prompt_tokens * COSTS[model]["prompt"]
            + completion_tokens * COSTS[model]["completion"]
        ) / 1000
        if print_total_cost:
            print(f"Total running cost: ${self.total_cost:.3f}")

    def set_total_budget(self, total_budget):
        """
        Sets the total user-defined budget for API calls.

        Args:
        prompt_tokens (int): The number of tokens used in the prompt.
        """
        self.total_budget = total_budget

    def get_total_prompt_tokens(self):
        """
        Get the total number of prompt tokens.

        Returns:
        int: The total number of prompt tokens.
        """
        return self.total_prompt_tokens

    def get_total_completion_tokens(self):
        """
        Get the total number of completion tokens.

        Returns:
        int: The total number of completion tokens.
        """
        return self.total_completion_tokens

    def get_total_cost(self):
        """
        Get the total cost of API calls.

        Returns:
        float: The total cost of API calls.
        """
        return self.total_cost

    def get_total_budget(self):
        """
        Get the total user-defined budget for API calls.

        Returns:
        float: The total budget for API calls.
        """
        return self.total_budget


if __name__ in "__main__":
    rai = ResearchAI()
    rai.define_question(
        question="What is the difference between low mass star formation, and high mass star formation?",
        context=[
            "What is the mass cutoff for a low mass star?",
            "Begin by looking up star formation regions",
        ],
    )

    actions = rai.get_proposed_research_json()
    
    from pprint import pprint
    pprint(actions)

    print(f"Cost of this run: ${rai.get_total_cost()}")
