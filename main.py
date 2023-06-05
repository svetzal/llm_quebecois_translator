import os
from textwrap import fill
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI

os.environ['OPENAI_API_KEY'] = os.environ['OPENAI_ACCESS_TOKEN']


def main():
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-4")
    original_phrase = """
    
Interesting how LLMs have mesmerized us with their illusion of thought. They mirror our ideas back at us in 
delightful new combinations. These ideas aren’t evenly known across humankind, and so we discover new ideas written 
down by others in the process. There is value in the remix. Prompt engineering is interesting because it’s all about 
setting up and asking the right question, to be answered in these delightful new ways. The real power remains in the 
questions we ask and how we ask them.
    
""".strip()

    quebecois = english_to_quebecois(llm, original_phrase)
    round_trip_phrase = quebecois_to_english(llm, quebecois)
    analysis = conversion_analysis(llm, original_phrase, round_trip_phrase)
    refined = refine_conversion(llm, original_phrase, quebecois, round_trip_phrase, analysis)
    verify = quebecois_to_english(llm, refined)

    print(f"Original:\n{original_phrase}\n")
    print(f"Final:\n{refined}")


def english_to_quebecois(llm, phrase):
    prompt = """

You are a francophone from Montreal, Quebec. You are a native French speaker, but you also speak English. You are
currently in a conversation with a friend who is a native English speaker.

Translate the phrases your friend says into French as if you were speaking to a fellow francophone in Montreal.

Translate the following into Quebecois:
{phrase}

""".strip()
    return convert_phrase(llm, prompt, phrase)


def quebecois_to_english(llm, phrase):
    prompt = """

You are a anglophone from Montreal, Quebec. You are a native English speaker, but you also speak French. You are
currently in a conversation with a friend who is a native French speaker.

Translate the phrases your friend says into English as if you were speaking to an anglophone in Toronto.

Translate the following into English:
{phrase}

""".strip()
    return convert_phrase(llm, prompt, phrase)


def conversion_analysis(llm, english_phrase, round_trip_phrase):
    prompt = """
    
Original:
{english_phrase}

Round Trip:
{round_trip_phrase}

List the differences between the Original content and the Round Trip content, and highlight the nuance in the 
differences between the two.
    
""".strip()
    task = PromptTemplate(
        input_variables=["english_phrase", "round_trip_phrase"],
        template=prompt,
    )
    chain = LLMChain(llm=llm, prompt=task, verbose=True)
    return chain.run(english_phrase=english_phrase, round_trip_phrase=round_trip_phrase)


def refine_conversion(llm, english_phrase, quebecois, round_trip_phrase, analysis):
    prompt = """

Original:
{english_phrase}

Quebecois:
{quebecois}

Round Trip:
{round_trip_phrase}

Analysis:
{analysis}

Try and re-state the Original english paragraph in quebecois, while attempting to retain the nuanced meaning described
in the Analysis above.

""".strip()
    task = PromptTemplate(
        input_variables=["english_phrase", "quebecois", "round_trip_phrase", "analysis"],
        template=prompt,
    )
    chain = LLMChain(llm=llm, prompt=task, verbose=True)
    return chain.run(english_phrase=english_phrase, quebecois=quebecois, round_trip_phrase=round_trip_phrase,
                     analysis=analysis)


def convert_phrase(llm, prompt, phrase):
    task = PromptTemplate(
        input_variables=["phrase"],
        template=prompt.strip(),
    )
    chain = LLMChain(llm=llm, prompt=task, verbose=True)
    return chain.run(phrase=phrase)


if __name__ == '__main__':
    main()
