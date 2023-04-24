from researcher import ResearchAI

def research_loop():
    rai = ResearchAI()

    import ipdb; ipdb.set_trace()

    answer = None
    while not rai.main_question_answered:
        rai.get_proposed_research_json()
        rai.summarize_findings()
        rai.devise_sub_questions()
        
        rai.search_google()
        rai.search_papers()

        rai.answer_sub_questions()
        answer = rai.answer_main_question()

    answer = rai.summarize_findings()
    print(f"Answer to the main research question: {answer}")


if __name__ == "__main__":
    research_loop()
