import dspy
import dotenv
import os

dotenv.load_dotenv()
lm = dspy.LM("gemini/gemini-2.5-flash", api_key=os.getenv("GEMINI_KEY"))
dspy.configure(lm=lm)


# for testing
if __name__ == "__main__":
    math = dspy.ChainOfThought("question -> answer: float")
    print(
        math(
            question="Two dice are tossed. What is the probability that the sum equals 6?"
        )
    )
