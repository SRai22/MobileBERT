import gradio as gr
from mobile_bert.mobile_bert import MobileBERT
import wikipedia


bert = MobileBERT()

def answer_question(question, context, topic):
    if not context and topic:
        try:
            context = wikipedia.page(topic).content
        except wikipedia.DisambiguationError as e:
            return f"Disambiguation error: {e}"
        except wikipedia.PageError:
            return "Page not found."
    elif not context:
        return "Please provide either context or a topic."
    
    answer = bert.run(question, context)
    return answer

iface = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your question here...", label="Question"),
        gr.Textbox(lines=10, placeholder="Enter the context here or provide a topic below.", label="Context"),
        gr.Textbox(lines=1, placeholder="Enter a Wikipedia topic here if no context is provided.", label="Topic")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Question Answering with MobileBERT",
    description="Ask a question and get an answer based on the provided context or a Wikipedia topic.",
    examples=[
        ["What year was the Declaration of Independence signed?", "", "United States Declaration of Independence"],
    ],
    allow_flagging="never"
)

if __name__ == '__main__':
    iface.launch()
