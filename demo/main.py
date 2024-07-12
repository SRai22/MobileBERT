from flask import Flask, request, render_template
from flask_cors import CORS
from mobilebert import MobileBERT
import wikipedia

bert = MobileBERT()

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

@app.route('/api', methods=['GET'])
def run():
    question = request.args.get('question', default=None, type=str)
    content = request.args.get('content', default=None, type=str)
    topic = request.args.get('topic', default=None, type=str)

    print(question)

    answer = bert.run(question, content)

    print(answer)

    return {"answer": answer}

@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
