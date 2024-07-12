# MobileBERT
Based on [MobileBERT](https://github.com/gemde001/MobileBERT)

# What is BERT?
BERT (Bidirectional Encoder Representations from Transformers) is a NLP model from Google AI, published in 2018. </br>
BERT can do many things, including what is demoed here: Questions and Answers-> pass any text and ask questions about it. 

This demo runs on top of the mobile (lightweight TensorflowLite-based) version. 

## Running the demo locally
### install requirements
* requires python3. Create virtual environment
```sh
python -m venv bert
```

* install dependencies:

```sh
pip3 install -r requirements.txt
```

### download the Model
[Download model and vocab.txt](https://www.tensorflow.org/lite/models/bert_qa/overview) 
and copy it to 
```
./demo/mobilebert/
    + vocab.txt
    + mobilebert.tflite
```

### run server
```sh
cd ./demo
python3 main.py
```
* Open http://localhost:5000


### The OG code
the Original code from [gemde001](https://github.com/gemde001/MobileBERT) is extremely simple to use, take a look at it
