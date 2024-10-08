import numpy as np
import openvino as ov
import bert
import math
import time

class MobileBERT:
    def __init__(self):
        self.max_length = 384
        vocab_path = __file__.replace("__init__.py", "../model/mobilebert/vocab.txt")
        self.tokenizer = bert.bert_tokenization.FullTokenizer("../model/mobilebert/vocab.txt", True)
        
        # Initialize OpenVINO Core
        self.core = ov.Core()
        
        # Load the model
        model_path = __file__.replace("__init__.py", "../model/mobilebert/mobilebert.xml")  # Adjust the path if necessary
        self.model = self.core.read_model(model="../model/mobilebert/mobilebert.xml")
        
        # Compile the model for the integrated GPU
        self.compiled_model = self.core.compile_model(self.model, device_name="GPU")
        
        # Get input and output layer names
        self.input_names = [input.get_any_name() for input in self.model.inputs]
        self.output_names = [output.get_any_name() for output in self.model.outputs]

    def get_summary(self):
        print("Inputs:")
        for input in self.model.inputs:
            print(f"Name: {input.get_any_name()}, Shape: {input.get_shape()}, Type: {input.get_element_type()}")
        print("\nOutputs:")
        for output in self.model.outputs:
            print(f"Name: {output.get_any_name()}, Shape: {output.get_shape()}, Type: {output.get_element_type()}")

    def get_masks(self, tokens):
        if len(tokens) > self.max_length:
            raise IndexError("Token length more than max seq length!")
        return np.asarray([1]*len(tokens) + [0] * (self.max_length - len(tokens)))

    def get_segments(self, tokens):
        if len(tokens) > self.max_length:
            raise IndexError("Token length more than max seq length!")
        segments = []
        current_segment_id = 0
        for token in tokens:
            segments.append(current_segment_id)
            if token == "[SEP]":
                current_segment_id = 1
        return np.asarray(segments + [0] * (self.max_length - len(tokens)))

    def get_ids(self, tokens):
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = token_ids + [0] * (self.max_length - len(token_ids))
        return np.asarray(input_ids)

    def compile_text(self, text):
        text = text.lower().replace("-", " ")
        return ["[CLS]"] + self.tokenizer.tokenize(text) + ["[SEP]"]

    def getEmbedding(self, string):
        stokens = self.compile_text(string)
        if len(stokens) > self.max_length:
            raise IndexError("Token length more than max seq length!")
        input_ids = self.get_ids(stokens)
        input_masks = self.get_masks(stokens)
        input_segments = self.get_segments(stokens)
        
        # Prepare inputs
        inputs = {
            self.input_names[0]: input_ids.reshape(1, -1),
            self.input_names[1]: input_masks.reshape(1, -1),
            self.input_names[2]: input_segments.reshape(1, -1)
        }
        
        # Run inference
        outputs = self.compiled_model.infer_new_request(inputs)
        
        # Get outputs
        start_logits = outputs[self.output_names[0]]
        end_logits = outputs[self.output_names[1]]
        
        return {"start": start_logits, "end": end_logits}

    def run(self, query, context):
        stokens = self.compile_text(query) + self.compile_text(context)
        if len(stokens) > self.max_length:
            raise IndexError("Token length more than max seq length!")
        input_ids = self.get_ids(stokens)
        input_masks = self.get_masks(stokens)
        input_segments = self.get_segments(stokens)
        
        # Prepare inputs
        inputs = {
            self.input_names[0]: input_ids.reshape(1, -1),
            self.input_names[1]: input_masks.reshape(1, -1),
            self.input_names[2]: input_segments.reshape(1, -1)
        }
        
        # Run inference
        outputs = self.compiled_model.infer_new_request(inputs)
        
        # Get outputs
        start_logits = outputs[self.output_names[0]]
        end_logits = outputs[self.output_names[1]]
        
        # Get start and end indices
        start = np.argmax(start_logits, axis=1)[0]
        end = np.argmax(end_logits, axis=1)[0]
        
        # Get the answer tokens
        answer_tokens = stokens[start:end+1]
        answer = " ".join(answer_tokens).replace("[CLS]", "").replace("[SEP]", "").replace(" ##", "")
        return answer

    def square_rooted(self, x):
        return math.sqrt(sum([a*a for a in x]))

    def cosine_similarity(self, x, y):
        numerator = sum(a*b for a,b in zip(x,y))
        denominator = self.square_rooted(x)*self.square_rooted(y)
        return numerator / float(denominator)

if __name__ == "__main__":
    m = MobileBERT()
    m.get_summary()
    avg = 0
    last = ""
    for x in range(0, 9):
        sTime = time.time()
        last = m.run("The declaration was not signed until ?", 
                     "In fact, independence was formally declared on July 2, 1776, a date that John Adams believed would be “the most memorable epoch in the history of America.” On July 4, 1776, Congress approved the final text of the Declaration. It wasn't signed until August 2, 1776.")
        avg += (time.time() - sTime)
    print(str(avg / 10), " seconds")
    print(last)
