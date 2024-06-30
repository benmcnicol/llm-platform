# Run Ollama Mistral Model before running this script

import dspy

lm = dspy.OllamaLocal(model='mistral',base_url="http://192.168.20.49:11434/")

colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

dspy.settings.configure(lm=lm, rm=colbertv2_wiki17_abstracts)

from dspy.datasets import HotPotQA

# Load the dataset.
dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0)

# Tell DSPy that the 'question' field is the input. Any other fields are labels and/or metadata.
trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]

len(trainset), len(devset)

train_example = trainset[0]
print("Example Training Questions")
print(f"Question: {train_example.question}")
print(f"Answer: {train_example.answer}")



dev_example = devset[18]

print("Example Dev Set Questions")
print(f"Question: {dev_example.question}")
print(f"Answer: {dev_example.answer}")
print(f"Relevant Wikipedia Titles: {dev_example.gold_titles}")


print(f"For this dataset, training examples have input keys {train_example.inputs().keys()} and label keys {train_example.labels().keys()}")
print(f"For this dataset, dev examples have input keys {dev_example.inputs().keys()} and label keys {dev_example.labels().keys()}")

class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

# Define the predictor.
generate_answer = dspy.Predict(BasicQA)

# Call the predictor on a particular input.
pred = generate_answer(question=dev_example.question)

# Print the input and the prediction.
print(f"Question: {dev_example.question}")
print(f"Predicted Answer: {pred.answer}")

# lm.inspect_history(n=1)


# Define the predictor. Notice we're just changing the class. The signature BasicQA is unchanged.
generate_answer_with_chain_of_thought = dspy.ChainOfThought(BasicQA)

# # Call the predictor on the same input.
# for dev_example in devset[:5]:
#     pred = generate_answer_with_chain_of_thought(question=dev_example.question)

#     # Print the input, the chain of thought, and the prediction.
#     print(f"Question: {dev_example.question}")
#     #print(f"Thought: {pred.rationale.split('.', 1)[1].strip()}")
#     print(f"Thought: {pred.rationale}")
#     print(f"Predicted Answer: {pred.answer}")


# retrieve = dspy.Retrieve(k=3)
# topK_passages = retrieve(dev_example.question).passages

# print(f"Top {retrieve.k} passages for question: {dev_example.question} \n", '-' * 30, '\n')

# for idx, passage in enumerate(topK_passages):
#     print(f'{idx+1}]', passage, '\n')


# retrieve("When was the first FIFA World Cup held?").passages[0]

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    question = data.get('question', '')
    question = "When was the first FIFA World Cup held?"
    print(question)
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    

    pred = generate_answer_with_chain_of_thought(question=question)
    return jsonify({'prediction': pred.answer})

if __name__ == '__main__':
    app.run(debug=False)
