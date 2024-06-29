# Run Ollama Mistral Model before running this script

import dspy

lm = dspy.OllamaLocal(model='mistral')

colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

dspy.settings.configure(lm=lm, rm=colbertv2_wiki17_abstracts)
