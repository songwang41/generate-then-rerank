from rouge_score import rouge_scorer, scoring
import json
import nltk

nltk.download('punkt')
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')


# read data
with open('../data/samsum/test_data.json', encoding='utf-8') as f:
        raw_data = json.load(f)
    # process dialogue
references = []
for d in raw_data:
    references.append('\n'.join(sent_detector.tokenize(d['summary'])))



rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

results = {
    "rouge1":[],
    "rouge2":[],
    "rougeL":[],
    "rougeLsum":[]
}

scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=True)

aggregator_1 = scoring.BootstrapAggregator()
aggregator_2 = scoring.BootstrapAggregator()
aggregator_L = scoring.BootstrapAggregator()
aggregator_Lsum = scoring.BootstrapAggregator()

for num in range(1,41):
    with open('results/generated_predictions_%d.txt'%(num)) as f:
        total_cnt = 0
        for i, ref in enumerate(references):
            pred_1 = ""
            max_1 = 0
            pred_2 = ""
            max_2 = 0
            pred_L = ""
            max_L = 0
            pred_Lsum = ""
            max_Lsum = 0
            for j in range(num):
                line = f.readline().strip()
                line = '\n'.join(sent_detector.tokenize(line))
                score = scorer.score(ref, line)
                if score['rouge1'].fmeasure > max_1:
                    max_1 = score['rouge1'].fmeasure
                    pred_1 = line
                if score['rouge2'].fmeasure > max_2:
                    max_2 = score['rouge2'].fmeasure
                    pred_2 = line
                if score['rougeL'].fmeasure > max_L:
                    max_L = score['rougeL'].fmeasure
                    pred_L = line
                if score['rougeLsum'].fmeasure > max_Lsum:
                    max_Lsum = score['rougeLsum'].fmeasure
                    pred_Lsum = line
            aggregator_1.add_scores(scorer.score(ref, pred_1))
            aggregator_2.add_scores(scorer.score(ref, pred_2))
            aggregator_L.add_scores(scorer.score(ref, pred_L))
            aggregator_Lsum.add_scores(scorer.score(ref, pred_Lsum))

        result_1 = aggregator_1.aggregate()
        result_2 = aggregator_2.aggregate()
        result_L = aggregator_L.aggregate()
        result_Lsum = aggregator_Lsum.aggregate()

        results["rouge1"].append(result_1["rouge1"].mid.fmeasure)
        results["rouge2"].append(result_2["rouge2"].mid.fmeasure)
        results["rougeL"].append(result_L["rougeL"].mid.fmeasure)
        results["rougeLsum"].append(result_Lsum["rougeLsum"].mid.fmeasure)

with open('eval_diverse_result.json', 'w', encoding='utf-8') as f:
    json.dump(results, f)


