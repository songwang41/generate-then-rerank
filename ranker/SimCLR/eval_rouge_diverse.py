from rouge_score import rouge_scorer, scoring
import json
import nltk

nltk.download('punkt')
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')


rouge_types = ["rouge1", "rouge2",  "rougeLsum"]

results = {
    "rouge1":[],
    "rouge2":[],
    "rougeLsum":[]
}

scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=True)

aggregator = scoring.BootstrapAggregator()


for i in range(1,41):
    preds = []
    refs = []
    for idx in range(819):
        with open('result/samsum-from32/candidate/rank_from_%d/%d.dec'%(i,idx), encoding='utf-8') as f:
            line = f.readline().strip()
            preds.append('\n'.join(sent_detector.tokenize(line)))
        with open('result/samsum-from32/reference/rank_from_%d/%d.ref'%(i,idx), encoding='utf-8') as f:
            line = f.readline().strip()
            refs.append('\n'.join(sent_detector.tokenize(line)))

    for p,r in zip(preds, refs):
        score = scorer.score(r, p)
        # print(score)
        aggregator.add_scores(score)

    result = aggregator.aggregate()
    # print(result)

    results["rouge1"].append(result["rouge1"].mid.fmeasure)
    results["rouge2"].append(result["rouge2"].mid.fmeasure)
    results["rougeLsum"].append(result["rougeLsum"].mid.fmeasure)

with open('eval_diverse_result.json', 'w', encoding='utf-8') as f:
    json.dump(results, f)
