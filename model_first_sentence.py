# baseline model: just combine first sentence from all reviews
import click
import json
import rouge


def predict(reviews):
    sum = ""
    for r in reviews:
        s = r.split(".", 1)[0]
        sum += s + ". "
    return sum


@click.command()
@click.argument("file_path", type=click.Path(exists=True))
def main(file_path):
    # read data
    f = open(file_path, "r")
    data = json.loads(f.read())
    f.close()

    # generate
    hyp, ref = [], []
    for entity in data:
        hyp.append(predict(entity['reviews']))
        ref.append(entity['summary'][0])

    # evaluate
    evaluator = rouge.Rouge(metrics=["rouge-n", "rouge-l"], max_n=2, limit_length=False, apply_avg=True,
                            stemming=True, ensure_compatibility=True)
    sums = evaluator.get_scores(hyp, ref)
    for metric in ['rouge-1', 'rouge-2', 'rouge-l']:
        val = sums[metric]
        print("  ", metric, ":", "%.2f" % (val['f']*100))


if __name__ == '__main__':
    main()
