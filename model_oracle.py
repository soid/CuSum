# Oracle model: find the sentense with the largest overlap with the gold reference
import click
import heapq
import json
import rouge


def predict(reviews, gold_ref):
    sum = ""
    scores = []
    for r in reviews:
        sentences = r.split(".")
        for sent in sentences:
            hyp, ref = [sent], [gold_ref]
            evaluator = rouge.Rouge(metrics=["rouge-n", "rouge-l"], max_n=2, limit_length=False, apply_avg=True,
                                    stemming=True, ensure_compatibility=True)
            sums = evaluator.get_scores(hyp, ref)
            score = sums['rouge-l']['f']
            scores.append((score, sent))
    top = heapq.nlargest(3, scores)
    sum = " . ".join([sent for score, sent in top])

    return sum

@click.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option('--show-sample/--no-show-sample', default=False)
def main(file_path, show_sample):
    if show_sample:
        reviews: List[str] = [
            "She has been one of the most accommodating professors during the coronavirus pandemic. Professor Jhanwar is also always available for office hours and is eager to help any students in need. This course was super manageable, especially as an intro course. It could be boring at times, but is still very comprehensive. Every topic is also taught through PowerPoint slides so don't expect an extremely exciting lecture. That being said, it does help students weed out what is the most useful information. I found that I didn't need to read the textbook and just referred to the slides while studying.",
            "If you are not a STEM person (but need to fulfill the science requirement) and are looking for a super easy A, TAKE THIS CLASS. This class is so manageable and easy. Professor Jhanwar is very nice and the material is presented in an organized way. Her lectures are all done via powerpoint (and she basically just reads off the slides), which I really liked. The class can get boring, but overall the material is interesting and not overly science-y at all. I definitely learned a lot.I never read the textbook because Prof. Jhanwar always sent out review slides before each exam with all the information we would be tested on. If you study it and memorize all the terms on the slides, you'll easily get a A on each exam.This class is pretty much purely about memorization. If you are good at memorizing and want to learn more about psychology or simply need to fulfill the science requirement, I would highly recommend Prof. Jhanwar's section."
        ]
        reference: str = "The professor is liked by the students because the class is easy even though not very engaging. They describe the class as boring, but multiple choice tests are easy.\n"
        print(predict(reviews, reference))
    else:
        # read data
        f = open(file_path, "r")
        data = json.loads(f.read())
        f.close()

        # generate
        hyp, ref = [], []
        for entity in data:
            hyp.append(predict(entity['reviews'], entity['summary'][0]))
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
