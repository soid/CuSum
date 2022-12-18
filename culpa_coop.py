from typing import List
import click
import torch
from coop import VAE, util
import rouge
import json
from tqdm import tqdm


class CuSumDecoder:
    def __init__(self, model_path):
        self.vae = VAE(model_path + "/model.tar.gz")

    def decode(self, reviews):
        # encode reviews
        z_raw: torch.Tensor = self.vae.encode(reviews)  # dim: [num_reviews * latent_size]

        # create combinations of reviews for averaging
        idxes: List[List[int]] = util.powerset(len(reviews))

        # Taking averages for all combinations of latent vectors
        zs: torch.Tensor = torch.stack([z_raw[idx].mean(dim=0) for idx in idxes])  # [2^num_reviews - 1 * latent_size]

        outputs: List[str] = self.vae.generate(zs, bad_words=[3])

        # Input-output overlap is measured by ROUGE-1 F1 score.
        best: str = max(outputs, key=lambda x: util.input_output_overlap(inputs=reviews, output=x))

        return outputs, best


@click.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("references_path", type=click.Path(exists=True))
def main(model_path, references_path):
    csd = CuSumDecoder(model_path)

    # CULPA playing
    reviews: List[str] = [
        "She has been one of the most accommodating professors during the coronavirus pandemic. Professor Jhanwar is also always available for office hours and is eager to help any students in need. This course was super manageable, especially as an intro course. It could be boring at times, but is still very comprehensive. Every topic is also taught through PowerPoint slides so don't expect an extremely exciting lecture. That being said, it does help students weed out what is the most useful information. I found that I didn't need to read the textbook and just referred to the slides while studying.",
        "If you are not a STEM person (but need to fulfill the science requirement) and are looking for a super easy A, TAKE THIS CLASS. This class is so manageable and easy. Professor Jhanwar is very nice and the material is presented in an organized way. Her lectures are all done via powerpoint (and she basically just reads off the slides), which I really liked. The class can get boring, but overall the material is interesting and not overly science-y at all. I definitely learned a lot.I never read the textbook because Prof. Jhanwar always sent out review slides before each exam with all the information we would be tested on. If you study it and memorize all the terms on the slides, you'll easily get a A on each exam.This class is pretty much purely about memorization. If you are good at memorizing and want to learn more about psychology or simply need to fulfill the science requirement, I would highly recommend Prof. Jhanwar's section."
    ]
    reference: str = "The professor is liked by the students because the class is easy even though not very engaging. They describe the class as boring, but multiple choice tests are easy.\n"

    outputs, best = csd.decode(reviews)

    print("Outputs:")
    print(outputs)

    print()
    print("best input-output overlap:")
    print(best)

    # compute rouge
    print()
    print("Score:")
    evaluator = rouge.Rouge(metrics=["rouge-n", "rouge-l"],
                            max_n=2, limit_length=False, apply_avg=True,
                            stemming=True, ensure_compatibility=True)
    sums = evaluator.get_scores([best], [reference])
    for metric in ['rouge-1', 'rouge-2', 'rouge-l']:
        val = sums[metric]
        print("  ", metric, ":", "%.2f" % (val['f']*100))


    # run on all references
    print()
    print("Running on all references")
    f = open(references_path, "r")
    data = json.loads(f.read())
    f.close()

    # generate
    hyp, ref = [], []
    max_reviews = 4
    for entity in tqdm(data):
        if len(entity['reviews']) > max_reviews:
            entity['reviews'] = entity['reviews'][:max_reviews]
        outpus, best = csd.decode(entity['reviews'])
        hyp.append(best)
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
