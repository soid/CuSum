from typing import List
import click
import torch
from coop import VAE, util
import rouge
import json
from tqdm import tqdm
from kmeans_pytorch import kmeans


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class CuSumDecoder:
    def __init__(self, model_path):
        self.vae = VAE(model_path + "/model.tar.gz")

    def decode(self, reviews):
        # split sentences
        ins = []
        for rv in reviews:
            ins.extend(rv.split("."))

        # find each sentence representation
        z_raw: torch.Tensor = self.vae.encode(ins)  # dim: [num_sentences * latent_size]

        # run k-means to find 10 center clusters
        cluster_ids_x, cluster_centers = kmeans(
            X=z_raw,
            num_clusters=10,
            distance='euclidean',
            iter_limit=10000,
            tqdm_flag=False,
            device=self.vae.device
        )
        cluster_centers = cluster_centers.to(self.vae.device)

        # create combinations of reviews for averaging
        idxes: List[List[int]] = util.powerset(len(cluster_centers))
        idxes = [x for x in idxes if len(x) <= 8]

        outputs: List[str] = []
        # split in chunks cause otherwise it runs out of GPU memory
        chunk_size = 2**8
        for chunk in tqdm(chunks(idxes, chunk_size), total=len(idxes)//chunk_size):
            # Taking averages for all combinations of latent vectors
            zs: torch.Tensor = torch.stack([cluster_centers[idx].mean(dim=0) for idx in chunk])

            chunk_outs = self.vae.generate(zs)
            outputs.extend(chunk_outs)

        # Input-output overlap is measured by ROUGE-1 F1 score.
        scores = map(lambda x: (util.input_output_overlap(inputs=reviews, output=x), x), tqdm(outputs))
        best: str = max(scores, key=lambda x: x[0])[1]

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

    # reviews = ["i like him! he teaches the class pretty straightforwardly. each class we talk about a different composer. he has ~opinions~ on the core and will share them sometimes and let us debate about women/western/what's considered art/can pop music be art, but this doesn't dominate the discussion -- usually we're talking about the readings, the music, techniques, context, etc. he is good at getting across info and allowing for interesting discussions/participation.", "take this review with a grain of salt, because i only sat in on one class with professor kozak before dropping the class. just wanted to note that the way professor kozak teaches music hum is with two tracks: one day each week, he focuses on music and its qualities (the way you might in most music hums); the other day, he talks about the socio-political ramifications of listening. this involves all sorts of identity politics, talking about why we privilege western, classical music over other forms of music, and breaking down that prejudice.\nprofessor kozak seems like a great teacher. his emphasis on the social and historical background, motivations, and consequences of different pieces seems fascinating, and i'm sure he'll teach that well. there's also no question that his politics line up with a particular strand that's extremely popular on campus. for that reason, many will see his class as a breath of fresh air, and a way to make music relevant to the world as a whole/activism/social justice. on the other hand, those looking for a traditional music hum experience where they'll actually learn the core elements of music (albeit with far less focus on the historical circumstances that gave rise to that music) will find that this class spends only half of its time - at best - on that subject.", "i had kozak for two classes, diatonic i and ii, and i thought he was the man. he explains things very well, and takes his job very seriously. he can be a little difficult to approach, but he covers each concept thoroughly, and is always willing to take time to explain things one-on-one if need be.\ncompared to the other sections of diatonic i and ii, our class was way ahead of them by the end of the year\u2013\u2013which is a really good thing if you're moving on to theory iii & iv.\nif you're willing to put in work, you should definitely take theory with him. if you find that he is too by-the-book for you, then you have plenty of other semesters of theory to try professors that are more your style.", "this is the worst professor i have ever had at columbia. if you don't want to read the rest of this review just take away one thing: do not any class with mariusz kozak.\nkozak does not teach. he makes the class read the textbook and we come to class and he says \"any questions? no? lets do some exercises.\" he forces us to work in groups when its absolutely meaningless to, because you have to do a lot of figuring out yourself. you will learn nothing in this class because he's not doing this class to help you learn; he literally just wants you to work hard.\ni approached him once asking why he took points off my assignment/tests and his response was literally... \"i... i can't give you an answer.\" seriously? you tell me my answer is wrong and you can't justify why your answer is right??\nthis class was a horror to take and every lesson was ridiculous and frustrating. every class we would walk away feeling confused. take my advice and never take a class with this joke of a professor. you would have learnt more if you just read the textbook (it's not like we did anything more than that in his class, except he would confuse us more)."]
    # reference = "Professor is generally liked by their students. The teaching style is straightforward; professor is opinionated, but allows interesting discussion. The class felt disengaged from the class as professor split them into groups that were meaningless. But it may be worth trying and seeing if you like their teaching style."

    outputs, best = csd.decode(reviews)

    # print("Outputs:")
    # print(outputs)

    print()
    print("best input-output overlap:")
    print(best)

    print()
    print("Reference:")
    print(reference)

    # compute rouge for the sample
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
