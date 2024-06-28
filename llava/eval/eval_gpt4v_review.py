import argparse
import json
import os
import base64
import openai
import time

NUM_SECONDS_TO_SLEEP = 0.5
openai.api_key = os.environ["OPENAI_API_KEY"]

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
def get_eval(image, content: str, max_tokens: int):
    image = encode_image(image)
    while True:
        try:
            response = openai.ChatCompletion.create(
                model='gpt-4o',
                messages=[
                {
                    'role': 'user',
                    'content': [
                        {"type" : "text",
                         "text" : content},
                        {
                        "type" : "image_url",
                        "image_url" : {
                            "url": f"data:image/jpeg;base64,{image}"
                        }
                        }
                    ],
                },],
                temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                max_tokens=max_tokens,
            )
            break
        except openai.error.RateLimitError:
            pass
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)

    return response['choices'][0]['message']['content']


def parse_score(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print('error', review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-q', '--question')
    parser.add_argument('-c', '--context')
    parser.add_argument('-a', '--answer-list', nargs='+', default=[])
    parser.add_argument('-r', '--rule')
    parser.add_argument('-o', '--output')
    parser.add_argument('--image_folder',type=str)
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()

    f_q = [json.loads(q) for q in open(os.path.expanduser(args.question), "r")]
    rule_dict = json.load(open(os.path.expanduser(args.rule), 'r'))

    if os.path.isfile(os.path.expanduser(args.output)):
        print("continue")
        cur_reviews = [json.loads(line) for line in open(os.path.expanduser(args.output))]
    else:
        cur_reviews = []

    review_file = open(f'{args.output}', 'a')

    handles = []
    idx = 2000
    for d in f_q[idx:]:
        image_file = d['img'] # image name
        questions = d['prompt'] # a list of questions
        anaswers = d['answers']
        image = os.path.join(args.image_folder, image_file)
        rule = rule_dict['preference']
        prompt = rule['prompt']
        content = (f'[Question]\n{questions}\n\n'
                   f'[Answer 1]\n{anaswers[0]}\n\n[End of Answer 1]\n\n'
                   f'[Answer 2]\n{anaswers[1]}\n\n[End of Answer 2]\n\n'
                   f'[System]\n{prompt}\n\n')
        cur_js = {
            'q': questions,
            'img':image_file,

        }
        if idx >= len(cur_reviews):
            review = get_eval(image, content, args.max_tokens)
            scores = parse_score(review)
            print(scores)
            if scores[0] == -1:
                continue
            if scores[0]>scores[1]:
                pref_a = anaswers[0]
                rej_a = anaswers[1]
            else:
                pref_a = anaswers[1]
                rej_a = anaswers[0]
            cur_js['p'] = pref_a
            cur_js['r'] = rej_a
            review_file.write(json.dumps(cur_js) + '\n')
            review_file.flush()
        else:
            print(f'Skipping {idx} as we already have it.')
        idx += 1
    review_file.close()
