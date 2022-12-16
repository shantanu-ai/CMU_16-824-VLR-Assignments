"""Dataset class for VQA."""

import os
import re
from collections import Counter

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from vqa_api import VQA


class VQADataset(Dataset):
    """VQA dataset class."""

    def __init__(self, image_dir, question_json_file_path,
                 annotation_json_file_path, image_filename_pattern,
                 answer_to_id_map=None, answer_list_length=5216, size=224):
        """
        Initialize dataset.

        Args:
            image_dir (str): Path to the directory with COCO images
            question_json_file_path (str): Path to json of questions
            annotation_json_file_path (str): Path to json of mapping
                images, questions, and answers together
            image_filename_pattern (str): The pattern the filenames
                (eg "COCO_train2014_{}.jpg")
        """
        # load the VQA api
        self._vqa = VQA(
            annotation_file=annotation_json_file_path,
            question_file=question_json_file_path
        )
        # also initialize whatever you need from self._vqa
        self.questions = self._vqa.questions['questions']
        self._image_dir = image_dir
        self._image_filename_pattern = image_filename_pattern

        # Publicly accessible dataset parameters
        self.answer_list_length = answer_list_length + 1
        self.unknown_answer_index = answer_list_length
        self.size = size

        # Create the answer map if necessary
        keys = sorted(self._vqa.qa.keys())
        if answer_to_id_map is None:
            all_answers = [
                ' '.join([
                    re.sub(r'\W+', '', word)
                    for word in a['answer'].lower().split()
                ])
                for key in keys
                for a in self._vqa.qa[key]['answers']
            ]
            self.answer_to_id_map = self._create_id_map(
                all_answers, answer_list_length
            )
        else:
            self.answer_to_id_map = answer_to_id_map

    def _create_id_map(self, word_list, max_list_length):
        """
        Create a str-id map for most common words.

        Args:
            word_list: a list of str, with most frequent elements picked out
            max_list_length: the number of strs picked

        Returns:
            A map (dict) from str to id (rank)
        """
        common = Counter(word_list).most_common(max_list_length)
        return {tup[0]: t for t, tup in enumerate(common)}

    def __len__(self):
        # TODO
        return len(self.questions)

    def __getitem__(self, idx):
        """
        Load an item of the dataset.

        Args:
            idx: index of the data item

        Returns:
            A dict containing torch tensors for image, question and answers
        """
        q_anno = self._vqa.load_qa(self.questions[idx]['question_id'])[0]  # TODO load annotation
        q_str = self.questions[idx]['question']
        # q_str.append()  # TODO question in str format

        # Load and pre-process image
        name = str(q_anno['image_id'])
        if len(name) < 12:
            name = '0' * (12 - len(name)) + name
        img_name = self._image_filename_pattern.format(name)
        _img = Image.open(
            os.path.join(self._image_dir, img_name)
        ).convert('RGB')
        width, height = _img.size
        max_wh = max(width, height)
        mean_ = [0.485, 0.456, 0.406]
        std_ = [0.229, 0.224, 0.225]
        preprocessing = transforms.Compose([
            transforms.Pad((0, 0, max_wh - width, max_wh - height)),
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean_, std_)
        ])
        img = preprocessing(_img)
        orig_prep = transforms.Compose([
            transforms.Pad((0, 0, max_wh - width, max_wh - height)),
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor()
        ])
        orig_img = orig_prep(_img)

        # Encode answer to tensor
        a_tensor = torch.zeros(len(q_anno['answers']), self.answer_list_length)
        for a, ans in enumerate(q_anno['answers']):
            a_tensor[
                a, self.answer_to_id_map.get(
                    ' '.join([
                        re.sub(r'\W+', '', word)
                        for word in ans['answer'].lower().split()
                    ]),
                    self.unknown_answer_index
                )
            ] = 1
        a_tensor = a_tensor.any(0).float()  # keep all answers!

        return {
            'image': img,
            'question': q_str,
            'answers': a_tensor,
            'orig_img': orig_img
        }


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    data_path = './data/'  # change this to your data path
    anno_file = data_path + 'mscoco_train2014_annotations.json'
    q_file = data_path + 'OpenEnded_mscoco_train2014_questions.json'
    image_dir = "/ocean/projects/asc170022p/shg121/PhD/CMU-Visual-Learning-Recognition/Hw3-vqa-main/data/train2014"
    image_filename_pattern = "COCO_train2014_{}.jpg"
    vqa_ds = VQADataset(
        image_dir=image_dir,
        question_json_file_path=q_file,
        annotation_json_file_path=anno_file,
        image_filename_pattern=image_filename_pattern
    )

    # print(vqa_ds[1])
    # print()
    print(vqa_ds[1]["question"], '', vqa_ds[1]["answers"])
    print(vqa_ds[1]["answers"].size())
    print()
    print(vqa_ds[2]["question"], '', vqa_ds[2]["answers"])
    print()
    print(vqa_ds[3]["question"], '', vqa_ds[3]["answers"])
    print()
    print(vqa_ds[4]["question"], '', vqa_ds[4]["answers"])
    print()
    print(vqa_ds[5]["question"], '', vqa_ds[5]["answers"])
    print()

    print(len(vqa_ds))
    print(len(vqa_ds.answer_to_id_map))

