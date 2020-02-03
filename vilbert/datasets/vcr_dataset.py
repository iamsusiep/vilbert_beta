import json
from typing import Any, Dict, List
import random
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import _pickle as cPickle
import json_lines

from pytorch_pretrained_bert.tokenization import BertTokenizer
from ._image_features_reader import ImageFeaturesH5Reader
import pdb
import csv
import sys
from collections import *
def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)

def _converId(img_id):

    img_id = img_id.split('-')
    if 'train' in img_id[0]:
        new_id = int(img_id[1])
    elif 'val' in img_id[0]:
        new_id = int(img_id[1]) + 1000000        
    elif 'test' in img_id[0]:
        new_id = int(img_id[1]) + 2000000    
    else:
        pdb.set_trace()

    return new_id

def _load_emotion_annotation(emotions_jsonpath): 
    emotion_entries = defaultdict(dict) 
    with open(emotions_jsonpath, 'rb') as f:
        for annotation in json_lines.reader(f):
            img_id = annotation["img_id"]
            emotion_entries[img_id] = dict(annotation["info_dict"])
    return emotion_entries

def _load_annotationsQ_A(annotations_jsonpath, split):
    """Build an index out of FOIL annotations, mapping each image ID with its corresponding captions."""
    entries = []
    with open(annotations_jsonpath, 'rb') as f: # opening file in binary(rb) mode    
        for annotation in json_lines.reader(f):
            with open(os.path.join('data/VCR/vcr1images', annotation["metadata_fn"]), 'r') as mf:
                metadata_fn = json.load(mf)
                boxes = metadata_fn["boxes"]
                det_names = metadata_fn["names"]
                #det_names = ""
                question = annotation["question"]
                if split == 'test':
                    ans_label = 0
                else:
                    ans_label = annotation["answer_label"]
                img_id = _converId(annotation["img_id"])
                anno_id = int(annotation["annot_id"].split('-')[1])
                entries.append(
                    {"question": question, 'answers':annotation["answer_choices"], "metadata_fn": annotation["metadata_fn"], 'target':ans_label, 'img_id':img_id, 'anno_id':anno_id, 'boxes':boxes, 'names': det_names}
                )
    return entries

def _load_annotationsQA_R(annotations_jsonpath, split):
    """Build an index out of FOIL annotations, mapping each image ID with its corresponding captions."""
    entries = []
    with open(annotations_jsonpath, 'rb') as f: # opening file in binary(rb) mode    
        for annotation in json_lines.reader(f):
            with open(os.path.join('data/VCR/vcr1images', annotation["metadata_fn"]), 'r') as mf:
                metadata_fn = json.load(mf)
                boxes = metadata_fn["boxes"]
                det_names = ""
                if split == 'test':
                    # for each answer
                    for answer in annotation["answer_choices"]:
                        question = annotation["question"] + ["[SEP]"] + answer
                        img_id = _converId(annotation["img_id"])
                        ans_label = 0
                        anno_id = int(annotation["annot_id"].split('-')[1])
                        entries.append(
                            {"question": question, 'answers':annotation["rationale_choices"], "metadata_fn": annotation["metadata_fn"], 'target':ans_label, 'img_id':img_id, 'boxes': boxes, 'names': det_names}
                        )
                else:
                    det_names = metadata_fn["names"]
                    question = annotation["question"] + ["[SEP]"] + annotation["answer_choices"][annotation['answer_label']]
                    ans_label = annotation["rationale_label"]
                    # img_fn = annotation["img_fn"]
                    img_id = _converId(annotation["img_id"])
                    anno_id = int(annotation["annot_id"].split('-')[1])
                    entries.append(
                        {"question": question, 'answers':annotation["rationale_choices"], "metadata_fn": annotation["metadata_fn"], 'target':ans_label, 'img_id':img_id, 'anno_id':anno_id, 'boxes': boxes, 'names': det_names}
                    )
    return entries

class VCRDataset(Dataset):
    def __init__(
        self,
        task: str,
        dataroot: str,
        annotations_jsonpath: str,
        emotion_jsonpath: str, #TODO: suji
        split: str,
        image_features_reader: ImageFeaturesH5Reader,
        gt_image_features_reader: ImageFeaturesH5Reader,
        tokenizer: BertTokenizer,
        padding_index: int = 0,
        max_seq_length: int = 40,
        max_region_num: int = 60, 
        val_indicator = ""
    ):
        self.emotion2idx = { 'Angry': 0, 'Disgust': 1, 'Fear': 2,'Happy': 3, 'Neutral': 4,'Sad': 5, 'Surprise': 6}
        # All the keys in `self._entries` would be present in `self._image_features_reader`
        if task == 'VCR_Q-A':
            self._entries = _load_annotationsQ_A(annotations_jsonpath, split)
        elif task == "VCR_QA-R":
            self._entries = _load_annotationsQA_R(annotations_jsonpath, split)
        else:
            assert False
        self._split = split
        self._image_features_reader = image_features_reader
        self._gt_image_features_reader = gt_image_features_reader
        self._tokenizer = tokenizer

        self._padding_index = padding_index
        self._max_caption_length = max_seq_length
        self._max_region_num = max_region_num
        self.num_labels = 1

        self._names = []
        with open('data/VCR/unisex_names_table.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[1] != 'name':
                    self._names.append(row[1])

        if not os.path.exists(os.path.join(dataroot, "cache")):
            os.makedirs(os.path.join(dataroot, "cache"))
        self.emotion_entries = _load_emotion_annotation(emotion_jsonpath)
        cache_path = "data/VCR/pretrained_emotion_all_small_update/" + split + val_indicator + '_' + task + "_" + str(max_seq_length) + "_" + str(max_region_num) + "_vcr.pkl"
        # cache file path data/cache/train_ques
        if not os.path.exists(cache_path):
            self.tokenize()
            self.tensorize()
            cf = open(cache_path, 'wb')
            cPickle.dump(self._entries,cf)
            cf.close() 
        else:
            print("cache read")
            cf = open(cache_path, "rb")
            self._entries = cPickle.load(cf)
            cf.close()
        print("init complete")
    def tokenize(self):
        """Tokenizes the captions.

        This will add caption_tokens in each entry of the dataset.
        -1 represents nil, and should be treated as padding_idx in embedding.

        emotion_ids: self._max_caption_length X 7
        """
        count = 0
        for entry in self._entries:
            mf = open(os.path.join('data/VCR/vcr1images', entry["metadata_fn"]), 'r')
            metadata_fn = json.load(mf)
            det_names = metadata_fn["names"]
            random_names, person_dict, person_idx = self.generate_random_name(det_names)
            self.person_idx = person_idx
            # replace with name
            img_id = entry["img_id"]
            info_dict = self.emotion_entries[int(img_id)] 
            tokens_a, mask_a,emotion_mask_a = self.replace_det_with_name(entry["question"], random_names, info_dict)
            
            input_ids_all = []
            co_attention_mask_all = []
            input_mask_all = []
            segment_ids_all = []
            emotion_ids_all = []

            for answer in entry["answers"]:
                tokens_b, mask_b,emotion_mask_b = self.replace_det_with_name(answer, random_names, info_dict)
                self._truncate_seq_pair(tokens_a, tokens_b, mask_a, mask_b, emotion_mask_a, emotion_mask_b, self._max_caption_length - 3)
                tokens = []
                segment_ids = []
                emotion_ids = []

                tokens.append("[CLS]")
                segment_ids.append(0)
                emotion_ids.append([0]*7)
                    
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)
                for emotion_m in emotion_mask_a:
                    if emotion_m == -1:
                        emotion_ids.append([0]*7)
                    else:
                        a = [0] * 7
                        a[emotion_m] = 1
                        emotion_ids.append(a)
                tokens.append("[SEP]")
                segment_ids.append(0)
                emotion_ids.append([0]* 7)

                assert len(tokens_b) > 0
                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                for emotion_m in emotion_mask_b:
                    if emotion_m == -1:
                        emotion_ids.append([0]*7)
                    else:
                        a = [0] * 7
                        a[emotion_m] = 1
                        emotion_ids.append(a)
                
                tokens.append("[SEP]")
                segment_ids.append(1)
                emotion_ids.append([0] * 7)
                input_ids = self._tokenizer.convert_tokens_to_ids(tokens)
                co_attention_mask = [-1] + mask_a + [-1] + mask_b + [-1]

                input_mask = [1] * len(input_ids)
                # Zero-pad up to the sequence length.
                while len(input_ids) < self._max_caption_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)
                    co_attention_mask.append(-1)
                    emotion_ids.append([0] * 7)
                assert len(input_ids) == self._max_caption_length
                assert len(input_mask) == self._max_caption_length
                assert len(segment_ids) == self._max_caption_length
                assert len(emotion_ids) == self._max_caption_length
                co_attention_mask_all.append(co_attention_mask)
                input_ids_all.append(input_ids)
                input_mask_all.append(input_mask)
                segment_ids_all.append(segment_ids)
                emotion_ids_all.append(emotion_ids) 
            entry["co_attention_mask"] = co_attention_mask_all
            entry["input_ids"] = input_ids_all
            entry["input_mask"] = input_mask_all
            entry["segment_ids"] = segment_ids_all
            entry["emotion_ids"] = emotion_ids_all
            sys.stdout.write('%d/%d\r' % (count, len(self._entries)))
            sys.stdout.flush()
            count += 1
            mf.close()

    def tensorize(self):
        for entry in self._entries:
            input_ids = torch.from_numpy(np.array(entry["input_ids"]))
            entry["input_ids"] = input_ids

            input_mask = torch.from_numpy(np.array(entry["input_mask"]))
            entry["input_mask"] = input_mask

            segment_ids = torch.from_numpy(np.array(entry["segment_ids"]))
            entry["segment_ids"] = segment_ids

            emotion_ids = torch.from_numpy(np.array(entry["emotion_ids"])).float()
            #print('emotion_t:',torch.sum(emotion_ids,-1))
            entry["emotion_ids"] = emotion_ids
    def generate_random_name(self, det_names):
        """
        person_idx: key is name_no, value is person_no
        person_dict: person 0 --> "Anne"
        """
        random_name = []
        person_dict = defaultdict(str)
        person_idx = dict()
        i = 0
        for idx, name in enumerate(det_names):
            if name == 'person':
                word = random.choice(self._names)
                person_dict[i] = word
                person_idx[idx] = i
                i += 1
            else:
                word = name
            random_name.append(word)
        return random_name, person_dict, person_idx
    def replace_det_with_name(self, inputs, random_names, info_dict):
        tokens = []
        mask = []
        emotion_mask = []
        for w in inputs:
            if isinstance(w, str):
                word = w
                det = -1
                word_token = self._tokenizer.tokenize(word)
                mask += [det] * len(word_token)
                emotion_mask += [-1] * len(word_token)
                tokens += word_token
            else:
                for idx in w:
                    word = random_names[idx]
                    word_token = self._tokenizer.tokenize(word)
                    mask += [idx] * len(word_token)
                    '''print('replace_det_with_name')
                    print('info_dict', info_dict)
                    print('idx', idx)
                    print('person_idx',self.person_idx)'''
                    if idx not in self.person_idx.keys() or str(self.person_idx[idx]) not in info_dict.keys():
                        emotion_mask += [-1] * len(word_token)
                    else:
                        #print("emotion added")
                        emotion = info_dict[str(self.person_idx[idx])]["emotion"]
                        emotion_mask += [self.emotion2idx[emotion]] * len(word_token)
                    tokens += word_token
        return tokens, mask, emotion_mask

    def _truncate_seq_pair(self, tokens_a, tokens_b, mask_a, mask_b, emotion_mask_a, emotion_mask_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
                mask_a.pop()
                emotion_mask_a.pop()
            else:
                tokens_b.pop()
                mask_b.pop()
                emotion_mask_b.pop()

    def bb_intersection_over_union(self, boxA, boxB):
        #https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
     
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
     
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
     
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
     
        # return the intersection over union value
        return iou

    def __getitem__(self, index):
        """
        emotion_ids = text, 4 X 60 X7
        emotion_v = image, 100 X 7
        """
        entry = self._entries[index]
        image_id = entry["img_id"]

        features, num_boxes, boxes, image_location_ori = self._image_features_reader[image_id]
        boxes = boxes[:num_boxes]
        features = features[:num_boxes]

        gt_features, gt_num_boxes, gt_boxes, gt_image_location_ori = self._gt_image_features_reader[image_id]
        # merge two features.
        features[0] = (features[0] * num_boxes + gt_features[0] * gt_num_boxes) / (num_boxes + gt_num_boxes)

        # merge two boxes, and assign the labels. 
        gt_boxes = gt_boxes[1:gt_num_boxes]
        gt_features = gt_features[1:gt_num_boxes]
        gt_image_location_ori = gt_image_location_ori[1:gt_num_boxes]
        gt_num_boxes = gt_num_boxes - 1

        gt_box_preserve = min(self._max_region_num-1, gt_num_boxes)
        gt_boxes = gt_boxes[:gt_box_preserve]
        gt_features = gt_features[:gt_box_preserve]
        gt_image_location_ori = gt_image_location_ori[:gt_box_preserve]
        gt_num_boxes = gt_box_preserve
 
        num_box_preserve = min(self._max_region_num - int(gt_num_boxes), int(num_boxes))
        boxes = boxes[:num_box_preserve]
        features = features[:num_box_preserve]
        image_location_ori = image_location_ori[:num_box_preserve]
        # concatenate the boxes
        mix_boxes = np.concatenate((boxes, gt_boxes), axis=0)
        mix_features = np.concatenate((features, gt_features), axis=0)
        mix_num_boxes = num_box_preserve + int(gt_num_boxes)
        mix_image_location_ori = np.concatenate((image_location_ori, gt_image_location_ori), axis=0)
        image_mask = [1] * (mix_num_boxes)
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        mix_boxes_pad = np.zeros((self._max_region_num, 5))
        mix_features_pad = np.zeros((self._max_region_num, 2048))
        #mix_image_location_ori_pad = np.zeros((self._max_region_num, 5))

        mix_boxes_pad[:mix_num_boxes] = mix_boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = mix_features[:mix_num_boxes]

        # appending the target feature.
        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        vcr_annot= [] # person no is index, boundaries are values
        for idx, (b, n) in enumerate(zip(entry["boxes"], entry["names"])): 
            if n == 'person':
                x1, y1, x2, y2, _ = b
                vcr_annot.append([x1, y1, x2, y2])

        gt2idx = dict() #key: gt value: index
        for i, gt in enumerate(vcr_annot):
            best_bb = None
            maxval = float('-inf')
            best_idx = None
            for idx, pred_bb in enumerate(mix_image_location_ori):
                pred_bb = pred_bb[:-1]
                iou = self.bb_intersection_over_union(gt, pred_bb)
                if iou > 0.5 and iou > maxval:
                    maxval = iou
                    best_bb = pred_bb
                    best_idx = idx
            if best_idx != None:
                gt2idx[i] = best_idx
        
        emotion_v_prev = [-1] * 100 
        for person_no, pred_idx in gt2idx.items():
            info_dict = self.emotion_entries[image_id] 
            if str(person_no) in info_dict.keys():
                emotion = info_dict[str(person_no)]["emotion"]
                emotion_index = self.emotion2idx[emotion]
                emotion_v_prev[pred_idx] = emotion_index
        emotion_v = []
        for emotion_m in emotion_v_prev:
            if emotion_m == -1:
                emotion_v.append([0]*7)
            else:
                a = [0] * 7
                a[emotion_m] = 1
                emotion_v.append(a)
        emotion_v = torch.tensor(emotion_v).float()
        input_ids = entry["input_ids"]
        input_mask = entry["input_mask"]
        segment_ids = entry["segment_ids"]
        target = int(entry["target"])
        emotion_ids = entry["emotion_ids"]
         
        if self._split == 'test':
            # anno_id = entry["anno_id"]
            anno_id = 0#entry["anno_id"]
        else:
            #anno_id = entry["img_id"]
            anno_id = entry["anno_id"]

        co_attention_idxs = entry["co_attention_mask"]
        co_attention_mask = torch.zeros((len(entry["co_attention_mask"]), self._max_region_num, self._max_caption_length))

        for ii, co_attention_idx in enumerate(co_attention_idxs):
            for jj, idx in enumerate(co_attention_idx):
                if idx != -1 and idx+num_box_preserve < self._max_region_num:
                    co_attention_mask[ii, idx+num_box_preserve, jj] = 1
        return features, spatials, image_mask, input_ids, target, input_mask, segment_ids, co_attention_mask, anno_id, emotion_ids, emotion_v

    def __len__(self):
        return len(self._entries)
