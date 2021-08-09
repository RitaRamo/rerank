"""PyTorch dataset classes for the image captioning training and testing datasets"""

import os
import h5py
import json
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import faiss
from sentence_transformers import SentenceTransformer
import numpy
from tqdm import tqdm
import gc
from scipy.misc import imresize
from PIL import Image
from torch import nn

from toolkit.utils import (
    DATA_CAPTIONS,
    DATA_CAPTION_LENGTHS,
    TRAIN_SPLIT,
    VALID_SPLIT,
    TEST_SPLIT,
    ENCODED_METAS_FILENAME,
    DATASET_SPLITS_FILENAME,
    IMAGENET_IMAGES_MEAN,
    IMAGENET_IMAGES_STD,
    CAPTIONS_FILENAME,
    WORD_MAP_FILENAME,
    TOKEN_START,
    TOKEN_END,
    TRAIN_CONTEXT_IMAGES_FILENAME,
    TRAIN_CONTEXT_FILENAME,
    TRAIN_TARGETS_FILENAME,
    IMAGES_NAMES_FILENAME,
    TOKEN_PAD
)

class CaptionDataset(Dataset):
    """
    PyTorch Dataset that provides batches of images of a given split
    """

    def __init__(self, dataset_splits_dir, features_fn, normalize=None, features_scale_factor=1):
        """
        :param data_dir: folder where data files are stored
        :param features_fn: Filename of the image features file
        :param split: split, indices of images that should be included
        :param normalize: PyTorch normalization transformation
        :param features_scale_factor: Additional scale factor, applied before normalization
        """
        self.image_features = h5py.File(features_fn, "r")
        self.features_scale_factor = features_scale_factor

        # Set PyTorch transformation pipeline
        self.transform = normalize

        # Load image meta data, including captions
        with open(os.path.join(dataset_splits_dir, ENCODED_METAS_FILENAME)) as f:
            self.image_metas = json.load(f)

        #load captions with text for retrieval
        with open(os.path.join(dataset_splits_dir, CAPTIONS_FILENAME)) as f:
            self.captions_text = json.load(f)

        self.captions_per_image = len(next(iter(self.image_metas.values()))[DATA_CAPTIONS])

        with open(os.path.join(dataset_splits_dir, DATASET_SPLITS_FILENAME)) as f:
            self.split = json.load(f)

    def get_image_features(self, coco_id):
        image_data = self.image_features[coco_id][()]
        # scale the features with given factor
        image_data = image_data * self.features_scale_factor
        image = torch.FloatTensor(image_data)
        if self.transform:
            image = self.transform(image)
        return image


    # def get_image_features(self, coco_id):
    #     return torch.zeros((2, 7,7,2048))

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class CaptionTrainDataset(CaptionDataset):
    """
    PyTorch training dataset that provides batches of images with a corresponding caption each.
    """

    def __init__(self, dataset_splits_dir, features_fn, normalize=None, features_scale_factor=1):
        super().__init__(dataset_splits_dir, features_fn,
                         normalize, features_scale_factor)
        self.split = self.split[TRAIN_SPLIT]

    def __getitem__(self, i):
        # Convert index depending on the dataset split
        coco_id = self.split[i // self.captions_per_image]
        caption_index = i % self.captions_per_image

        image = self.get_image_features(coco_id)
        caption = torch.LongTensor(self.image_metas[coco_id][DATA_CAPTIONS][caption_index])
        caption_length = torch.LongTensor([self.image_metas[coco_id][DATA_CAPTION_LENGTHS][caption_index]])

        return image, caption, caption_length

    def __len__(self):
        return len(self.split) * self.captions_per_image


class CaptionEvalDataset(CaptionDataset):
    """
    PyTorch test dataset that provides batches of images and all their corresponding captions.

    """

    def __init__(self, dataset_splits_dir, features_fn, normalize=None, features_scale_factor=1, eval_split="val"):
        super().__init__(dataset_splits_dir, features_fn,
                         normalize, features_scale_factor)
        if eval_split == "val":
            self.split = self.split[VALID_SPLIT]
        else:
            self.split = self.split[TEST_SPLIT]

    def __getitem__(self, i):
        # Convert index depending on the dataset split
        coco_id = self.split[i]

        image = self.get_image_features(coco_id)
        all_captions_for_image = torch.LongTensor(self.image_metas[coco_id][DATA_CAPTIONS])
        caption_lengths = torch.LongTensor(self.image_metas[coco_id][DATA_CAPTION_LENGTHS])

        return image, all_captions_for_image, caption_lengths, coco_id

    def __len__(self):
        return len(self.split)





# class CaptionTrainContextRetrievalDataset(CaptionDataset):
#     """
#     PyTorch training dataset that provides batches of images with a corresponding caption each.
#     """

#     def __init__(self, dataset_splits_dir, features_fn, normalize=None, features_scale_factor=1):
#         super().__init__(dataset_splits_dir, features_fn,
#                          normalize, features_scale_factor)
#         #self.split = self.split[TRAIN_SPLIT]
#         self.split = self.split[VALID_SPLIT]

#         word_map_filename = os.path.join(dataset_splits_dir, WORD_MAP_FILENAME)
#         with open(word_map_filename) as f:
#             self.word_map = json.load(f)
#         self.sentence_model = SentenceTransformer('paraphrase-distilroberta-base-v1')

#     def __getitem__(self, i):
#         # Convert index depending on the dataset split
#         coco_id = self.split[i]

#         image = self.get_image_features(coco_id) #32, 2048
#         image = image.mean(dim=0) #2048 dim

#         image_caps = self.captions_text[coco_id]
#         text_encs = torch.tensor([])
#         targets= []
        
#         for cap_index in range(len(image_caps)):
#             text_caption = TOKEN_START + " " + image_caps[cap_index] + " "+ TOKEN_END
#             words_caption = text_caption.split()
#             for i in range(len(words_caption)):
#                 text_enc=self.sentence_model.encode(words_caption[:i]) #tens de substituir isto... 
#                 text_encs= torch.cat((text_encs,torch.tensor(text_enc)))
#                 targets.append(self.word_map[words_caption[i]])

        
#         images = image.expand(text_encs.size(0), image.size(-1))
#         context = torch.cat((images,text_encs), dim=-1) #(n_contexts, 2048 + 768)
#         return context, targets

#     def __len__(self):
#         return len(self.split)

class CaptionTrainContextRetrievalDataset(CaptionDataset):
    """
    PyTorch training dataset that provides batches of images with a corresponding caption each.
    """

    def __init__(self, dataset_splits_dir, features_fn, normalize=None, features_scale_factor=1):
        super().__init__(dataset_splits_dir, features_fn,
                         normalize, features_scale_factor)

        all_images_filename = os.path.join(dataset_splits_dir, TRAIN_CONTEXT_IMAGES_FILENAME) #aqui esse ficheio
        all_contexts_filename = os.path.join(dataset_splits_dir, TRAIN_CONTEXT_FILENAME) #aqui esse ficheio
        all_targets_filename = os.path.join(dataset_splits_dir, TRAIN_TARGETS_FILENAME) #aqui esse ficheio
        
        images_names = os.path.join(dataset_splits_dir, IMAGES_NAMES_FILENAME) #aqui esse ficheio
        self.images_dir="../remote-sensing-images-caption/src/data/COCO/raw_dataset/images/"


        with open(all_images_filename) as f:
            self.all_images = json.load(f)

        with open(all_contexts_filename) as f:
            self.all_contexts = json.load(f)

        with open(all_targets_filename) as f:
            self.all_targets = json.load(f)  

        # with open(images_names) as f:
        #     self.images_name = json.load(f)
        # self.enc_model = SentenceTransformer('clip-ViT-B-32')
        #self.sentence_model = SentenceTransformer('paraphrase-distilroberta-base-v1')

    def __getitem__(self, i):
        coco_id=self.all_images[i]
        image = self.get_image_features(coco_id) #32, 2048
        #image = self.enc_model.encode(Image.open(self.images_dir+self.images_name[coco_id]))
        context=self.all_contexts[i] 
        target=self.all_targets[i] 
        #gc.collect()
        return image, context, numpy.array(target)

    def __len__(self):
        return len(self.all_images)


class ContextRetrieval():

    def __init__(self, dim_examples, nlist = 10000, m = 8):
        self.dim_examples=dim_examples
        #quantizer = faiss.IndexFlatL2(dim_examples)
        #self.datastore = faiss.IndexIVFPQ(quantizer, dim_examples, nlist, m, 8)
        #self.datastore = faiss.IndexIVFFlat(quantizer, dim_examples, nlist)
        
        #sub_index = faiss.IndexIVFFlat(quantizer, dim_examples, nlist)
        #pca_matrix = faiss.PCAMatrix (dim_examples, 1024, 0, True) 
        #self.datastore = faiss.IndexPreTransform(pca_matrix, sub_index)
        #self.datastore.nprobe = 16

        self.datastore = faiss.IndexIDMap(faiss.IndexFlatL2(dim_examples))

        self.sentence_model = SentenceTransformer('paraphrase-TinyBERT-L6-v2')
        #'paraphrase-distilroberta-base-v1')
        

    def train_retrieval(self, train_dataloader_images):
        print("starting training")
        start_training=False

        max_to_fit_in_memory =4000000
        
        all_images_and_text_context=numpy.ones((max_to_fit_in_memory,self.dim_examples), dtype=numpy.float32)
        all_targets=numpy.ones((max_to_fit_in_memory), dtype=numpy.int64)
        is_to_add = False
        added_so_far=0
        #enc_model=train_dataloader_images.dataset.enc_model

        for (images, contexts, targets) in tqdm(train_dataloader_images):
            #add to the datastore
            #print("context added", targets)
            #enc_contexts=self.sentence_model.encode(contexts)
            #images_and_text_context = numpy.concatenate((images.mean(dim=1).numpy(),enc_contexts), axis=-1) #(n_contexts, 2048 + 768)
            
            # self.datastore.add(images_and_text_context)
            # 
            if start_training:
                batch_size=len(targets)
                all_images_and_text_context[added_so_far:(added_so_far+batch_size),:] = numpy.concatenate((images.mean(dim=1).numpy(),self.sentence_model.encode(contexts)), axis=-1)    
                #all_images_and_text_context[added_so_far:(added_so_far+batch_size),:] = numpy.concatenate((images,enc_model.encode(contexts)), axis=-1)    
                all_targets[added_so_far:(added_so_far+batch_size)]=targets
                added_so_far+=batch_size

                if added_so_far>=max_to_fit_in_memory:
                    print("training")
                    self.datastore.train(all_images_and_text_context)
                    self.datastore.add_with_ids(all_images_and_text_context, all_targets)
                    start_training = False
            else:
                all_images_and_text_context = numpy.concatenate((images.mean(dim=1).numpy(),self.sentence_model.encode(contexts)), axis=-1)    
                #all_images_and_text_context = numpy.concatenate((images,enc_model.encode(contexts)), axis=-1)    
                self.datastore.add_with_ids(all_images_and_text_context, numpy.array(targets, dtype=numpy.int64))
        
            gc.collect()

        faiss.write_index(self.datastore, "/media/rprstorage2/context_retrieval")
        print("n of examples", self.datastore.ntotal)

    def retrieve_nearest_for_train_query(self, query_img, k=16):
        #print("self query img", query_img)
        D, I = self.datastore.search(query_img, k)     # actual search
        nearest_input=torch.tensor(I)
        print("all nearest", torch.tensor(I))
        return nearest_input, D

    def retrieve_nearest_for_val_or_test_query(self, query_img, k=16):
        D, I = self.datastore.search(query_img, k)     # actual search
        #nearest_input = self.targets_of_dataloader[torch.tensor(I)]
        nearest_input=torch.tensor(I)

        # print("the nearest input", nearest_input)
        return nearest_input, D


class CaptionTrainContextLSTMRetrievalDataset(CaptionDataset):
    """
    PyTorch training dataset that provides batches of images with a corresponding caption each.
    """

    def __init__(self, dataset_splits_dir, features_fn, normalize=None, features_scale_factor=1, max_caption_len=50):
        super().__init__(dataset_splits_dir, features_fn,
                         normalize, features_scale_factor)

        all_images_filename = os.path.join(dataset_splits_dir, TRAIN_CONTEXT_IMAGES_FILENAME) #aqui esse ficheio
        all_contexts_filename = os.path.join(dataset_splits_dir, TRAIN_CONTEXT_FILENAME) #aqui esse ficheio
        all_targets_filename = os.path.join(dataset_splits_dir, TRAIN_TARGETS_FILENAME) #aqui esse ficheio
        
        images_names = os.path.join(dataset_splits_dir, IMAGES_NAMES_FILENAME) #aqui esse ficheio
        self.images_dir="../remote-sensing-images-caption/src/data/COCO/raw_dataset/images/"


        with open(all_images_filename) as f:
            self.all_images = json.load(f)

        with open(all_contexts_filename) as f:
            self.all_contexts = json.load(f)

        with open(all_targets_filename) as f:
            self.all_targets = json.load(f)  
        
        word_map_filename = os.path.join(dataset_splits_dir, WORD_MAP_FILENAME)
        with open(word_map_filename) as f:
            self.word_map = json.load(f)     

        self.max_caption_len =max_caption_len

        # with open(images_names) as f:
        #     self.images_name = json.load(f)
        # self.enc_model = SentenceTransformer('clip-ViT-B-32')
        #self.sentence_model = SentenceTransformer('paraphrase-distilroberta-base-v1')

    def __getitem__(self, i):
        coco_id=self.all_images[i]
        image = self.get_image_features(coco_id) #32, 2048
        #image = self.enc_model.encode(Image.open(self.images_dir+self.images_name[coco_id]))
        context=[self.word_map[w] for w in self.all_contexts[i].split()] 
        len_context =len(context)

        context = context + [self.word_map[TOKEN_PAD]] * (self.max_caption_len - len_context)
        #print("context", context)
        target=self.all_targets[i] 
        #gc.collect()
        return image, numpy.array(context), numpy.array(len_context), numpy.array(target)

    def __len__(self):
        return len(self.all_images)


class ContextLSTMRetrieval():

    def __init__(self, dim_examples, context_model, nlist = 10000, m = 8):
        self.dim_examples=dim_examples
        #quantizer = faiss.IndexFlatL2(dim_examples)
        #self.datastore = faiss.IndexIVFPQ(quantizer, dim_examples, nlist, m, 8)
        #self.datastore = faiss.IndexIVFFlat(quantizer, dim_examples, nlist)
        
        #sub_index = faiss.IndexIVFFlat(quantizer, dim_examples, nlist)
        #pca_matrix = faiss.PCAMatrix (dim_examples, 1024, 0, True) 
        #self.datastore = faiss.IndexPreTransform(pca_matrix, sub_index)
        #self.datastore.nprobe = 16

        self.datastore = faiss.IndexIDMap(faiss.IndexFlatL2(dim_examples))
        self.context_model = context_model 

    def train_retrieval(self, train_dataloader_images):
        print("starting training")
        start_training=False

        max_to_fit_in_memory =4000000
        
        all_images_and_text_context=numpy.ones((max_to_fit_in_memory,self.dim_examples), dtype=numpy.float32)
        all_targets=numpy.ones((max_to_fit_in_memory), dtype=numpy.int64)
        is_to_add = False
        added_so_far=0
        teacher_forcing = True
        
        #enc_model=train_dataloader_images.dataset.enc_model

        for (images, contexts, decode_lengths, targets) in tqdm(train_dataloader_images):
            #add to the datastore
            #print("context added", targets)
            #enc_contexts=self.sentence_model.encode(contexts)
            #images_and_text_context = numpy.concatenate((images.mean(dim=1).numpy(),enc_contexts), axis=-1) #(n_contexts, 2048 + 768)
            
            # self.datastore.add(images_and_text_context)
            # 
            if start_training:
                batch_size=len(targets)
                all_images_and_text_context[added_so_far:(added_so_far+batch_size),:] = numpy.concatenate((images.mean(dim=1).numpy(),self.sentence_model.encode(contexts)), axis=-1)    
                #all_images_and_text_context[added_so_far:(added_so_far+batch_size),:] = numpy.concatenate((images,enc_model.encode(contexts)), axis=-1)    
                all_targets[added_so_far:(added_so_far+batch_size)]=targets
                added_so_far+=batch_size

                if added_so_far>=max_to_fit_in_memory:
                    print("training")
                    self.datastore.train(all_images_and_text_context)
                    self.datastore.add_with_ids(all_images_and_text_context, all_targets)
                    start_training = False
            else:
                _, _, extras = self.context_model(images, contexts, decode_lengths, teacher_forcing)
                hidden_state=extras.get("hidden_states", None)
                self.datastore.add_with_ids(hidden_state.detach().numpy(), numpy.array(targets, dtype=numpy.int64))
        
            gc.collect()

        faiss.write_index(self.datastore, "/media/rprstorage2/context_lstm_retrieval")
        print("n of examples", self.datastore.ntotal)

    def retrieve_nearest_for_train_query(self, query_img, k=16):
        #print("self query img", query_img)
        D, I = self.datastore.search(query_img, k)     # actual search
        nearest_input=torch.tensor(I)
        print("all nearest", torch.tensor(I))
        return nearest_input, D

    def retrieve_nearest_for_val_or_test_query(self, query_img, k=16):
        D, I = self.datastore.search(query_img, k)     # actual search
        #nearest_input = self.targets_of_dataloader[torch.tensor(I)]
        nearest_input=torch.tensor(I)

        # print("the nearest input", nearest_input)
        return nearest_input, D



# class ContextRetrieval():

#     def __init__(self, dim_examples, train_dataloader_images, device):
#         #print("self dim exam", dim_examples)
#         self.datastore = faiss.IndexFlatL2(dim_examples) #datastore

#         #data
#         self.device=device
#         self.targets_of_dataloader = torch.tensor([]).long().to(device)
#         #print("self.imgs_indexes_of_dataloader type", self.imgs_indexes_of_dataloader)

#         #print("len img dataloader", self.imgs_indexes_of_dataloader.size())
#         self._add_examples(train_dataloader_images)
#         #print("len img dataloader final", self.imgs_indexes_of_dataloader.size())
#         #print("como ficou img dataloader final", self.imgs_indexes_of_dataloader)


#     def _add_examples(self, train_dataloader_images):
#         print("\nadding input examples to datastore (retrieval)")
#         for i, (encoder_text_outputs, targets) in enumerate(train_dataloader_images):
#             #add to the datastore
#             print("enc tex", encoder_text_outputs.squeeze(0))
#             print("enc tex", encoder_text_outputs.squeeze(0).numpy().astype(dtype=numpy.float32, copy=False))

#             self.datastore.add(encoder_text_outputs.squeeze(0).numpy().astype(dtype=numpy.float32, copy=False))
#             targets = torch.tensor(targets).to(self.device)
#             self.targets_of_dataloader= torch.cat((self.targets_of_dataloader,targets))

#             if i%5==0:
#                 print("i and img index of ImageRetrival", i, self.targets_of_dataloader)
#                 print("n of examples", self.datastore.ntotal)
    
#     def retrieve_nearest_for_train_query(self, query_img, k=2):
#         #print("self query img", query_img)
#         D, I = self.datastore.search(query_img, k)     # actual search
#         # print("all nearest", I)
#         # print("I firt", I[:,0])
#         # print("I second", I[:,1])

#         # print("if you choose the first", self.imgs_indexes_of_dataloader[I[:,0]])
#         # print("this is the img indexes", self.imgs_indexes_of_dataloader)
#         # print("n of img index", len(self.imgs_indexes_of_dataloader))
#         # print("n of examples", self.datastore.ntotal)

#         nearest_input = self.targets_of_dataloader[I[:,1]]
#         #print("the nearest input is actual the second for training", nearest_input)
#         return nearest_input

#     def retrieve_nearest_for_val_or_test_query(self, query_img, k=1):
#         D, I = self.datastore.search(query_img, k)     # actual search
#         nearest_input = self.targets_of_dataloader[I[:,0]]
#         # print("all nearest", I)
#         # print("the nearest input", nearest_input)
#         return nearest_input


class CaptionTrainRetrievalDataset(CaptionDataset):

    def __init__(self, dataset_splits_dir, features_fn, normalize=None, features_scale_factor=1):
        super().__init__(dataset_splits_dir, features_fn,
                         normalize, features_scale_factor)
        self.split = self.split[TRAIN_SPLIT]
        #self.split = self.split[VALID_SPLIT]
        #TODO: REMOVE VALID SPLIT JUST FOR CHECKING


    def __getitem__(self, i):
        # Convert index depending on the dataset split
        coco_id = self.split[i]

        image = self.get_image_features(coco_id)
        return image, coco_id

    def __len__(self):
        return len(self.split)


class ImageRetrieval():

    def __init__(self, dim_examples, train_dataloader_images, device):
        #print("self dim exam", dim_examples)
        self.datastore = faiss.IndexFlatL2(dim_examples) #datastore

        #data
        self.device=device
        self.imgs_indexes_of_dataloader = torch.tensor([]).long().to(device)
        #print("self.imgs_indexes_of_dataloader type", self.imgs_indexes_of_dataloader)

        #print("len img dataloader", self.imgs_indexes_of_dataloader.size())
        self._add_examples(train_dataloader_images)
        #print("len img dataloader final", self.imgs_indexes_of_dataloader.size())
        #print("como ficou img dataloader final", self.imgs_indexes_of_dataloader)


    def _add_examples(self, train_dataloader_images):
        print("\nadding input examples to datastore (retrieval)")
        for i, (encoder_output, imgs_indexes) in enumerate(train_dataloader_images):
            #add to the datastore
            encoder_output=encoder_output.to(self.device)
            imgs_indexes = torch.tensor(list(map(int, imgs_indexes))).to(self.device)
            #print("\nimages ind", imgs_indexes)
            #print("img index type", imgs_indexes)
            #print("encoder out", encoder_output.size())
            #encoder_output = encoder_output.view(encoder_output.size()[0], -1, encoder_output.size()[-1])
            #print("encoder out", encoder_output.size())
            input_img = encoder_output.mean(dim=1)
            #print("input image size", input_img.size())
            
            self.datastore.add(input_img.cpu().numpy())
            self.imgs_indexes_of_dataloader= torch.cat((self.imgs_indexes_of_dataloader,imgs_indexes))

            if i%5==0:
                print("i and img index of ImageRetrival",i, imgs_indexes)
                print("n of examples", self.datastore.ntotal)
    
    def retrieve_nearest_for_train_query(self, query_img, k=2):
        #print("self query img", query_img)
        D, I = self.datastore.search(query_img, k)     # actual search
        # print("all nearest", I)
        # print("I firt", I[:,0])
        # print("I second", I[:,1])

        # print("if you choose the first", self.imgs_indexes_of_dataloader[I[:,0]])
        # print("this is the img indexes", self.imgs_indexes_of_dataloader)
        # print("n of img index", len(self.imgs_indexes_of_dataloader))
        # print("n of examples", self.datastore.ntotal)

        nearest_input = self.imgs_indexes_of_dataloader[I[:,1]]
        #print("the nearest input is actual the second for training", nearest_input)
        return nearest_input

    def retrieve_nearest_for_val_or_test_query(self, query_img, k=1):
        D, I = self.datastore.search(query_img, k)     # actual search
        nearest_input = self.imgs_indexes_of_dataloader[I[:,0]]
        # print("all nearest", I)
        # print("the nearest input", nearest_input)
        return nearest_input


class Scale(object):
  """Scale transform with list as size params"""

  def __init__(self, size): # interpolation=Image.BILINEAR):
    self.size = size

  def __call__(self, img):
    return imresize(img.numpy().transpose(1,2,0), (224,224))


def get_data_loader(split, batch_size, dataset_splits_dir, image_features_fn, workers, image_normalize=None):

    if not image_normalize:
        normalize = None
        features_scale_factor = 1
    if image_normalize == "imagenet":
        normalize = transforms.Normalize(mean=IMAGENET_IMAGES_MEAN, std=IMAGENET_IMAGES_STD)
        normalize = transforms.Compose([normalize])
        features_scale_factor = 1/255.0
    if image_normalize == "scaleimagenet":
        normalize = transforms.Normalize(mean=IMAGENET_IMAGES_MEAN, std=IMAGENET_IMAGES_STD)
        normalize = transforms.Compose([Scale([224,224]), transforms.ToTensor(), normalize])
        features_scale_factor = 1

    if split == "train":
        data_loader = torch.utils.data.DataLoader(
            CaptionTrainDataset(dataset_splits_dir, image_features_fn, normalize, features_scale_factor),
            batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True
        )
    elif split in {"val", "test"}:
        data_loader = torch.utils.data.DataLoader(
            CaptionEvalDataset(dataset_splits_dir, image_features_fn, normalize, features_scale_factor, split),
            batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True
        )
    elif split == "retrieval":
        data_loader = torch.utils.data.DataLoader(
                CaptionTrainRetrievalDataset(dataset_splits_dir, image_features_fn, normalize, features_scale_factor),
                batch_size=100, shuffle=False, num_workers=workers, pin_memory=True
            )

    elif split == "context_retrieval":
        data_loader = torch.utils.data.DataLoader(
                CaptionTrainContextRetrievalDataset(dataset_splits_dir, image_features_fn, normalize, features_scale_factor),
                batch_size=5000, shuffle=True, num_workers=0, pin_memory=False
            )

    elif split == "context_retrieval_lstm":
        data_loader = torch.utils.data.DataLoader(
                CaptionTrainContextLSTMRetrievalDataset(dataset_splits_dir, image_features_fn, normalize, features_scale_factor),
                batch_size=5000, shuffle=True, num_workers=0, pin_memory=False
            )

    else:
        raise ValueError("Invalid data_loader split. Options: train, val, test")

    return data_loader


def get_retrieval(retrieval_data_loader, device):

    encoder_output_dim = 2048 #faster r-cnn features

    image_retrieval = ImageRetrieval(encoder_output_dim, retrieval_data_loader,device)
  
    
    # for i, (encoder_output, coco_ids) in enumerate(retrieval_data_loader):
    #     print("this is the first batch of images", encoder_output.size())
    #     print("coco ids", coco_ids)
    #     input_imgs = encoder_output.mean(dim=1)
    #     print("this  input_imgs suze after", input_imgs.size())
    #     nearest=image_retrieval.retrieve_nearest_for_train_query(input_imgs.numpy())
    #     print("this is nearest images", nearest)

    #     nearest = image_retrieval.retrieve_nearest_for_val_or_test_query(input_imgs.numpy())
        
    #     print("retrieve for test query", nearest)

        #print(stop)
    #     encoder_output= encoder_output.view(encoder_output.size()[0], -1, encoder_output.size()[-1])
    #     input_imgs = encoder_output.mean(dim=1)
    #     nearest=retrieval.retrieve_nearest_for_train_query(input_imgs.numpy())
    #     print("this is nearest images", nearest)
    #     #falta imprimir o coco id dos nearest 

    #print("stop remove from dataloader o VAL e coloca TRAIN", stop)


    return image_retrieval


def get_context_retrieval(create, retrieval_data_loader=None):

    encoder_output_dim = 2048+768 #faster r-cnn features
    image_retrieval = ContextRetrieval(encoder_output_dim)

    if create:
        image_retrieval.train_retrieval(retrieval_data_loader)
        #image_retrieval.add_vectors(retrieval_data_loader)
        # Como está mas adiciona o index map...
        # Ve se funciona chamar o eval
        # Depois tenta usar o tal Index
        #Tenta com um dataset mais pequeno e vê se funciona 
        print(stop)
    else:
        image_retrieval.datastore = faiss.read_index("/media/rprstorage2/context_retrieval")

    for i, (images, contexts, targets) in enumerate(retrieval_data_loader):
        print("targt", targets)
        images = images.mean(dim=1).numpy()
        enc_contexts=image_retrieval.sentence_model.encode(contexts)
        images_and_text_context = numpy.concatenate((images,enc_contexts), axis=-1) #(n_contexts, 2048 + 768)
          
        #nearest_targets, distances=image_retrieval.retrieve_nearest_for_train_query(images_and_text_context)
        nearest_targets, distances=image_retrieval.retrieve_nearest_for_train_query(images_and_text_context)

        print("this is nearest train images", nearest_targets)#, distances)

        print("targt", targets)
        softmax = nn.Softmax()

        softmax_nearest = torch.zeros(nearest_targets.size()[0], 16,10004)
        nearest_probs = softmax(-1.*torch.tensor(distances))
        print("nearest_probs",nearest_probs)
        ind=torch.arange(0, 16).expand(softmax_nearest.size(0), -1)
        ind_batch=torch.arange(0, nearest_targets.size()[0]).reshape(-1,1)
        softmax_nearest[ind_batch, ind,nearest_targets] = nearest_probs
        softmax_nearest = softmax_nearest.sum(1)
        print("topk", softmax_nearest.topk(4,largest=True, sorted=True))
        print("softmax_nearest argmax 0", len(softmax_nearest.argmax(dim=0)))
        print("softmax_nearest argmax 1", len(softmax_nearest.argmax(dim=1)))
        print("softmax_nearest max", softmax_nearest.max(dim=1))

        # image_retrieval.datastore.nprobe= 50
        # nearest_targets, distances=image_retrieval.retrieve_nearest_for_train_query(images_and_text_context)
        # print("this is nearest train images", nearest_targets, distances)
        # print("targt", targets)

        # image_retrieval.datastore.nprobe= 100
        # nearest_targets, distances=image_retrieval.retrieve_nearest_for_train_query(images_and_text_context)
        # print("this is nearest train images", nearest_targets, distances)
        # print("targt", targets)

        # nearest_targets, distances = image_retrieval.retrieve_nearest_for_val_or_test_query(images_and_text_context)
        # print("retrieve for test query", nearest_targets, distances)
        print(stop)

    # print("stop remove from dataloader o VAL e coloca TRAIN", stop)

    #faiss.write_index(image_retrieval.datastore, "/media/jlsstorage/rita/context_retrieval")

    return image_retrieval


def get_context_lstm_retrieval(create, context_model, retrieval_data_loader=None):

    encoder_output_dim = 1000 #faster r-cnn features
    image_retrieval = ContextLSTMRetrieval(encoder_output_dim, context_model)

    if create:
        image_retrieval.train_retrieval(retrieval_data_loader)
        #image_retrieval.add_vectors(retrieval_data_loader)
        # Como está mas adiciona o index map...
        # Ve se funciona chamar o eval
        # Depois tenta usar o tal Index
        #Tenta com um dataset mais pequeno e vê se funciona 
        print(stop)
    else:
        image_retrieval.datastore = faiss.read_index("/media/rprstorage2/context_lstm_retrieval")

    for i, (images, contexts, targets) in enumerate(retrieval_data_loader):
        print("targt", targets)
        images = images.mean(dim=1).numpy()
        enc_contexts=image_retrieval.sentence_model.encode(contexts)
        #images_and_text_context = numpy.concatenate((images,enc_contexts), axis=-1) #(n_contexts, 2048 + 768)
          
        #nearest_targets, distances=image_retrieval.retrieve_nearest_for_train_query(images_and_text_context)
        nearest_targets, distances=image_retrieval.retrieve_nearest_for_train_query(enc_contexts)

        print("this is nearest train images", nearest_targets, distances)

        print("targt", targets)

        # image_retrieval.datastore.nprobe= 50
        # nearest_targets, distances=image_retrieval.retrieve_nearest_for_train_query(images_and_text_context)
        # print("this is nearest train images", nearest_targets, distances)
        # print("targt", targets)

        # image_retrieval.datastore.nprobe= 100
        # nearest_targets, distances=image_retrieval.retrieve_nearest_for_train_query(images_and_text_context)
        # print("this is nearest train images", nearest_targets, distances)
        # print("targt", targets)


        # nearest_targets, distances = image_retrieval.retrieve_nearest_for_val_or_test_query(images_and_text_context)
        
        # print("retrieve for test query", nearest_targets, distances)
        print(stop)

    # print("stop remove from dataloader o VAL e coloca TRAIN", stop)

    #faiss.write_index(image_retrieval.datastore, "/media/jlsstorage/rita/context_retrieval")

    return image_retrieval