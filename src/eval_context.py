"""Evaluate an image captioning model on the specified evaluation set using the specified set of evaluation metrics"""

import sys
import json
import os.path
import logging
import argparse
from tqdm import tqdm
from torch import nn
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn

from toolkit.data.datasets import get_data_loader, get_retrieval, get_context_retrieval
from toolkit.util.analysis.visualize_attention import visualize_attention
from toolkit.common.sequence_generator import beam_search, beam_search_context, beam_re_ranking, nucleus_sampling, get_retrieved_caption
from toolkit.utils import rm_caption_special_tokens, MODEL_SHOW_ATTEND_TELL, MODEL_BOTTOM_UP_TOP_DOWN_RETRIEVAL,MODEL_BOTTOM_UP_TOP_DOWN_CONTEXT, get_log_file_path, decode_caption
from train import build_model, abbr2name
from options import add_model_args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # improve performance if inputs to model are fixed size


def evaluate(image_features_fn, dataset_splits_dir, split, checkpoint_path, output_path,
             max_caption_len, beam_size, eval_beam_size, re_ranking, keep_special_tokens, nucleus_sampling_size,
             visualize, print_beam, print_captions, eval_retrieved=False, args=None):
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_name = abbr2name[args.model]
    logging.info("Model: {}".format(model_name))

    print("model name", model_name)
    model = build_model(args, model_name)    
    print("model", model)
    if torch.cuda.device_count() > 1:
        print("data parallel")
        model = nn.DataParallel(model)
    model = model.to(device)

    print("after device")

    model.decoder.load_state_dict(checkpoint["model"].decoder.state_dict())
    #model.encoder.load_state_dict(checkpoint["model"].encoder.state_dict())
    
    model.eval()
    word_map = model.decoder.word_map
    logging.info("Model params: {}".format(vars(model)))

    # DataLoader
    image_normalize = None
    if model_name == MODEL_SHOW_ATTEND_TELL:
        image_normalize = "imagenet"

    if model_name == MODEL_BOTTOM_UP_TOP_DOWN_RETRIEVAL:
        train_retrieval_loader = get_data_loader("retrieval", 100, dataset_splits_dir, image_features_fn,
                                        1, image_normalize)
        target_lookup= train_retrieval_loader.dataset.image_metas
        image_retrieval = get_retrieval(train_retrieval_loader, device)
    elif model_name == MODEL_BOTTOM_UP_TOP_DOWN_CONTEXT:
        print("loading datastore")
        train_context_retrieval_loader = get_data_loader("context_retrieval", 100, args.dataset_splits_dir, args.image_features_filename,
                                        0, args.image_normalize)
 
        image_retrieval = get_context_retrieval(train_context_retrieval_loader, device)
        target_lookup= image_retrieval.targets_of_dataloader
        
        #preencher isto e depois
    else:
        target_lookup=None
        image_retrieval=None

    if eval_retrieved:
        train_retrieval_loader = get_data_loader("retrieval", 100, dataset_splits_dir, image_features_fn,
                                1, image_normalize)
        target_lookup= train_retrieval_loader.dataset.image_metas
        image_retrieval = get_retrieval(train_retrieval_loader, device)

    data_loader = get_data_loader(split, 1, dataset_splits_dir, image_features_fn, 1, image_normalize)

    if keep_special_tokens:
        rm_special_tokens = lambda x, y: x 
    else:
        rm_special_tokens = rm_caption_special_tokens

    # Lists for target captions and generated captions for each image
    target_captions = {}
    generated_captions = {}
    generated_beams = {}

    for image_features, all_captions_for_image, caption_lengths, coco_id in tqdm(
            data_loader, desc="Evaluate with beam size " + str(beam_size)):
        coco_id = coco_id[0]

        # Target captions
        target_captions[coco_id] = [rm_special_tokens(caption, word_map)
                                    for caption in all_captions_for_image[0].tolist()]

        # Generate captions
        image_features = image_features.to(device)
        store_beam = True  # if METRIC_BEAM_OCCURRENCES in metrics else False

        if nucleus_sampling_size:
            top_k_generated_captions, alphas, beam = nucleus_sampling(
                model, image_features, beam_size,
                top_p=nucleus_sampling_size,
                print_beam=print_beam,
            )
        else:
            if eval_retrieved:
                top_k_generated_captions, alphas, beam = get_retrieved_caption(
                    image_features, 
                    image_retrieval=image_retrieval,
                    target_lookup=target_lookup
                )

            elif model_name == MODEL_BOTTOM_UP_TOP_DOWN_CONTEXT:
                 top_k_generated_captions, alphas, beam = beam_search_context(
                    model, image_features, beam_size,
                    max_caption_len=max_caption_len,
                    store_alphas=visualize,
                    store_beam=store_beam,
                    print_beam=print_beam,
                    image_retrieval=image_retrieval,
                    target_lookup=target_lookup
                )
                
            else:
                top_k_generated_captions, alphas, beam = beam_search(
                    model, image_features, beam_size,
                    max_caption_len=max_caption_len,
                    store_alphas=visualize,
                    store_beam=store_beam,
                    print_beam=print_beam,
                    image_retrieval=image_retrieval,
                    target_lookup=target_lookup
                )

        if visualize:
            logging.info("Image COCO ID: {}".format(coco_id))
            for caption, alpha in zip(top_k_generated_captions, alphas):
                visualize_attention(image_features.squeeze(0), caption, alpha, word_map, smoothen=True)

        if re_ranking:
            if print_captions:
                logging.info("COCO ID: {}".format(coco_id))
                logging.info("Before re-ranking:")
                for caption in top_k_generated_captions[:eval_beam_size]:
                    logging.info(decode_caption(rm_special_tokens(caption, word_map), word_map))
            top_k_generated_captions = beam_re_ranking(model, image_features, top_k_generated_captions, word_map)

        generated_captions[coco_id] = top_k_generated_captions[:eval_beam_size]
        if print_captions:
            logging.info("COCO ID: {}".format(coco_id))
            for caption in generated_captions[coco_id]:
                logging.info(decode_caption(rm_special_tokens(caption, word_map), word_map))
        if store_beam:
            generated_beams[coco_id] = beam

        assert len(target_captions) == len(generated_captions)

    # Save results
    name = split
    if re_ranking:
        name += ".re_ranking"
    if nucleus_sampling_size:
        name += ".nucleus_" + str(nucleus_sampling_size)
    else:
        name += ".beam_" + str(beam_size)
    if keep_special_tokens:
        name += ".tagged"
    outputs_dir = os.path.join(output_path, "outputs")
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    # Save results file with top caption for each image :
    # (JSON file of image -> caption output by the model)
    results = []
    if eval_retrieved:
        #cena
        #print("vit model")
        for coco_id, top_k_captions in generated_captions.items():
            caption = decode_caption(rm_special_tokens(top_k_captions[0], word_map), word_map)
            results.append({"image_id": int(coco_id), "caption": caption})
            print("cap", caption)
    else:  
        for coco_id, top_k_captions in generated_captions.items():
            caption = decode_caption(rm_special_tokens(top_k_captions[0], word_map), word_map)
            results.append({"image_id": int(coco_id), "caption": caption})
    results_output_file_name = os.path.join(outputs_dir, name + ".json")
    json.dump(results, open(results_output_file_name, "w"))

    # Save results file with all generated captions for each image:
    # JSON file of image -> top-k captions output by the model. Used for recall.
    results = []
    if eval_retrieved:
        #cena
        #print("vit model")
        for coco_id, top_k_captions in generated_captions.items():
            caption = decode_caption(rm_special_tokens(top_k_captions[0], word_map), word_map)
            results.append({"image_id": int(coco_id), "caption": caption})
    else:
        for coco_id, top_k_captions in generated_captions.items():
            captions = [decode_caption(rm_special_tokens(capt, word_map), word_map) for capt in top_k_captions]
            results.append({"image_id": int(coco_id), "captions": captions})
    results_output_file_name = os.path.join(outputs_dir, name + ".top_%d" % eval_beam_size + ".json")
    json.dump(results, open(results_output_file_name, "w"))

    # Save target captions:
    # JSON file of image -> ground-truth captions.
    results = []
    for coco_id, all_captions_for_image in target_captions.items():
        captions = [decode_caption(caption, word_map) for caption in all_captions_for_image]
        results.append({"image_id": int(coco_id), "captions": captions})
    results_output_file_name = os.path.join(outputs_dir, split + ".targets.json")
    json.dump(results, open(results_output_file_name, "w"))


def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-features-filename",
                        help="Folder where the preprocessed image data is located")
    parser.add_argument("--dataset-splits-dir",
                        help="Pickled file containing the dataset splits")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint of trained model")
    parser.add_argument("--logging-dir")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--output-path",
                        help="Folder where to store outputs")
    parser.add_argument("--max-caption-len", type=int, default=20)
    parser.add_argument("--beam-size", type=int, default=5,
                        help="Size of the decoding beam")
    parser.add_argument("--eval-beam-size", type=int, default=5,
                        help="Number of sequences from the beam that should be used for evaluation")
    parser.add_argument("--re-ranking", default=False, action="store_true",
                        help="Use re-ranking to sort the beam")
    parser.add_argument("--keep-special-tokens", default=False, action="store_true",
                        help="Keep special tokens in captions")
    parser.add_argument("--nucleus-sampling-size", type=float,
                        help="Use nucleus sampling with the given p instead of beam search")
    parser.add_argument("--visualize-attention", default=False, action="store_true",
                        help="Visualize the attention for every sample")
    parser.add_argument("--print-beam", default=False, action="store_true",
                        help="Print the decoding beam for every sample")
    parser.add_argument("--print-captions", default=False, action="store_true",
                        help="Print the generated captions for every sample")
    parser.add_argument("--eval_retrieved", default=False, action="store_true",
                        help="Eval retrieved caps")

    add_model_args(parser)


    parsed_args = parser.parse_args(args)
    return parsed_args


if __name__ == "__main__":
    args = check_args(sys.argv[1:])
    #logging.basicConfig(filename=get_log_file_path(args.logging_dir, args.split), level=logging.INFO)
    logging.basicConfig(
            format='%(levelname)s: %(message)s', level=logging.INFO)

    logging.info(args)
    evaluate(
        image_features_fn=args.image_features_filename,
        dataset_splits_dir=args.dataset_splits_dir,
        split=args.split,
        checkpoint_path=args.checkpoint,
        output_path=args.output_path,
        max_caption_len=args.max_caption_len,
        beam_size=args.beam_size,
        eval_beam_size=args.eval_beam_size,
        re_ranking=args.re_ranking,
        keep_special_tokens=args.keep_special_tokens,
        nucleus_sampling_size=args.nucleus_sampling_size,
        visualize=args.visualize_attention,
        print_beam=args.print_beam,
        print_captions=args.print_captions,
        eval_retrieved = args.eval_retrieved,
        args=args
    )
