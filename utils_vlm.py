import torch
import numpy as np
from qwen_vl_utils import process_vision_info


## convert training sample into dialog format
def format_data(sample, system_message):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["image"],
                },
                {
                    "type": "text",
                    "text": sample['query'],
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": sample["label"][0]
                }
            ],
        },
    ]


def generate_text_from_sample(model, processor, sample, device, max_new_tokens=1024):
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        sample[0:2],  
        tokenize=False,
        add_generation_prompt=True)
    # Process the visual input from the sample
    ## 返回sample中的image_inputs和video_inputs
    image_inputs, _ = process_vision_info(sample)
    # Prepare the inputs for the model
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(device)  # Move inputs to the specified device
    # Generate text with the model
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    # Decode the output text
    output_text = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return output_text[0]  # Return the first decoded output text


def generate_actions_from_sample(model, processor, action_tokenizers, sample, device, max_new_tokens=1024):
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        sample[0:2], 
        tokenize=False,
        add_generation_prompt=True)
    
    # Process the visual input from the sample
    image_inputs, _ = process_vision_info(sample)

    # Prepare the inputs for the model
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(device)  # Move inputs to the specified device

    # Generate text with the model
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    # print(generated_ids)

    # trim the generated ids to remove the input ids
    trimmed_generated_ids = [
        out_ids[len(in_ids):].cpu() for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # get the last action_dim tokens
    action_dim = len(action_tokenizers)
    action_tokens_ids = trimmed_generated_ids[0][-action_dim:]

    # convert the last action_dim tokens to actions
    #print(action_tokens_ids)
    actions = [action_tokenizers[i].detokenize(action_tokens_ids[i]) for i in range(action_dim)]
    
    return actions

    
def find_assistant_content_sublist_indexes(l):
    '''
    A message from train_data/data.json may look like below:
        {
            "messages": [
                {'role': 'user', 'content': [{'type': 'image', 'image': 'train_data/1.jpeg'}, {'type': 'text', 'text': '描述一下这个图片'}]}, 
                {'role': 'assistant', 'content': [{'type': 'text', 'text': '这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。'}]}
            ]
        }
    After apply_chat_template, the text will look like below:
        ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>描述一下这个图片<|im_end|>\n<|im_start|>assistant\n这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。<|im_end|>\n']

    This function tries to find the indexes of the assistant content in the input_ids list to build labels.
    '''
    # (Pdb++) processor.tokenizer.encode("<|im_start|>assistant\n")
    # [151644, 77091, 198]
    # (Pdb++) processor.tokenizer.encode("<|im_end|>\n")
    # [151645, 198]

    start_indexes = []
    end_indexes = []

    # Iterate through the list to find starting points
    for i in range(len(l) - 2):
        # Check if the current and next elements form the start sequence
        if l[i] == 151644 and l[i+1] == 77091 and l[i+2] == 198:
            start_indexes.append(i+3)
            # Now look for the first 151645 and 198 after the start
            for j in range(i+3, len(l)-1):
                if l[j] == 151645 and l[j+1] == 198:
                    end_indexes.append(j+2) # **NOTE** the <|im_end|>\n 2 tokens should be included in the label, so that model can predicate end of output.
                    break  # Move to the next start after finding the end

    return list(zip(start_indexes, end_indexes))

# Create a data collator to encode text and image pairs
def collate_fn(examples, processor):
    # Get the texts and images, and apply the chat template
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]  # Prepare texts for processing
    image_inputs = [process_vision_info(example)[0] for example in examples]  # Process the images to extract inputs

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)  # Encode texts and images into tensors

    input_ids_list = batch["input_ids"].tolist()
    labels_list = []
    for ids_list in input_ids_list:
        label_ids = [-100] * len(ids_list)
        for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
            label_ids[begin_end_indexs[0]:begin_end_indexs[1]] = ids_list[begin_end_indexs[0]:begin_end_indexs[1]]
        labels_list.append(label_ids)

    batch["labels"] = torch.tensor(labels_list, dtype=torch.int64)  # Add labels to the batch
    return batch  # Return the prepared batch


class ActionTokenizer:
    def __init__(self, bins, tokenizer, min_action, max_action):
        self.n_bins = bins
        self.tokenizer = tokenizer
        self.min_action = min_action
        self.max_action = max_action

        self.bins = np.linspace(min_action, max_action, bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2

        self.vocab_size = len(self.tokenizer)

    def tokenize(self, action):
        ## min -> max, 1 -> 256
        action = np.clip(action, self.min_action, self.max_action)
        discretized_action = np.digitize(action, self.bins)
        return self.tokenizer.batch_decode((self.vocab_size - discretized_action).tolist())
    
    def detokenize(self, action_tokens_ids):
        ## 255 and 256 are mapped to the same action
        discretized_actions = self.vocab_size - np.array(action_tokens_ids)
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        return self.bin_centers[discretized_actions]

        
