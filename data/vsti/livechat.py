import json, torch, tqdm, random

from .vstibench import VSTIBench
from ..stream import StreamMixIn
from ..utils import ceil_time_by_fps, floor_time_by_fps, rand_bool, DictWithTo
from transformers import PreTrainedTokenizer, EvalPrediction
import Levenshtein
import numpy as np

class VSTIBenchLiveChat(VSTIBench, StreamMixIn):
    evaluation_kwargs = DictWithTo(evaluator='generate_after_embed', max_new_tokens=512, do_sample=False, use_cache=True, temperature=1.0, top_p=1.0)


    def __init__(self, *, split: str, frame_fps: int, is_training: bool, **kwargs):
        super().__init__(split=split, frame_fps=frame_fps, is_training=is_training, **kwargs)
        self.is_training = is_training
        self.frame_fps = frame_fps
        anno_path = f'datasets/vlm3r_ours/my_qa/{split}/combined.json'
        annos = json.load(open(anno_path))
        self.annos = []
        self.labels = []
        for anno in tqdm.tqdm(annos):
            video_uid = anno['video_uid']
            duration = self.metadata[video_uid]['duration']
            if not anno['conversation']:
                continue
            role = anno['conversation'][0]['role']
            time = anno['conversation'][0]['time']
            content = anno['conversation'][0]['content']
            
            if not (role == 'user' and time > 0 and time <= duration and content):
                continue
            # 1. add random frames before the user
            fps_time = floor_time_by_fps(time, frame_fps, 0, duration)
            waiting_frames = random.randint(0, min(20, int(fps_time * frame_fps)))
            conversation = []
            if waiting_frames:
                conversation.append({'role': 'stream', 'num_frames': waiting_frames, 'learn': waiting_frames - 1})
            conversation.append({'role': 'user', 'content': content, 'time': time, 'fps_time': fps_time})
            start_fps_time = fps_time - (waiting_frames - 1) / frame_fps
            # 2. for loop to add message

            for message in anno['conversation'][1:]:
                role, content, time = message['role'], message['content'], message['time']
                if time > duration:
                    break
                if time < conversation[-1]['time']:
                    break
                if time == conversation[-1]['time']:
                    if role == 'user':
                        break
                    else:
                        if conversation[-1]['role'] == 'user':
                            conversation.append({'role': 'assistant', 'content': content, 'time': time, 'fps_time': conversation[-1]['fps_time'], 'learn': True})
                            self.labels.append(content)
                        else:
                            conversation[-1]['content'] = content
                            self.labels.append(content)
                        continue
                if role == 'user':
                    fps_time = floor_time_by_fps(time, frame_fps, conversation[-1]['fps_time'], duration)
                    if fps_time > duration:
                        break
                    if fps_time > conversation[-1]['fps_time']:
                        conversation.append({'role': 'stream', 'num_frames': int((fps_time - conversation[-1]['fps_time']) * frame_fps), 'learn': True})
                    conversation.append({'role': 'user', 'content': content, 'time': time, 'fps_time': fps_time})
                else:
                    fps_time = ceil_time_by_fps(time, frame_fps, conversation[-1]['fps_time'], duration)
                    if fps_time > duration:
                        break
                    if fps_time > conversation[-1]['fps_time']:
                        # breakpoint()
                        conversation.append({'role': 'stream', 'num_frames': int((fps_time - conversation[-1]['fps_time']) * frame_fps), 'learn': True})
                        conversation.append({'role': 'assistant', 'content': content, 'time': time, 'fps_time': fps_time, 'learn': True})
                    self.labels.append(content)
            if not conversation:
                continue
            
            
            # # --- prefilter: skip samples whose video tokens would overflow the 8192-token context ---
            # load_range_frames = int(conversation[-1]['fps_time'] * frame_fps) + 1 - int(start_fps_time * frame_fps)
            # effective_frames = min(load_range_frames, self.max_num_frames)
            # # v1+: 10 <v> tokens/frame + 1 ',' separator between frames = 11 tokens/frame
            # # Add 300-token budget for system prompt, user text, and delimiters
            # if effective_frames * 11 + 300 > 8192:
            #     continue
            
            

            self.annos.append({
                'conversation': conversation,
                'load_ranges': {self.metadata[video_uid]['path']: range(int(start_fps_time*frame_fps), int(conversation[-1]['fps_time']*frame_fps)+1)}
            })
        self.labels = np.array(self.labels)
        
        
        # Only keep sample at index 876 that has bug
        
        # self.annos = [self.annos[876]]
        # self.labels = self.labels[876:877]

      

        
        # filtered_annos = []
        # filtered_labels = []
        # for i, (anno, label) in enumerate(zip(self.annos, self.labels)):
        #     total_frames = sum(v.stop - v.start for v in anno['load_ranges'].values())
        #     frame_tokens = total_frames * 10
        #     if frame_tokens + 4500 > 8192:  # 4500 is roughly what you saw for text
        #         breakpoint()
        #         print(f"[FILTER] idx={i}, frames={total_frames}, frame_tokens={frame_tokens}, skipping")
        #         continue
        #     filtered_annos.append(anno)
        #     filtered_labels.append(label)

        # print(f"Filtered {len(self.annos) - len(filtered_annos)}/{len(self.annos)} oversized samples")
        # self.annos = filtered_annos
        # self.labels = np.array(filtered_labels)




    def __getitem__(self, index):
        anno = self.annos[index]
        
        conversation = anno['conversation'][:-1] if not self.is_training else anno['conversation'] # if not training, do not include the assistant message
        # return *super().__getitem__(conversation=conversation, load_ranges=anno['load_ranges'], add_generation_prompt=not self.is_training), index, self.evaluation_kwargs
        
        result = *super().__getitem__(conversation=conversation, load_ranges=anno['load_ranges'], add_generation_prompt=not self.is_training), index, self.evaluation_kwargs
    
        # frames = result[1]  # shape: [num_frames, 10, 1024]
        # text = result[0]    # string
        
        # # Rough estimate: ~4 chars per token for English text
        # est_text_tokens = len(text) // 3  # conservative estimate
        # frame_tokens = frames.shape[0] * 10
        # total = est_text_tokens + frame_tokens
        
        # if total > 8192 - 128:
        #     print(f"[SKIP] idx={index}, text_chars={len(text)}, est_text_tokens={est_text_tokens}, frame_tokens={frame_tokens}, total={total}")
        #     return self.__getitem__((index + 1) % len(self))
    
        return result

    
    
    def fuzzy_match(text, choices):
        return min([(Levenshtein.distance(text, choice), choice) for choice in choices])[1]

    
    def compute_metrics(self, eval_predictions: EvalPrediction, tokenizer: PreTrainedTokenizer, **kwargs):
        batch_pred_tensor, sample_idxs = eval_predictions.predictions, eval_predictions.label_ids
        batch_pred_tensor[batch_pred_tensor < 0] = tokenizer.bos_token_id # not use clamp(min=0), since 0 is ! in Llama-3 tokenizer and may affect matching
        tmp = batch_pred_tensor[0][batch_pred_tensor[0] != -100]
        predictions = tokenizer.batch_decode(batch_pred_tensor, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        correct = 0

        for prediction, label in zip(predictions, self.labels[sample_idxs]): # should be self.labels[sample_idx] to get the correct order
            prediction = prediction.lower().rstrip('.')
            # breakpoint()
            # if prediction == label or self.fuzzy_match(prediction, self.categories) == label:
            #     correct += 1
            if float(prediction) - float(label) < 0.1:
                correct += 1
        return dict(accuracy=correct / len(predictions) * 100) # * 100

    def preprocess_conversation(self, conversation):
        if self.augmentation and self.is_training and len(conversation) >= 4: # 2 round
            i = random.randint(0, len(conversation) - 1) # stream, assistant, stream, ...
            if i > len(conversation) - 3:
                return [random.choice(self.user_instructions)] + conversation
            if conversation[i]['role'] == 'stream':
                i += 1 # assistant
            assert conversation[i]['role'] == 'assistant'
            correct_assistant = conversation[i]
            wrong_texts = set([turn['content'] for turn in conversation if 'assistant' == turn['role']]) - set(correct_assistant['content'])
            wrong_texts = list(wrong_texts) + ['']
            wrong_assistant = {'role': 'assistant', 'content': random.choice(wrong_texts)}
            augmented = [wrong_assistant]
            num_next_frames = conversation[i+1]['intervals'].numel()
            if num_next_frames > 1:
                if rand_bool(): # promptly fix behavior
                    frame_placeholder_with_interval = self.v_placeholders_per_frame + self.frame_interval
                    next_stream_placeholder = frame_placeholder_with_interval * (num_next_frames - 1)
                    next_intervals = torch.arange(len(frame_placeholder_with_interval), len(next_stream_placeholder)+1, len(frame_placeholder_with_interval)) - len(self.frame_interval)
                    if self.frame_interval: # last frame does not have frame interval
                        next_stream_placeholder = next_stream_placeholder[:-len(self.frame_interval)]
                    augmented += [
                        {'role': 'stream', 'content': self.v_placeholders_per_frame, 'intervals': torch.tensor([len(self.v_placeholders_per_frame)])},
                        correct_assistant,
                        {'role': 'stream', 'content': next_stream_placeholder, 'intervals': next_intervals}
                    ]
                else: # condition on video behavior
                    augmented += [
                        {'role': 'stream', 'content': conversation[i+1]['content']}
                    ]
            else:
                augmented += [conversation[i+1]]
            conversation = conversation[:i] + augmented + conversation[i+2:]
        return [random.choice(self.user_instructions)] + conversation

    # def __getitem__(self, index):
    #     anno = self.annos[index]
    #     return *super().__getitem__(
    #         conversation=anno['conversation'],
    #         load_ranges=anno['load_ranges'],
    #     ), index, self.evaluation_kwargs

def build_vstibench_goalstep_livechat_train(**kwargs):
    return VSTIBenchLiveChat(split='train', **kwargs)


def build_vstibench_goalstep_livechat_test(**kwargs):
    return VSTIBenchLiveChat(split='test', **kwargs)

if __name__ == '__main__':
    build_vstibench_goalstep_livechat_train(
        is_training=True, augmentation=False, embed_mark='2fps_384_1+3x3', system_prompt='', tokenizer=None,
        frame_fps=2, vision_pretrained='google/siglip-large-patch16-384'
    )
