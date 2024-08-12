import cv2
import numpy as np


def read_txt_input_file(file_path):
    with open(file_path, 'rt') as f:
        lines = [line.rstrip() for line in f]
        return lines
    
def write_txt_file(str_lst, file_path):
    with open(file_path, 'a') as f:
        for i in str_lst:
            f.write(i + '\n')

def overwrite_txt_file(str_lst, file_path):
    with open(file_path, 'w+') as f:
        for i in str_lst:
            f.write(i + '\n')
    
def plot_profiles(profiles, max_val=20000000 , min_val=-20000000):
    max_h = 0       # red
    min_h = 120     # blue
    if not max_val:
        max_val = np.max(profiles)
    if not min_val:
        min_val = np.min(profiles)
    #print(max_val, min_val)
    heat_map_val = np.clip(profiles, min_val, max_val)
    heat_map = np.zeros(
        (heat_map_val.shape[0], heat_map_val.shape[1], 3), dtype=np.uint8)
    # print(heat_map_val.shape)
    heat_map[:, :, 0] = heat_map_val / \
        (max_val + 1e-6) * (max_h - min_h) + min_h
        #(max_val + 1e-6) * (max_h - min_h) + min_h
    heat_map[:, :, 1] = np.ones(heat_map_val.shape) * 255
    heat_map[:, :, 2] = np.ones(heat_map_val.shape) * 255
    heat_map = cv2.cvtColor(heat_map, cv2.COLOR_HSV2BGR)
    heat_map = cv2.cvtColor(heat_map, cv2.COLOR_BGR2RGB)
    return heat_map 


# if there is an overlap betwene the first and second frame, return the 
# index of the column that starts the overlap in the first frame.
# precondition: fst_frame has length shorter or equal to snd_frame
def check_overlap(fst_frame, snd_frame):
    for i in range(180):
        fst_overlap = fst_frame[:, i:]
        snd_overlap = snd_frame[: , : fst_overlap.shape[1]]
        if np.array_equal(fst_overlap, snd_overlap):
            return i
    return -1

def remove_overlap_snd_frame(fst_frame, snd_frame):
    overlap_index = check_overlap(fst_frame, snd_frame)
    if (overlap_index == -1):
        return snd_frame
    start_non_overlap_index_snd_frame = fst_frame.shape[1] - overlap_index
    return snd_frame[:, start_non_overlap_index_snd_frame:]

def concatenate_frames(frames):
    if len(frames) == 0:
        return np.zeros((100, 180))
    accumulate_frames = frames[0] #np.array
    previous_frame = frames[0] # this the the previous frame that has its overlap removed
    #pixel_time = [(179, frames_time[0])]
    for i in range(1, len(frames)):
        if check_overlap(previous_frame, frames[i]) == 0:
            continue
        removed_overlap = remove_overlap_snd_frame(previous_frame, frames[i])
        previous_frame = frames[i]
        accumulate_frames = np.concatenate((accumulate_frames, removed_overlap), axis = 1)
        #pixel_time.append((frames_time[i], accumulate_frames.shape[1] - 1))
    return accumulate_frames

def word_to_num(word):
    return [str(ord(c) - ord('a')) for c in word]

def get_first_col(phrase):
    return phrase.split(',')[0]

def get_col(phrase, col_num):
    return phrase.split(',')[col_num]