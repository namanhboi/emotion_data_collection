import cv2
import argparse
import os
from datetime import datetime
import time
# import traceback
import threading
import numpy as np
from copy import deepcopy
import random
import sys
import shutil
import utils
from enum import Enum


asl_letter_res = 75

emo_lst = [
    "MM",
    "CS",
    "TH",
    "INTENSE",
    "PUFF",
    "PS"
]
resolution = (960, 540)
FPS = 30.0 

class SaveDataState(Enum):
    NO_WRITE_TO_BUF = 0
    WRITE_TO_BUF = 1



def hconcat_resize(img_list,  
                   interpolation  
                   = cv2.INTER_CUBIC): 
      # take minimum hights 
    h_min = min(img.shape[0]  
                for img in img_list) 
      
    # image resizing  
    im_list_resize = [cv2.resize(img, 
                       (int(img.shape[1] * h_min / img.shape[0]), 
                        h_min), interpolation 
                                 = interpolation)  
                      for img in img_list] 
      
    # return final image 
    return cv2.hconcat(im_list_resize) 


FPS = 30.0


def write_file(frame_buffer, clip_path, clip_name, frame_width, frame_height):
    try:
        four_cc = cv2.VideoWriter_fourcc(*'mp4v')
        vid = cv2.VideoWriter( os.path.join(clip_path, clip_name), four_cc, FPS, (frame_width, frame_height))
        print(frame_width, frame_height)
        for frame in frame_buffer:
            vid.write(frame)
        vid.release()
    except Exception as e:
        pass
        # traceback.print_exc()

def create_if_not_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)

def if_exist_delete_file(path):
    if os.path.exists(path):
        os.remove(path)

#this doesn't delete line in prediction
def delete_data_phrase_id(id, phrase):
    clip_file_path = os.path.join('./clips', f"clip_{id}_{phrase}.mp4")
    if_exist_delete_file(clip_file_path)

    
def write_signing_time(gnd_truth_file, phrases, id, start_time, end_time):
    phrases_plus_num = utils.read_txt_input_file(gnd_truth_file)
    phrases_plus_num[id] = f"{id},{start_time:.5f},{end_time:.5f},{phrases[id]},{end_time - start_time:.5f}"
    utils.overwrite_txt_file(phrases_plus_num, gnd_truth_file)

def create_emo_img(emotion, height = 200):
    if emotion in emo_lst:
        # print(emotion)
        img = cv2.imread(f'./emo_dataset/figures/{emotion}.png')
        return cv2.resize(img, (img.shape[1], height))
    return None

def read_emotion(emo_text):
    for i in emo_lst:
        if i in emo_text:
            return i
    return None

def data_record(path, gnd_truth_index):
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH , resolution[0])
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, FPS)

    karaoke_block_height = 200

    white_space_height = 400
    white_space_width = resolution[0]

    empty_white_space = np.zeros((white_space_height, white_space_width, 3), dtype = np.uint8)
    empty_white_space.fill(255)

    phrases_id = gnd_truth_index

    with open('./emo_dataset/emodataset.txt', 'r') as f:
        phrases = [line.rstrip() for line in f] 

    start_flag = False
    save_data_state = None

    frame_buffer = []


    emotion_img = None

    gnd_truth_file = os.path.join(path,'clips', 'gnd_truth.txt')
    utils.overwrite_txt_file(phrases, gnd_truth_file)

    start_time = 0
    exit_flag = False

    while cap.isOpened():
        if start_time == 0:
            start_time = time.time()
        fps_start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("All cameras off")
            break
        canvas = cv2.vconcat((empty_white_space, frame))
        # print(frame.shape, canvas.shape, empty_white_space.shape)
        if  not start_flag:
            cv2.putText(canvas, "Please press space to begin the session.", (10, karaoke_block_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 225), thickness = 2 )
            cv2.putText(canvas, "Once in the session, press space to move onto the next phrase", (10, karaoke_block_height + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 225), thickness = 2 )
            cv2.putText(canvas, "and x to restart the current word.", (10, karaoke_block_height + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 225), thickness = 2 )
            cv2.namedWindow("window", cv2.WINDOW_KEEPRATIO)
            cv2.imshow("window", canvas)
            pressed_key = cv2.waitKeyEx(1) & 0xFF
            if pressed_key == 27: 
                exit_flag = True
            elif pressed_key == ord(' ') or pressed_key == 3:
                start_flag = True
                last_cmd_time = time.time() - start_time
                save_data_state = SaveDataState.NO_WRITE_TO_BUF
        else:
            current_phrase = phrases[phrases_id]
            
            if save_data_state == SaveDataState.NO_WRITE_TO_BUF:
                cv2.putText(canvas, f"{phrases_id}/{len(phrases)}:", (10, karaoke_block_height + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), thickness = 2)
                cv2.putText(canvas, f"{current_phrase}", (200, karaoke_block_height + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), thickness = 2)
                cv2.putText(canvas, f"(NOT RECORDING)", (0, karaoke_block_height + 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness = 2)
                cv2.putText(canvas, f"Press space to start signing:", (10, karaoke_block_height + 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness = 2)
                print(read_emotion(current_phrase))
                emotion = read_emotion(current_phrase)
                if emotion is not None:
                    if emotion_img is None :
                        emotion_img = create_emo_img(emotion)
                    canvas[:emotion_img.shape[0], :emotion_img.shape[1]] = emotion_img
                else:
                    pass
                pressed_key = cv2.waitKeyEx(1) & 0xFF
                if last_cmd_time:
                    if pressed_key == 27:
                        exit_flag = True
                    elif (pressed_key == ord(' ') or pressed_key == 3):
                        print("space pressed")
                        timestamp_now = time.time() - start_time
                        previous_shared_pause = False
                        cnt_num_pauses = 0
                        last_cmd_time = timestamp_now  
                        frame_buffer.clear()
                        save_data_state = SaveDataState.WRITE_TO_BUF
                    elif pressed_key == ord('b'):
                        frame_buffer.clear()
                        last_cmd_time = time.time() - start_time
                        delete_data_phrase_id(phrase= phrases[phrases_id - 1], id=phrases_id - 1)
                        emotion_img = None
                        if not phrases_id == 0:
                            phrases_id -= 1                        
                        save_data_state = SaveDataState.NO_WRITE_TO_BUF
            elif save_data_state == SaveDataState.WRITE_TO_BUF:
                cv2.putText(canvas, f"{phrases_id}/{len(phrases)}:", (10, karaoke_block_height + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), thickness = 2)
                cv2.putText(canvas, f"{current_phrase}", (200, karaoke_block_height + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), thickness = 2)
                emotion = read_emotion(current_phrase)
                if emotion is not None:
                    if emotion_img is None :
                        emotion_img = create_emo_img(emotion)
                    canvas[:emotion_img.shape[0], :emotion_img.shape[1]] = emotion_img
                else:
                    pass
                frame_buffer.append(deepcopy(canvas))
                
                # Process user commands
                pressed_key = cv2.waitKeyEx(1) & 0xFF
                if last_cmd_time:
                    if pressed_key == 27:
                        frame_buffer.clear()
                        exit_flag = True
                    elif (pressed_key == ord(' ') or pressed_key == 3):
                        timestamp_now = time.time() - start_time
                        clip_file_name = f"sign_{phrases_id}_{current_phrase}.mp4"
                        print(clip_file_name)
                        copy_buffer = frame_buffer.copy()

                        write_signing_time_thread = threading.Thread(target=write_signing_time, args=(gnd_truth_file, phrases, phrases_id, last_cmd_time, timestamp_now))
                        write_signing_time_thread.start()

                        async_file_write = threading.Thread(target=write_file, args=(copy_buffer, os.path.join(path, 'clips'), clip_file_name, canvas.shape[1], canvas.shape[0]))
                        async_file_write.start()

                        last_cmd_time = timestamp_now  
                        frame_buffer.clear()
                        phrases_id += 1 
                        save_data_state = SaveDataState.NO_WRITE_TO_BUF
                        emotion_img = None
                    elif (pressed_key == ord('x')):
                        cv2.putText(canvas, f"Restart: {current_phrase}", (10, karaoke_block_height + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 225), thickness = 2)
                        frame_buffer.clear()
                        last_cmd_time = time.time() - start_time
                        save_data_state = SaveDataState.NO_WRITE_TO_BUF
                        emotion_img = None
                    elif pressed_key == ord('b'):
                        frame_buffer.clear()
                        last_cmd_time = time.time() - start_time
                        delete_data_phrase_id(phrase= phrases[phrases_id - 1], id=phrases_id - 1)
                        phrases_id -= 1
                        save_data_state = SaveDataState.NO_WRITE_TO_BUF
                        emotion_img = None
                if phrases_id == len(phrases):
                    exit_flag = True
        fps_end = time.time()
        cv2.namedWindow("window", cv2.WINDOW_KEEPRATIO)
        print(1 / (fps_end - fps_start))
        cv2.imshow('window', canvas)
        if exit_flag: 
            break
    print("Saving data, please wait 2 seconds...")
    time.sleep(2)
    cap.release()
    cv2.destroyAllWindows()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help = 'path to store the collected clips data', required=True)
    args = parser.parse_args()

    gnd_truth_start_index = 0

    if not os.path.exists(args.path):
        print("The -p path you specified doesn't exist.")
        sys.exit(1)
    if os.path.exists(os.path.join(args.path, 'clips')):
        a = input("The path you specified to store video clips already has a clips folder in it. Do you want to delete all files in this folder or resume progress [1/2]: ")
        while(a != "1" and a != "2"):
            a = input("Please enter only 1 or 2:")
        if (a == "1"):
            shutil.rmtree(os.path.join(args.path, 'clips'))
        else:
            gnd_truth_start_index = len(os.listdir(os.path.join(args.path, 'clips')))
    create_if_not_exists(os.path.join(args.path, 'clips'))
    data_record(args.path, gnd_truth_index=gnd_truth_start_index)
