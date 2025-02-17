#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

import time
import socket,os,struct, time

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

import argparse
import time
import socket,os,struct, time
import numpy as np

############################ Calvin crazy fly

import logging
import sys
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper
from cflib.utils.multiranger import Multiranger
URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

if len(sys.argv) > 1:
    URI = sys.argv[1]

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)
############################
# Args for setting IP/port of AI-deck. Default settings are for when
# AI-deck is in AP mode.
parser = argparse.ArgumentParser(description='Connect to AI-deck JPEG streamer example')
parser.add_argument("-n",  default="192.168.4.1", metavar="ip", help="AI-deck IP")
parser.add_argument("-p", type=int, default='5000', metavar="port", help="AI-deck port")
parser.add_argument('--save', action='store_true', help="Save streamed images")
args = parser.parse_args()

deck_port = args.p
deck_ip = args.n

print("Connecting to socket on {}:{}...".format(deck_ip, deck_port))
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((deck_ip, deck_port))
print("Socket connected")

imgdata = None
data_buffer = bytearray()

def rx_bytes(size):
  data = bytearray()
  while len(data) < size:
    data.extend(client_socket.recv(size-len(data)))
  return data

import cv2

start = time.time()
global count
count = 0




# def get_args():
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--device", type=int, default=0)
#     parser.add_argument("--width", help='cap width', type=int, default=960)
#     parser.add_argument("--height", help='cap height', type=int, default=540)

#     parser.add_argument('--use_static_image_mode', action='store_true')
#     parser.add_argument("--min_detection_confidence",
#                         help='min_detection_confidence',
#                         type=float,
#                         default=0.7)
#     parser.add_argument("--min_tracking_confidence",
#                         help='min_tracking_confidence',
#                         type=int,
#                         default=0.5)

#     args = parser.parse_args()

#     return args

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args, _ = parser.parse_known_args()  # Ignore unrecognized arguments

    return args



def main():

    ##########################crazyflie drivers initialization
    cflib.crtp.init_drivers()
    velocity_x = 0.0 #initialize for crazy fly
    velocity_y = 0.0
    Frame_avg =[]
    Frame_count = 0
    ###################

    # Argument parsing #################################################################
    global count
    args = get_args()
    #for saving video
    

    # cap_device = args.device
    # cap_width = args.width
    # cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    # cap = cv.VideoCapture(cap_device)
    # cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    # cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################CALVIN
    mode = 0
    cf = Crazyflie(rw_cache='./cache')
    with SyncCrazyflie(URI, cf=cf) as scf:
        with MotionCommander(scf, default_height = 0.8) as motion_commander:
            with Multiranger(scf) as multiranger:
                keep_flying = True
                while keep_flying:
                    ###############crazy fly sensors flying
                    VELOCITY = 0.5
                    velocity_x = 0.0
                    velocity_y = 0.0

                    if is_close(multiranger.front):
                        velocity_x -= VELOCITY
                        # print("Doing action F")
                    if is_close(multiranger.back):
                        velocity_x += VELOCITY
                        # print("Doing action B")
                    if is_close(multiranger.left):
                        velocity_y -= VELOCITY
                        # print("Doing action L")
                    if is_close(multiranger.right):
                        velocity_y += VELOCITY
                        # print("Doing action R")

                    # if is_close(multiranger.up):
                    #     keep_flying = False

                    motion_commander.start_linear_motion(
                        velocity_x, velocity_y, 0)

                    # time.sleep(0.1) maybe no delay? Calvin

                    ################################CALVIN

                    #First get the info
                    packetInfoRaw = rx_bytes(4)
                    #print(packetInfoRaw)
                    [length, routing, function] = struct.unpack('<HBB', packetInfoRaw)
                    #print("Length is {}".format(length))
                    #print("Route is 0x{:02X}->0x{:02X}".format(routing & 0xF, routing >> 4))
                    #print("Function is 0x{:02X}".format(function))

                    imgHeader = rx_bytes(length - 2)
                    #print(imgHeader)
                    #print("Length of data is {}".format(len(imgHeader)))
                    [magic, width, height, depth, format, size] = struct.unpack('<BHHBBI', imgHeader)

                    if magic == 0xBC:
                        #print("Magic is good")
                        #print("Resolution is {}x{} with depth of {} byte(s)".format(width, height, depth))
                        #print("Image format is {}".format(format))
                        #print("Image size is {} bytes".format(size))

                        # Now we start rx the image, this will be split up in packages of some size
                        imgStream = bytearray()

                        while len(imgStream) < size:
                            packetInfoRaw = rx_bytes(4)
                            [length, dst, src] = struct.unpack('<HBB', packetInfoRaw)
                            #print("Chunk size is {} ({:02X}->{:02X})".format(length, src, dst))
                            chunk = rx_bytes(length - 2)
                            imgStream.extend(chunk)
                        
                        count = count + 1
                        meanTimePerImage = (time.time()-start) / count
                        print("{}".format(meanTimePerImage))
                        print("{}".format(1/meanTimePerImage))
                        #   print("****This is Format*****",format) 
                        if format == 0:
                            bayer_img = np.frombuffer(imgStream, dtype=np.uint8)   
                            bayer_img.shape = (244, 324)
                            color_img = cv2.cvtColor(bayer_img, cv2.COLOR_BayerBG2BGRA)
                            cv2.imshow('Raw', bayer_img)
                            cv2.imshow('Color', color_img)
                            if args.save:
                                cv2.imwrite(f"stream_out/raw/img_{count:06d}.png", bayer_img)
                                cv2.imwrite(f"stream_out/debayer/img_{count:06d}.png", color_img)
                            cv2.waitKey(1)
                        else:
                            with open("img.jpeg", "wb") as f:
                                f.write(imgStream)
                            nparr = np.frombuffer(imgStream, np.uint8)
                            decoded = cv2.imdecode(nparr,cv2.IMREAD_UNCHANGED)
                            print("****decoded**** = ", decoded)
                            #cv2.imshow('JPEG', decoded)
                            #cv2.waitKey(1)
                    fps = cvFpsCalc.get()

                    # Process Key (ESC: end) #################################################
                    key = cv.waitKey(10)
                    if key == 27:  # ESC
                        break
                    number, mode = select_mode(key, mode)

                    # Camera capture #####################################################
                    # ret, image = cap.read()
                    # if not ret:
                    #     break
                    # image = cv.flip(image, 1)  # Mirror display
                    

                    # Detection implementation #############################################################
                    
                    image = decoded
                    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

                    #mirror image
                    image = cv.flip(image, 1)

                    #try resizing here
                    scale_factor = 1.5
                    image = cv.resize(image, None, fx=scale_factor, fy=scale_factor)

                    debug_image = copy.deepcopy(image)
                    print("image recieved")
                    image.flags.writeable = False
                    results = hands.process(image) #returns hand landmarks
                    image.flags.writeable = True

                    #  ####################################################################
                    if results.multi_hand_landmarks is not None:
                        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                            results.multi_handedness):

                            # print(hand_landmarks)
                            # Bounding box calculation
                            brect = calc_bounding_rect(debug_image, hand_landmarks)
                            # Landmark calculation
                            landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                            # Conversion to relative coordinates / normalized coordinates
                            pre_processed_landmark_list = pre_process_landmark(
                                landmark_list)

                            pre_processed_point_history_list = pre_process_point_history(
                                debug_image, point_history)
                            # Write to the dataset file
                            logging_csv(number, mode, pre_processed_landmark_list,
                                        pre_processed_point_history_list)

                            # Hand sign classification
                            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                            if hand_sign_id == 4:  # Point gesture, change to 2 for point history
                                point_history.append(landmark_list[8])
                            else:
                                point_history.append([0, 0])

                            # Finger gesture classification
                            finger_gesture_id = 0
                            point_history_len = len(pre_processed_point_history_list)
                            if point_history_len == (history_length * 2):
                                finger_gesture_id = point_history_classifier(
                                    pre_processed_point_history_list)

                            # Calculates the gesture IDs in the latest detection
                            finger_gesture_history.append(finger_gesture_id)
                            most_common_fg_id = Counter(
                                finger_gesture_history).most_common()

                            # Drawing part
                            debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                            debug_image = draw_landmarks(debug_image, landmark_list)
                            debug_image = draw_info_text(
                                debug_image,
                                brect,
                                handedness,
                                keypoint_classifier_labels[hand_sign_id], #keypoint_classifier_labels[hand_sign_id] Isolates the string passed to drawing function
                                point_history_classifier_labels[most_common_fg_id[0][0]],
                            )
                            
                            print("classification is this", keypoint_classifier_labels[hand_sign_id])
                            Frame_avg.append(hand_sign_id)
                            Frame_count = Frame_count + 1

                            if(Frame_count == 30):
                                SIGN = most_frequent(Frame_avg)
                            #   AMNA MOTION COMMANDER FUNCTIONS WITH GESTURE CLASSIFICATION
                                if SIGN == 0:  # UP
                                    motion_commander.up(0.5)
                                    print("Doing action UP")
                                elif SIGN == 1:  # DOWN
                                    motion_commander.down(0.5)
                                    print("Doing action DOWN")
                                elif SIGN == 2:  # LEFT
                                    motion_commander.right(0.5)
                                    # time.sleep(1)
                                    print("Doing action LEFT")
                                elif SIGN == 3:  # RIGHT
                                    motion_commander.left(0.5)
                                    print("Doing action RIGHT")
                                elif SIGN == 4:  # FORWARD
                                    motion_commander.forward(0.5)
                                    # time.sleep(1)
                                    print("Doing action FORWARD")
                                elif SIGN == 5:  # BACKWARD
                                    motion_commander.back(0.5)
                                    print("Doing action BACK")
                                elif SIGN == 6:  # STOP lands
                                    motion_commander.land()
                                    print("Doing action STOP")
                                
                                Frame_avg =[]
                                Frame_count = 0

                    debug_image = draw_point_history(debug_image, point_history)
                    debug_image = draw_info(debug_image, fps, mode, number)

                    

                    # Screen reflection #############################################################
                    cv.imshow('Hand Gesture Recognition', debug_image)


    # cap.release()
    cv.destroyAllWindows()
    print('Demo terminated!')#CALVIN


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list) #copy of pixel coordinates before normalizing

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)




    # if finger_gesture_text != "":
    #     cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
    #                cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    #     cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
    #                cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
    #                cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


def is_close(range): #CALVIN
    MIN_DISTANCE = 0.2  

    if range is None:
        return False
    else:
        return range < MIN_DISTANCE

def most_frequent(List): #CALVIN
    return max(set(List), key = List.count)

if __name__ == '__main__':
    main()