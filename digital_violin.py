import cv2
import mediapipe as mp
import pygame
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Init pygame mixer for sound
pygame.mixer.init()

notes = {
    "G3": pygame.mixer.Sound("notes/G3.wav"),
    "D4": pygame.mixer.Sound("notes/D4.wav"),
    "A4": pygame.mixer.Sound("notes/A4.wav"),
    "E5": pygame.mixer.Sound("notes/E5.wav"),
    "B4": pygame.mixer.Sound("notes/B4.wav"),
    "C5": pygame.mixer.Sound("notes/C5.wav"),
}
note_names = list(notes.keys())

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
h, w, _ = frame.shape

# Divide screen into vertical zones for left hand = note selection
num_zones = len(note_names)
zone_width = w // num_zones

current_note = None
playing_note = None
prev_right_y = None
movement_threshold = 4  # min pixel movement per frame to count as "bowing"
silence_frames = 0
silence_limit = 3  # frames of no movement before stopping sound

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = detector.detect(mp_image)

    left_hand = None
    right_hand = None

    if result.hand_landmarks and result.handedness:
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            label = handedness[0].category_name
            if label == "Left":
                right_hand = landmarks
            else:
                left_hand = landmarks

    # Draw zones
    for i, name in enumerate(note_names):
        x_start = i * zone_width
        cv2.rectangle(frame, (x_start, 0), (x_start + zone_width, h), (50, 50, 50), 1)
        cv2.putText(frame, name, (x_start + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Left hand selects note
    if left_hand:
        palm_x = int(left_hand[9].x * w)
        palm_y = int(left_hand[9].y * h)
        zone_index = min(palm_x // zone_width, num_zones - 1)
        current_note = note_names[zone_index]
        cv2.circle(frame, (palm_x, palm_y), 15, (0, 255, 255), -1)
        cv2.putText(frame, f"Note: {current_note}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Right hand = bow
    is_moving = False
    if right_hand:
        bow_x = int(right_hand[9].x * w)
        bow_y = int(right_hand[9].y * h)
        cv2.circle(frame, (bow_x, bow_y), 15, (0, 0, 255), -1)

        if prev_right_y is not None:
            movement = abs(bow_y - prev_right_y)
            if movement > movement_threshold:
                is_moving = True

        prev_right_y = bow_y
    else:
        prev_right_y = None

    # Handle note playing/stopping
    if is_moving and current_note:
        if playing_note != current_note:
            # Stop previous note, start new one
            if playing_note:
                notes[playing_note].stop()
            notes[current_note].play(loops=-1)
            playing_note = current_note
        silence_frames = 0
    else:
        silence_frames += 1
        if silence_frames > silence_limit and playing_note:
            notes[playing_note].stop()
            playing_note = None

    cv2.imshow("Digital Violin", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
if playing_note:
    notes[playing_note].stop()
cap.release()
cv2.destroyAllWindows()