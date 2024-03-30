import cv2
import mediapipe as mp

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize OpenCV
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image with Mediapipe Hands
    results = hands.process(frame_rgb)

    # Initialize finger counts for each hand
    finger_counts = [0, 0]

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw hand landmarks
            for lm in hand_landmarks.landmark:
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1)

            # Count fingers
            fingers = [8, 12, 16, 20]  # Landmark indices for thumb, index, middle, ring, and pinky finger tips
            finger_count = sum(1 for idx in fingers if hand_landmarks.landmark[idx].y < hand_landmarks.landmark[idx - 1].y)

            # Display finger count for each hand
            if i == 0:
                cv2.putText(frame, f'Hand 1 Finger Count: {finger_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f'Hand 2 Finger Count: {finger_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Update finger counts for each hand
            finger_counts[i] = finger_count

    # Display the frame
    cv2.imshow('Hand Tracking', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
